"""Frozen MM contract and shared native TTIR compilation/launch path."""

import hashlib
import json
import os
from pathlib import Path

import torch
import triton
from environment import check_environment
from triton.backends.nvidia.compiler import CUDABackend

ROOT = Path(__file__).resolve().parent
SEED_SHAPE = (1024, 1024, 1024)
PERF_SHAPES = [
    (n, n, n) for n in [128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
] + [
    (256, 1024, 512),
    (1024, 256, 768),
    (128, 2048, 512),
    (2048, 128, 1024),
    (272, 400, 528),
]
ATOL, RTOL = 1e-2, 1e-3


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(
        value if isinstance(value, bytes) else value.encode()
    ).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        f.write(json.dumps(value, indent=2, default=str) + "\n")


def gate():
    check_environment()
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "Explicitly isolate one idle GPU"
    assert len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1
    assert not triton.knobs.compilation.override, "Disable kernel override"
    assert triton.knobs.runtime.add_stages_inspection_hook is None
    target = triton.runtime.driver.active.get_current_target()
    assert target.backend == "cuda" and target.arch == 90
    return target


def frozen():
    data = json.loads((ROOT / "baseline.json").read_text())
    assert sha256(ROOT / "tutorial.py") == data["source_sha256"]
    assert sha256(ROOT / "ir/baseline.ttir") == data["ttir_sha256"]
    assert sha256(ROOT / "optimization_plan.md") == data["contract_sha256"]
    assert sha256(ROOT / "common.py") == data["common_sha256"]
    for name, expected in data["protocol_sha256"].items():
        assert sha256(ROOT / name) == expected, f"Frozen protocol changed: {name}"
    return data


def compile_ir(name="baseline"):
    target = gate()
    config = frozen()
    path = ROOT / "ir" / f"{name}.ttir"
    options = config["options"]
    assert CUDABackend(target).parse_options(options).ir_override is None
    ck = triton.compile(str(path), options=options)
    assert ck.src.ext == "ttir" and ck.src.src == path.read_text()
    ck._init_handles()
    dump_kernel(ck, name)
    return ck


def dump_kernel(ck, name):
    out = ROOT / "artifacts" / name / ck.hash
    out.mkdir(parents=True, exist_ok=True)
    for ext, value in ck.asm.items():
        raw = value if isinstance(value, bytes) else value.encode()
        dest = out / f"kernel.{ext}"
        if dest.exists():
            assert dest.read_bytes() == raw
        else:
            dest.write_bytes(raw)
    return dict(
        hash=ck.hash,
        stages={ext: digest(value) for ext, value in ck.asm.items()},
        metadata=ck.metadata._asdict(),
        n_regs=ck.n_regs,
        n_spills=ck.n_spills,
    )


def allocate(m, n, k, pad=0):
    # Test-only data generation. Operator wrappers below do no PyTorch compute.
    a_storage = torch.randn((m, k + pad), device="cuda", dtype=torch.float16)
    b_storage = torch.randn((k, n + pad), device="cuda", dtype=torch.float16)
    c_storage = torch.full(
        (m, n + pad), float("nan"), device="cuda", dtype=torch.float16
    )
    return a_storage[:, :k], b_storage[:, :n], c_storage[:, :n], c_storage


def runtime_args(tensors):
    a, b, c = tensors[:3]
    m, k = a.shape
    n = b.shape[1]
    assert b.shape[0] == k and c.shape == (m, n)
    assert all(0 < d < 2**31 and d % 16 == 0 for d in (m, n, k))
    assert all(
        t.dtype == torch.float16 and t.device == a.device and t.is_cuda
        for t in (a, b, c)
    )
    assert all(t.stride(1) == 1 and t.stride(0) % 16 == 0 for t in (a, b, c))
    assert all(t.data_ptr() % 16 == 0 for t in (a, b, c))
    assert len({t.untyped_storage().data_ptr() for t in (a, b, c)}) == 3
    assert all(t.stride(0) >= t.shape[1] and t.numel() > 0 for t in (a, b, c))
    assert all(t.shape[0] * t.stride(0) < 2**31 for t in (a, b, c))
    return (a, b, c, m, n, k, a.stride(0), 1, b.stride(0), 1, c.stride(0), 1)


def grid(tensors, meta):
    m, n = tensors[2].shape
    return (
        triton.cdiv(m, meta["BLOCK_SIZE_M"]) * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
        1,
        1,
    )


def launch_source(tensors, config):
    from tutorial import matmul_kernel

    return matmul_kernel[grid(tensors, config["meta"])](
        *runtime_args(tensors), **config["meta"], **config["options"]
    )


def launch(ck, tensors, config):
    # Unit column strides are constexpr-specialized by Triton's source compiler.
    args = runtime_args(tensors)
    ck[grid(tensors, config["meta"])](*[args[i] for i in config["runtime_arg_indices"]])


def matmul(a, b, ck, config):
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float16)
    launch(ck, (a, b, c), config)
    return c
