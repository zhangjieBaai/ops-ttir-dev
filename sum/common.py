"""Frozen FP32 full-sum contract; one native TTIR compiler and two-stage launcher."""

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
import triton
from environment import check_environment
from triton.backends.nvidia.compiler import CUDABackend

ROOT = Path(__file__).resolve().parent
OPTIONS = dict(num_warps=4, num_ctas=1, num_stages=3, enable_fp_fusion=True)
PERF_SIZES = [
    1,
    15,
    16,
    31,
    32,
    63,
    64,
    127,
    128,
    255,
    256,
    511,
    1024,
    4096,
    16384,
    65535,
    65536,
    65537,
    262144,
    1048575,
    1048576,
    1048577,
    2097152,
    4194304,
    8388608,
    16777215,
    16777216,
    16777217,
    33554432,
    67108864,
]
TUNING = [
    16,
    127,
    1024,
    65535,
    65536,
    1048576,
    1048577,
    16777215,
    16777216,
    33554432,
    67108864,
]
ATOL, L1_FACTOR = 2e-5, 2e-6


def digest(value):
    return hashlib.sha256(
        value if isinstance(value, bytes) else value.encode()
    ).hexdigest()


def sha256(path):
    return digest(Path(path).read_bytes())


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, default=str) + "\n")


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text() == value, f"Immutable file changed: {path}"
    else:
        with path.open("x") as stream:
            stream.write(value)


def gate():
    check_environment()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    assert visible and "," not in visible, "Isolate a single idle GPU"
    assert not triton.knobs.compilation.override
    assert triton.knobs.runtime.add_stages_inspection_hook is None
    target = triton.runtime.driver.active.get_current_target()
    assert target.backend == "cuda" and target.arch == 90
    assert CUDABackend(target).parse_options(OPTIONS).ir_override is None
    return target


def geometry(n):
    block = triton.next_power_of_2(math.ceil(math.sqrt(n)))
    mid = triton.cdiv(n, block)
    return block, mid, triton.next_power_of_2(mid)


def frozen():
    data = json.loads((ROOT / "baseline.json").read_text())
    for name, expected in data["frozen_files"].items():
        assert sha256(ROOT / name) == expected, f"Frozen file changed: {name}"
    assert list(map(int, data["shapes"])) == PERF_SIZES
    for group in data["groups"].values():
        assert sha256(ROOT / group["path"]) == group["sha256"]
    return data


def snapshot():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,pci.bus_id,"
            "clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used",
            "--format=csv",
        ],
        text=True,
    )


def receipt():
    return dict(
        command=[sys.executable, *sys.argv],
        torch=torch.__version__,
        python=sys.executable,
        torch_path=torch.__file__,
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        protocol_sha256=sha256(ROOT / "optimization_plan.md"),
        manifest_sha256=sha256(ROOT / "baseline.json"),
        gpu_snapshot=snapshot(),
        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
    )


def allocate(n):
    _, size, _ = geometry(n)
    x = torch.randn(n, dtype=torch.float32, device="cuda")
    mid_storage = torch.full((size + 16,), 1234567.0, device="cuda")
    out_storage = torch.full((17,), 1234567.0, device="cuda")
    return x, mid_storage[:size], out_storage[:1], mid_storage, out_storage


def launch_source(tensors):
    from tutorial import sum_kernel_1, sum_kernel_2

    x, mid, out = tensors[:3]
    block, size, tail = geometry(x.numel())
    first = sum_kernel_1[(size, 1, 1)](x, mid, x.numel(), block, **OPTIONS)
    second = sum_kernel_2[(1, 1, 1)](mid, out, size, tail, **OPTIONS)
    return first, second


def kernel_info(ck, path=None):
    return dict(
        input_path=str(path) if path else None,
        input_sha256=sha256(path) if path else None,
        hash=ck.hash,
        stages={ext: digest(value) for ext, value in ck.asm.items()},
        metadata=ck.metadata._asdict(),
        registers=ck.n_regs,
        spills=ck.n_spills,
    )


def compile_ir(group, name="baseline"):
    gate()
    config = frozen()["groups"][group]
    path = (ROOT / config["path"]).with_name(f"{name}.ttir")
    ck = triton.compile(str(path), options=config["options"])
    assert ck.src.ext == "ttir" and ck.src.src == path.read_text()
    expected = CUDABackend(gate()).parse_options(config["options"])
    for key, value in vars(expected).items():
        actual = ck.metadata._asdict()[key]
        assert json.dumps(actual, default=str) == json.dumps(value, default=str), key
    ck._init_handles()
    out = ROOT / "artifacts" / group / name / ck.hash
    out.mkdir(parents=True, exist_ok=True)
    for ext, value in ck.asm.items():
        raw = value if isinstance(value, bytes) else value.encode()
        dest = out / f"kernel.{ext}"
        if dest.exists():
            assert dest.read_bytes() == raw
        else:
            dest.write_bytes(raw)
    return ck


def load_kernels(name):
    return {group: compile_ir(group, name) for group in frozen()["groups"]}


def hashes(name):
    return {
        g: sha256((ROOT / c["path"]).with_name(f"{name}.ttir"))
        for g, c in frozen()["groups"].items()
    }


def launch(kernels, tensors, manifest):
    x, mid, out = tensors[:3]
    n = x.numel()
    assert str(n) in manifest["shapes"], "Input has no frozen TTIR mapping"
    assert x.ndim == 1 and x.is_contiguous()
    assert all(
        t.dtype == torch.float32 and t.device == x.device and t.is_cuda
        for t in (x, mid, out)
    )
    assert all(t.data_ptr() % 16 == 0 and t.is_contiguous() for t in (x, mid, out))
    assert len({t.untyped_storage().data_ptr() for t in (x, mid, out)}) == 3
    _, size, _ = geometry(n)
    assert mid.numel() == size and out.numel() == 1
    for stage, full_args, grid in [
        (0, (x, mid, n), (size, 1, 1)),
        (1, (mid, out, size), (1, 1, 1)),
    ]:
        group = manifest["shapes"][str(n)][stage]
        config = manifest["groups"][group]
        kernels[group][grid](*[full_args[i] for i in config["runtime_indices"]])
