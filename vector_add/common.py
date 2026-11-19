"""One native TTIR compiler and launcher for the frozen vector-add contract."""

import hashlib
import json
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
PERF_SIZES = [2**i for i in range(12, 28)]
TUNING = [2**i for i in (12, 13, 16, 17, 20, 21, 24, 25)]
VALIDATION = [n for n in PERF_SIZES if n not in TUNING]
BOUNDARIES = [16, 32, 1008, 1024, 1040, 2032, 2048, 2064, 98432, 4194288, 4194320]


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


def gate():
    check_environment()
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    assert len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1
    assert not triton.knobs.compilation.override
    assert triton.knobs.runtime.add_stages_inspection_hook is None
    target = triton.runtime.driver.active.get_current_target()
    assert target.backend == "cuda" and target.arch == 90
    assert CUDABackend(target).parse_options(OPTIONS).ir_override is None
    return target


def manifest():
    data = json.loads((ROOT / "baseline.json").read_text())
    for name, expected in data["frozen_files"].items():
        assert sha256(ROOT / name) == expected, f"Frozen file changed: {name}"
    return data


def kernel_info(ck, name):
    dest = ROOT / "artifacts" / name / ck.hash
    dest.mkdir(parents=True, exist_ok=True)
    for stage, value in ck.asm.items():
        raw = value if isinstance(value, bytes) else value.encode()
        path = dest / f"kernel.{stage}"
        if path.exists():
            assert path.read_bytes() == raw
        else:
            path.write_bytes(raw)
    return dict(
        hash=ck.hash,
        stages={k: digest(v) for k, v in ck.asm.items()},
        metadata=ck.metadata._asdict(),
        n_regs=ck.n_regs,
        n_spills=ck.n_spills,
    )


def compile_ir(name="baseline"):
    target = gate()
    manifest()
    path = ROOT / "ir" / f"{name}.ttir"
    ck = triton.compile(str(path), options=OPTIONS)
    assert ck.src.ext == "ttir" and ck.src.src == path.read_text()
    options = CUDABackend(target).parse_options(OPTIONS)
    assert ck.metadata.target == target
    assert all(
        json.dumps(ck.metadata._asdict()[k]) == json.dumps(v)
        for k, v in vars(options).items()
    )
    ck._init_handles()
    kernel_info(ck, name)
    return ck


def runtime_args(x, y, out):
    n = x.numel()
    assert 0 < n <= 2**27 and n % 16 == 0
    assert all(
        t.ndim == 1 and t.numel() == n and t.is_contiguous() for t in (x, y, out)
    )
    assert all(
        t.is_cuda and t.device == x.device and t.dtype == torch.float32
        for t in (x, y, out)
    )
    assert all(t.data_ptr() % 16 == 0 for t in (x, y, out))
    assert len({t.untyped_storage().data_ptr() for t in (x, y, out)}) == 3
    return x, y, out, n


def launch(ck, x, y, out):
    args = runtime_args(x, y, out)
    ck[(triton.cdiv(args[-1], 1024), 1, 1)](*args)


def launch_source(x, y, out):
    from tutorial import add_kernel

    args = runtime_args(x, y, out)
    return add_kernel[(triton.cdiv(args[-1], 1024), 1, 1)](
        *args, BLOCK_SIZE=1024, **OPTIONS
    )


def allocate(n, guarded=False):
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn_like(x)
    storage = torch.full((n + 32 if guarded else n,), 123456.0, device="cuda")
    out = storage[16:-16] if guarded else storage
    return x, y, out, storage


def snapshot():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,pci.bus_id,clocks.sm,clocks.mem,"
            "temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used",
            "--format=csv",
        ],
        text=True,
    )


def receipt():
    return dict(
        command=[sys.executable, *sys.argv],
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        git_status=subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ),
        protocol_sha256=sha256(ROOT / "optimization_plan.md"),
        baseline_manifest_sha256=sha256(ROOT / "baseline.json"),
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpu_snapshot=snapshot(),
    )
