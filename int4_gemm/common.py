"""Frozen INT4 experiment inputs, native TTIR compilation and launch."""

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import torch
import triton

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent / "layer_norm"))
from environment import check_environment  # noqa: E402

SHAPES = [
    (16, 4096, 4096),
    (128, 11008, 4096),
    (1, 256, 128),
    (32, 1024, 1024),
    (64, 2048, 4096),
    (128, 4096, 4096),
    (17, 80, 256),
]
CONFIGS = [
    (32, 64, 64, 4, 3),
    (16, 64, 64, 4, 3),
    (32, 128, 64, 4, 3),
    (32, 64, 128, 4, 3),
]
GROUP = 128


def digest(value):
    if isinstance(value, str):
        value = value.encode()
    return hashlib.sha256(value).hexdigest()


def sha(path):
    return digest(Path(path).read_bytes())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), f"Evidence is immutable: {path}"
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text() == value, path
    else:
        path.write_text(value)


def case_id(shape):
    return "m%d_n%d_k%d" % tuple(shape)


def options(config):
    return dict(
        num_warps=config[3], num_stages=config[4], num_ctas=1, enable_fp_fusion=True
    )


def environment():
    result = check_environment()
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "Isolate GPU first"
    assert not triton.knobs.compilation.override
    assert triton.knobs.runtime.add_stages_inspection_hook is None
    assert triton.runtime.driver.active.get_current_target().arch == 90
    result.update(
        torch=torch.__version__,
        cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
        gpu=subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,clocks.sm,power.draw,temperature.gpu",
                "--format=csv",
            ],
            text=True,
        ),
        command=sys.argv,
        clocks_locked=False,
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    )
    return result


def allocate(shape, seed=42, mode="random"):
    torch.manual_seed(seed)
    m, n, k = shape
    a = torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.25
    packed = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    scale = torch.rand((k // GROUP, n), device="cuda", dtype=torch.float16) * 0.125
    zero = torch.rand_like(scale) * 15
    if mode == "zero":
        a.zero_()
    elif mode == "extreme_nibble":
        packed.fill_(240)
        zero.fill_(7.5)
    elif mode == "zero_scale":
        scale.zero_()
    elif mode == "constant":
        a.fill_(0.125)
        packed.fill_(136)
        zero.fill_(8)
    out_storage = torch.full(
        (m * n + 64,), float("nan"), device="cuda", dtype=torch.float16
    )
    out = out_storage[: m * n].view(m, n)
    return a, packed, scale, zero, out, out_storage


def runtime_args(ts):
    a, packed, scale, zero, out = ts[:5]
    m, k = a.shape
    n = out.shape[1]
    return (
        a,
        packed,
        scale,
        zero,
        out,
        m,
        n,
        k,
        k // 2,
        GROUP,
        k // GROUP,
        *a.stride(),
        *packed.stride(),
        *scale.stride(),
        *zero.stride(),
        *out.stride(),
    )


def grid(shape, config):
    return (triton.cdiv(shape[0], config[0]) * triton.cdiv(shape[1], config[1]), 1, 1)


def source_launch(ts, config):
    from tutorial import _weight_int4pack_mm_kernel

    shape = (ts[0].shape[0], ts[4].shape[1], ts[0].shape[1])
    return _weight_int4pack_mm_kernel[grid(shape, config)](
        *runtime_args(ts),
        BLOCK_M=config[0],
        BLOCK_N=config[1],
        BLOCK_K=config[2],
        **options(config),
    )


def launch(ck, ts, config):
    a, packed, scale, zero, out = ts[:5]
    shape = (a.shape[0], out.shape[1], a.shape[1])
    assert all(t.is_contiguous() for t in ts[:5])
    assert tuple(shape) in SHAPES and shape[2] % GROUP == 0
    names = [
        "A_ptr",
        "mat2_ptr",
        "qScale_ptr",
        "qZeros_ptr",
        "C_ptr",
        "M",
        "N",
        "K",
        "K2",
        "qGroupSize",
        "G",
        "stride_am",
        "stride_ak",
        "stride_mn",
        "stride_mk",
        "stride_sg",
        "stride_sn",
        "stride_zg",
        "stride_zn",
        "stride_cm",
        "stride_cn",
    ]
    values = dict(zip(names, runtime_args(ts)))
    header = next(
        line for line in ck.asm["ttir"].splitlines() if "tt.func public" in line
    )
    actual = re.findall(r"%([A-Za-z_]\w*):", header)
    ck[grid(shape, config)](*(values[name] for name in actual))


def compile_ir(case, name):
    check_environment()
    config = case["config"]
    path = ROOT / "ir" / case["id"] / (name + ".ttir")
    ck = triton.compile(str(path), options=options(config))
    ck._init_handles()
    assert ck.src.ext == "ttir"
    dest = ROOT / "artifacts" / case["id"] / name
    dest.mkdir(parents=True, exist_ok=True)
    for stage, value in ck.asm.items():
        p = dest / ("kernel." + stage)
        p.write_bytes(value if isinstance(value, bytes) else value.encode())
    return ck


def record(ck, path):
    return dict(
        ttir_sha256=sha(path),
        stages={stage: digest(v) for stage, v in ck.asm.items()},
        n_regs=ck.n_regs,
        n_spills=ck.n_spills,
        metadata=ck.metadata._asdict(),
    )


def manifest():
    return json.loads((ROOT / "baseline.json").read_text())


def reference(ts, cpu=False):
    a, packed, scale, zero = ts[:4]
    device = "cpu" if cpu else "cuda"
    a, packed, scale, zero = (t.to(device) for t in (a, packed, scale, zero))
    n, k2 = packed.shape
    unpacked = (
        torch.stack((packed & 15, packed >> 4), dim=2).reshape(n, 2 * k2).T.float()
    )
    weight = (
        (unpacked - zero.float().repeat_interleave(GROUP, 0))
        * scale.float().repeat_interleave(GROUP, 0)
    ).half()
    dtype = torch.float64 if cpu else torch.float32
    return (a.to(dtype) @ weight.to(dtype)).to(torch.float16)


def assert_output(ts, ref):
    torch.testing.assert_close(ts[4], ref.to(ts[4].device), rtol=0.003, atol=0.035)
    assert torch.isfinite(ts[4]).all()
    assert torch.isnan(ts[5][-64:]).all(), "output canary modified"
    return float((ts[4].float() - ref.to(ts[4].device).float()).abs().max())
