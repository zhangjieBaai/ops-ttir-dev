"""Shared frozen compilation and launch contract. No source fallback for IR kernels."""

import hashlib
import json
from pathlib import Path

import torch
import triton
from environment import check_environment

ROOT = Path(__file__).resolve().parent
OPTIONS = dict(num_warps=8, num_ctas=1, num_stages=3, enable_fp_fusion=True)
BLOCK = 8192
EPS = 1e-5


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def compile_ir(name):
    check_environment()
    assert not triton.knobs.compilation.override, "Disable kernel overrides"
    assert triton.knobs.runtime.add_stages_inspection_hook is None
    path = ROOT / "ir" / f"{name}.ttir"
    ck = triton.compile(str(path), options=OPTIONS)
    out = ROOT / "artifacts" / name
    out.mkdir(parents=True, exist_ok=True)
    for ext, value in ck.asm.items():
        dest = out / f"kernel.{ext}"
        if isinstance(value, bytes):
            dest.write_bytes(value)
        else:
            dest.write_text(value)
    ck._init_handles()
    save_json(
        out / "metadata.json",
        dict(
            ck.metadata._asdict(),
            n_regs=ck.n_regs,
            n_spills=ck.n_spills,
            input_sha256=sha256(path),
        ),
    )
    return ck


def allocate(m, n, pad=0):
    # Match X/Y stride because the tutorial uses a shared stride argument.
    x = torch.randn((m, n + pad), device="cuda", dtype=torch.float16)[:, :n]
    y_storage = torch.full((m, n + pad), float("nan"), device="cuda", dtype=x.dtype)
    y = y_storage[:, :n]
    w = torch.rand(n, device="cuda", dtype=x.dtype)
    b = torch.rand_like(w)
    mean = torch.empty(m, device="cuda", dtype=torch.float32)
    rstd = torch.empty_like(mean)
    return x, y, w, b, mean, rstd, y_storage


def launch(ck, tensors, eps=EPS):
    x, y, w, b, mean, rstd = tensors[:6]
    assert x.shape[1] > 0 and x.shape[1] % 16 == 0 and x.stride(0) % 16 == 0
    assert y.stride() == x.stride()
    ck[(x.shape[0], 1, 1)](x, y, w, b, mean, rstd, x.stride(0), x.shape[1], eps)
