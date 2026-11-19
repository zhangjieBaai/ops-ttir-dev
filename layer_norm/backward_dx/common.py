"""One compiler and launcher for the extracted backward dX experiment."""

import hashlib
import importlib.util
import json
from pathlib import Path

import torch
import triton

ROOT = Path(__file__).resolve().parent
BLOCKS = [1024, 2048, 4096, 8192, 16384]
_spec = importlib.util.spec_from_file_location(
    "dx_environment", ROOT.parent / "environment.py"
)
_env = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_env)
check_environment = _env.check_environment
verify_environment = _env.verify_distribution_files


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def block_for(n):
    assert 0 < n <= 16384 and n % 16 == 0
    return max(1024, triton.next_power_of_2(n))


def options(block):
    return dict(
        num_warps=min(max(block // 256, 1), 8),
        num_ctas=1,
        num_stages=3,
        enable_fp_fusion=True,
    )


def compile_ir(name, block):
    check_environment()
    assert not triton.knobs.compilation.override
    assert triton.knobs.runtime.add_stages_inspection_hook is None
    path = ROOT / f"ir/b{block}/{name}.ttir"
    ck = triton.compile(str(path), options=options(block))
    ck._init_handles()
    folder = ROOT / f"artifacts/b{block}/{name}"
    folder.mkdir(parents=True, exist_ok=True)
    for ext, data in ck.asm.items():
        p = folder / f"kernel.{ext}"
        p.write_bytes(data) if isinstance(data, bytes) else p.write_text(data)
    save(
        folder / "metadata.json",
        dict(
            ck.metadata._asdict(),
            n_regs=ck.n_regs,
            n_spills=ck.n_spills,
            ttir_sha256=digest(path),
        ),
    )
    return ck


def allocate(m, n, pad=0, distribution="normal"):
    x = torch.randn(m, n + pad, device="cuda", dtype=torch.float16)[:, :n]
    # Create matching padded row strides explicitly.
    dy_storage = torch.randn(m, n + pad, device="cuda", dtype=x.dtype)
    dy = dy_storage[:, :n]
    if distribution == "normal":
        x.mul_(0.5).add_(-2.3)
    elif distribution == "offset":
        x.add_(1000)
    elif distribution == "constant":
        x.fill_(2)
    elif distribution == "lowvar":
        x.mul_(0.001).add_(1)
    if distribution == "zero_dy":
        dy.zero_()
    w = torch.randn(n, device="cuda", dtype=x.dtype)
    storage = torch.full((m, n + pad), float("nan"), device="cuda", dtype=x.dtype)
    dx = storage[:, :n]
    xf = x.double()
    mean = xf.mean(-1).float()
    rstd = torch.rsqrt(xf.var(-1, unbiased=False) + 1e-5).float()
    return (dx, dy, x, w, mean, rstd, storage)


def launch(ck, tensors):
    dx, dy, x, w, mean, rstd = tensors[:6]
    m, n = x.shape
    assert n % 16 == 0 and x.stride(0) % 16 == 0
    assert dx.stride() == dy.stride() == x.stride()
    ck[(m, 1, 1)](dx, dy, x, w, mean, rstd, x.stride(0), n)


def reference(tensors):
    dx, dy, x, w, mean, rstd = tensors[:6]
    xf = x.double()
    df = dy.double()
    wf = w.double()
    xhat = (xf - mean.double()[:, None]) * rstd.double()[:, None]
    wdy = df * wf
    ref = (
        wdy - (xhat * (xhat * wdy).mean(-1)[:, None] + wdy.mean(-1)[:, None])
    ) * rstd.double()[:, None]
    # Native autograd uses its own saved statistics; compare this independent path as well.
    native_x = x.detach().requires_grad_(True)
    out = torch.nn.functional.layer_norm(native_x, (x.shape[1],), w, None, 1e-5)
    native = torch.autograd.grad(out, native_x, dy)[0]
    return ref, native
