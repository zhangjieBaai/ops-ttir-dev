"""Native TTIR compiler and one shared launcher for dK/dV."""

import hashlib
import importlib.util
import json
import math
import os
import subprocess
from pathlib import Path

import torch
import triton

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "attention_environment", ROOT.parent / "layer_norm/environment.py"
)
env = importlib.util.module_from_spec(spec)
spec.loader.exec_module(env)
OPTIONS = dict(num_warps=4, num_stages=5, num_ctas=1, enable_fp_fusion=True)
MATRIX = [
    (b, h, n, d, c)
    for d in (64, 128)
    for c in (False, True)
    for b, h, n in ((1, 4, 256), (1, 4, 1024), (2, 8, 2048))
]


def digest(value):
    if isinstance(value, Path):
        value = value.read_bytes()
    if isinstance(value, str):
        value = value.encode()
    return hashlib.sha256(value).hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def group(d, c):
    return f"d{d}_c{int(c)}"


def snapshot():
    result = env.verify_distribution_files()
    result.update(
        torch=torch.__version__,
        gpu=subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                os.environ["CUDA_VISIBLE_DEVICES"],
                "--query-gpu=uuid,name,driver_version,clocks.sm,clocks.mem,power.draw,temperature.gpu",
                "--format=csv",
            ],
            text=True,
        ),
        environment={
            k: v
            for k, v in os.environ.items()
            if k.startswith(("CUDA", "TRITON", "LD_", "PYTHON"))
        },
        clocks_locked=False,
        cache_policy="Repeated graph warm cache; no flush",
    )
    return result


def compile_ir(name, d, c):
    env.check_environment()
    assert not triton.knobs.compilation.override
    assert triton.knobs.runtime.add_stages_inspection_hook is None
    path = ROOT / "ir" / group(d, c) / (name + ".ttir")
    ck = triton.compile(str(path), options=OPTIONS)
    ck._init_handles()
    dest = ROOT / "artifacts" / group(d, c) / name
    dest.mkdir(parents=True, exist_ok=True)
    for ext, data in ck.asm.items():
        if isinstance(data, bytes):
            (dest / f"kernel.{ext}").write_bytes(data)
        else:
            (dest / f"kernel.{ext}").write_text(data)
    meta = dict(
        ck.metadata._asdict(),
        n_regs=ck.n_regs,
        n_spills=ck.n_spills,
        ttir_sha256=digest(path),
        hashes={k: digest(v) for k, v in ck.asm.items()},
    )
    save(dest / "metadata.json", meta)
    return ck, meta


def allocate(b, h, n, d, c, distribution="normal", seed=20):
    torch.manual_seed(seed)
    shape = (b, h, n, d)
    q, k, v, do = [
        torch.randn(shape, device="cuda", dtype=torch.float16) * 0.5 for _ in range(4)
    ]
    if distribution == "zero_do":
        do.zero_()
    elif distribution == "constant":
        q.fill_(0.25)
        k.fill_(0.25)
    elif distribution == "large":
        q.mul_(3)
        k.mul_(3)
    scale = 1 / math.sqrt(d)
    kp = (k * (scale * 1.4426950408889634)).half()
    # Test-only saved statistics; use FP64 actual original Q/K attention.
    scores = q.double() @ k.double().transpose(-1, -2) * scale
    if c:
        mask = (
            torch.arange(n, device="cuda")[:, None]
            < torch.arange(n, device="cuda")[None, :]
        )
        scores.masked_fill_(mask, float("-inf"))
    probs = scores.softmax(-1)
    out = probs @ v.double()
    m = (scores.logsumexp(-1) / math.log(2)).float()
    delta = (out * do.double()).sum(-1).float()
    dp = do.double() @ v.double().transpose(-1, -2)
    ds = probs * (dp - (probs * dp).sum(-1, keepdim=True))
    rk = ds.transpose(-1, -2) @ q.double() * scale
    rv = probs.transpose(-1, -2) @ do.double()
    dk, dv, dq = [
        torch.full(shape, float("nan"), device="cuda", dtype=torch.float16)
        for _ in range(3)
    ]
    return dict(
        q=q,
        k=kp,
        original_k=k,
        v=v,
        do=do,
        dk=dk,
        dv=dv,
        dq=dq,
        m=m,
        delta=delta,
        scale=scale,
        causal=c,
        reference=(rk, rv),
        shape=shape,
    )


def source_args(t):
    q = t["q"]
    return (
        q,
        t["k"],
        t["v"],
        t["scale"],
        t["do"],
        t["dq"],
        t["dk"],
        t["dv"],
        t["m"],
        t["delta"],
        *q.stride(),
        q.shape[1],
        q.shape[2],
    )


def launch_source(kernel, t):
    q = t["q"]
    b, h, n, d = q.shape
    return kernel[(n // 128, 1, b * h)](
        *source_args(t),
        BLOCK_M1=32,
        BLOCK_N1=128,
        BLOCK_M2=128,
        BLOCK_N2=32,
        BLK_SLICE_FACTOR=2,
        HEAD_DIM=d,
        CAUSAL=t["causal"],
        **OPTIONS,
    )


def launch(ck, t):
    q = t["q"]
    b, h, n, d = q.shape
    # stride_d=1 is specialized out of the native runtime ABI.
    args = list(source_args(t))
    del args[13]
    ck[(n // 128, 1, b * h)](*args)
