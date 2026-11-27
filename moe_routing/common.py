"""Frozen Hopper routing baseline and one native TTIR launcher."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import frozen_kernels as source
import torch
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parent
PERF = [(m, e) for e in (256, 512) for m in (16, 128, 512, 4096)]
_spec = importlib.util.spec_from_file_location(
    "moe_environment", ROOT.parent / "layer_norm/environment.py"
)
_env = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_env)


def sha(data):
    if isinstance(data, Path):
        data = data.read_bytes()
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def immutable(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text() == text, f"Refusing to overwrite {path}"
    else:
        path.write_text(text)


def environment():
    result = _env.check_environment()
    assert not triton.knobs.compilation.override
    assert triton.knobs.runtime.add_stages_inspection_hook is None
    result.update(
        torch=torch.__version__,
        command=sys.argv,
        env={
            k: v
            for k, v in os.environ.items()
            if any(t in k for t in ("TRITON", "CUDA", "PATH", "LD_LIBRARY"))
        },
        gpu=subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.mem,"
                "power.draw,temperature.gpu,utilization.gpu",
                "--format=csv",
            ],
            text=True,
        ),
        libtriton_sha256=sha(Path(result["libtriton_path"])),
        backend_sha256=sha(
            Path(triton.__file__).parent / "backends/nvidia/compiler.py"
        ),
        ptxas_sha256=sha(Path(triton.__file__).parent / "backends/nvidia/bin/ptxas"),
    )
    return result


def options(warps):
    return dict(num_warps=warps, num_stages=3, num_ctas=1, enable_fp_fusion=True)


def allocate(m, e, distribution="normal", seed=0):
    assert m > 0 and m % 16 == 0 and e in (256, 512)
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m, e), generator=g, device="cuda")
    b = torch.randn((e,), generator=g, device="cuda") * 0.05
    if distribution == "ties":
        x.zero_()
        b.zero_()
    elif distribution == "group_ties":
        x.copy_(torch.arange(e, device="cuda").remainder(e // 8)[None, :] / 32)
        b.zero_()
    elif distribution == "bias_dominant":
        b.mul_(100)
    elif distribution == "saturated":
        x.mul_(100)
        b.zero_()
    elif distribution == "negative":
        x.fill_(-100)
        b.zero_()
    # Guard suffixes are disjoint from output views; no padding hidden in ABI.
    ws = torch.full((m * 8 + 32,), float("nan"), device="cuda")
    iw = torch.full((m * 8 + 32,), -666, device="cuda", dtype=torch.int32)
    gs = torch.full((m * 8 + 32,), float("nan"), device="cuda")
    return {
        "x": x,
        "b": b,
        "g": gs[: m * 8].view(m, 8),
        "w": ws[: m * 8].view(m, 8),
        "i": iw[: m * 8].view(m, 8),
        "guard_storage": (ws, iw, gs),
    }


def runtime_args(t):
    x, b, g, w, i = (t[k] for k in ("x", "b", "g", "w", "i"))
    m, e = x.shape
    return (
        (x, b, g, e // 8, 8, e, 8),
        (x, g, w, i, b, m, 8, 4, 8, e, e // 8, 1.0, e, 8, 8),
    )


def source_launch(t, warps):
    m, e = t["x"].shape
    a, b = runtime_args(t)
    c1 = source.topk_with_k2_triton[(m * 8,)](
        *a,
        scoring_func=1,
        BLOCK_SIZE=e // 8,
        INPUT_DTYPE=tl.float32,
        **options(4),
    )
    # scoring_func is a constexpr between scale and the three strides.
    c2 = source.group_idx_and_topk_triton[(m,)](
        *b[:12],
        1,
        *b[12:],
        N_GROUP=8,
        TOPK_GROUP=4,
        TOPK=8,
        BLOCK_GROUP=8,
        BLOCK_EXPERT=e,
        INPUT_DTYPE=tl.float32,
        renormalize=1,
        **options(warps),
    )
    return c1, c2


def launch(kernels, t):
    a, b = runtime_args(t)
    m = t["x"].shape[0]
    kernels[0][(m * 8, 1, 1)](*a)
    kernels[1][(m, 1, 1)](*b)


def group(e, warps):
    return f"e{e}_w{warps}"


def compile_pair(e, warps, variant="baseline"):
    _env.check_environment()
    assert not triton.knobs.compilation.override
    assert triton.knobs.runtime.add_stages_inspection_hook is None
    path = ROOT / "ir" / group(e, warps)
    result = []
    for stage, w in ((1, 4), (2, warps)):
        p = path / f"{variant if stage == 2 else 'baseline'}_s{stage}.ttir"
        c = triton.compile(str(p), options=options(w))
        c._init_handles()
        result.append(c)
    return tuple(result)


def artifacts(kernels, e, warps, variant):
    folder = ROOT / "artifacts" / group(e, warps) / variant
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for stage, c in enumerate(kernels, 1):
        for ext, data in c.asm.items():
            p = folder / f"stage{stage}.{ext}"
            p.write_bytes(data) if isinstance(data, bytes) else p.write_text(data)
        p = (
            ROOT
            / "ir"
            / group(e, warps)
            / f"{variant if stage == 2 else 'baseline'}_s{stage}.ttir"
        )
        records.append(
            dict(
                ir=str(p.relative_to(ROOT)),
                ir_sha256=sha(p),
                hashes={ext: sha(data) for ext, data in c.asm.items()},
                metadata=c.metadata._asdict(),
                regs=c.n_regs,
                spills=c.n_spills,
            )
        )
    return records


def reference(t):
    x, bias = t["x"].cpu(), t["b"].cpu()
    m, e = x.shape
    score = 0.5 * torch.tanh(0.5 * x) + 0.5
    selection = score + bias
    groupvals = selection.view(m, 8, e // 8).sort(descending=True, stable=True).values
    groups = groupvals[:, :, :2].sum(-1)
    groupids = groups.argsort(dim=-1, descending=True, stable=True)[:, :4]
    mask = torch.zeros((m, 8), dtype=torch.bool).scatter_(1, groupids, True)
    selected = selection.masked_fill(~mask.repeat_interleave(e // 8, -1), -float("inf"))
    ids = selected.argsort(dim=-1, descending=True, stable=True)[:, :8]
    vals = score.gather(1, ids)
    weights = vals / (vals.sum(-1, keepdim=True) + 1e-20)
    return weights, ids


def check_guards(t):
    ws, iw, gs = t["guard_storage"]
    assert torch.isnan(ws[-32:]).all()
    assert (iw[-32:] == -666).all()
    assert torch.isnan(gs[-32:]).all()
    assert torch.isfinite(t["w"]).all()
    assert torch.isfinite(t["g"]).all()
    assert (t["i"] >= 0).all() and (t["i"] < t["x"].shape[1]).all()
