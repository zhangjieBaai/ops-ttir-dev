"""Compile and execute public-language witnesses, without feeding primary timings."""

import argparse
import inspect
import re
from pathlib import Path

import torch
import triton
import triton.language as tl
from ablation_common import NAMES
from common import (
    ALL_SHAPES,
    ROOT,
    allocate,
    check_contention,
    digest,
    gate,
    group_for,
    receipt,
    save_json,
    sha256,
)
from triton_syntax_ablation import public_mean
from validate import DISTRIBUTIONS, prepare


def config(name, n):
    result = dict(
        FLAT=True, INDEX64=False, UNROLL=1, RANK2=False, GUARD=False, TAIL=False
    )
    if name == "baseline":
        result.update(FLAT=False, INDEX64=True)
    elif name.startswith("ab_f"):
        result.update(FLAT="f1" in name, INDEX64="i0" in name)
    else:
        result["RANK2"] = True
        if name.startswith("ab_u"):
            u, g, t = map(int, re.fullmatch(r"ab_u([01])g([01])t([01])", name).groups())
            result.update(UNROLL=16 if u else 1, GUARD=bool(g), TAIL=bool(t))
            if n == 1:
                result.update(UNROLL=1, GUARD=False, TAIL=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gate()
    check_contention()
    torch.manual_seed(734)
    points = []
    kernels = {}
    for shape in ALL_SHAPES:
        m, n, k = shape
        if k < 1024:
            continue
        group, info = group_for(shape)
        for distribution in DISTRIBUTIONS:
            tensors = allocate(shape, guarded=True)
            x, y, storage = tensors
            prepare(x, distribution)
            saved = x.clone()
            ref = torch.mean(x.cpu().double(), dim=1).half().float().cuda()
            aten = x.mean(dim=1).float()
            for name in NAMES:
                y.fill_(float("nan"))
                ck = public_mean[(m, triton.cdiv(k, 1024), 1)](
                    y, x, m, n, k, **config(name, n), **info["options"]
                )
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    y.float(), ref, rtol=2e-3, atol=2e-3, equal_nan=True
                )
                torch.testing.assert_close(
                    y.float(), aten, rtol=2e-3, atol=2e-3, equal_nan=True
                )
                assert torch.isfinite(y[torch.isfinite(ref)]).all()
                assert (storage[:16] == 12345).all() and (storage[-16:] == 12345).all()
                torch.testing.assert_close(x, saved, rtol=0, atol=0, equal_nan=True)
                finite = torch.isfinite(ref)
                error = (
                    float((y.float()[finite] - ref[finite]).abs().max())
                    if finite.any()
                    else 0
                )
                points.append(
                    dict(
                        shape=shape,
                        name=name,
                        distribution=distribution,
                        passed=True,
                        max_abs_error_fp64_rounded=error,
                        source_kernel_hash=ck.hash,
                    )
                )
                identity = group + "/" + name
                if identity not in kernels:
                    ck._init_handles()
                    folder = ROOT / "artifacts/syntax_witness" / group / name / ck.hash
                    folder.mkdir(parents=True, exist_ok=True)
                    for stage, data in ck.asm.items():
                        raw = data if isinstance(data, bytes) else data.encode()
                        dest = folder / f"kernel.{stage}"
                        if dest.exists():
                            assert dest.read_bytes() == raw
                        else:
                            dest.write_bytes(raw)
                    kernels[identity] = dict(
                        hash=ck.hash,
                        options=ck.metadata._asdict(),
                        config=config(name, n),
                        stages={stage: digest(data) for stage, data in ck.asm.items()},
                        static_ttir_features=dict(
                            loads=ck.asm["ttir"].count("tt.load"),
                            loops=ck.asm["ttir"].count("scf.for"),
                            branches=ck.asm["ttir"].count("scf.if"),
                            i64_mentions=ck.asm["ttir"].count("i64"),
                        ),
                        n_regs=ck.n_regs,
                        n_spills=ck.n_spills,
                        ttir_static_loads=ck.asm["ttir"].count("tt.load"),
                        llvm_integer64_ops=len(
                            re.findall(r"\b(?:add|mul) i64\b", ck.asm["llir"])
                        ),
                    )
            print("syntax", shape, distribution, "PASS", flush=True)
    api_sources = {}
    for api in (tl.arange, tl.static_range, tl.sum, triton.jit):
        path = inspect.getfile(getattr(api, "fn", api))
        api_sources[getattr(api, "__name__", str(api))] = dict(
            path=path, sha256=sha256(path)
        )
    check_contention()
    save_json(
        args.output,
        dict(
            **receipt(),
            passed=True,
            source_sha256=sha256(ROOT / "triton_syntax_ablation.py"),
            runner_sha256=sha256(__file__),
            plan_sha256=sha256(ROOT / "ablation_plan.md"),
            scope="public-language semantic witnesses only; not timed, no binary equivalence claim",
            api_sources=api_sources,
            kernels=kernels,
            records=points,
        ),
    )
    print("PASS public syntax", len(points), "checks", flush=True)


if __name__ == "__main__":
    main()
