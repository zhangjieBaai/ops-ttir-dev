"""Baseline correctness, stress distributions, padding and CPU FP64 references."""

import argparse
import json
from pathlib import Path

import torch
from common import (
    ATOL,
    PERF_SHAPES,
    ROOT,
    RTOL,
    allocate,
    compile_ir,
    frozen,
    launch,
    save_json,
    sha256,
)


def cases(suite):
    edges = [
        (16, 16, 16),
        (48, 80, 48),
        (112, 240, 16),
        (144, 272, 80),
        (1008, 1040, 496),
    ]
    if suite == "sanitizer":
        return [(shape, 16, "normal") for shape in edges]
    result = [(shape, 0, "normal") for shape in PERF_SHAPES]
    result += [(shape, pad, "normal") for shape in edges for pad in [0, 16]]
    result += [
        ((144, 272, 80), 16, distribution)
        for distribution in ["zero", "constant", "alternating", "large", "small"]
    ]
    # Cover both sides of every tile/group boundary for any selected tutorial config.
    meta = frozen()["meta"]
    bm, bn, bk, group = [
        meta[key]
        for key in ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M"]
    ]
    result += [
        ((bm + d, bn + d, bk + d), 16, "normal")
        for d in [-16, 0, 16]
        if min(bm, bn, bk) + d > 0
    ]
    result += [((bm * group + d, bn + 16, bk + 16), 16, "normal") for d in [-16, 0, 16]]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["full", "sanitizer"], default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    config = frozen()
    pipeline = json.loads(args.pipeline.read_text())
    assert pipeline["passed"] and pipeline["baseline_sha256"] == config["ttir_sha256"]
    assert pipeline["config_sha256"] == sha256(ROOT / "baseline.json")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.manual_seed(args.seed)
    ck = compile_ir()
    result = dict(
        passed=True,
        suite=args.suite,
        seed=args.seed,
        atol=ATOL,
        rtol=RTOL,
        baseline_sha256=config["ttir_sha256"],
        config_sha256=sha256(ROOT / "baseline.json"),
        validator_sha256=sha256(__file__),
        pipeline_sha256=sha256(args.pipeline),
        device_uuid=str(torch.cuda.get_device_properties(0).uuid),
        records=[],
    )
    for shape, pad, distribution in cases(args.suite):
        ts = allocate(*shape, pad=pad)
        a, b, c, storage = ts
        if distribution == "zero":
            a.zero_()
        elif distribution == "constant":
            a.fill_(0.25)
            b.fill_(0.25)
        elif distribution == "alternating":
            a[:, ::2].neg_()
            b[::2, :].copy_(b[1::2, :])
            a[:, 1::2].copy_(-a[:, ::2])
        elif distribution == "large":
            a.mul_(4)
            b.mul_(4)
        elif distribution == "small":
            a.mul_(1e-3)
            b.mul_(1e-3)
        before = [a.clone(), b.clone()]
        refs = {
            "native_fp16": torch.mm(a, b),
            "native_fp32": torch.mm(a.float(), b.float()).half(),
        }
        if max(shape) <= 272:
            refs["cpu_fp64"] = (
                torch.mm(a.cpu().double(), b.cpu().double()).half().cuda()
            )
        storage.fill_(float("nan"))
        launch(ck, ts, config)
        errors = {
            key: (c.float() - ref.float()).abs().max().item()
            for key, ref in refs.items()
        }
        passed = bool(torch.isfinite(c).all()) and bool(
            storage[:, shape[1] :].isnan().all()
        )
        passed &= all(torch.equal(x, y) for x, y in zip([a, b], before))
        passed &= all(
            torch.allclose(c, ref, atol=ATOL, rtol=RTOL) for ref in refs.values()
        )
        row = dict(
            shape=shape,
            padding=pad,
            distribution=distribution,
            max_abs_errors=errors,
            passed=passed,
        )
        result["records"].append(row)
        result["passed"] &= passed
        print(f"{shape} pad={pad} {distribution}: {errors} passed={passed}", flush=True)
    save_json(args.output, result)
    if not result["passed"]:
        raise SystemExit(1)
    print(f"PASS: {len(result['records'])} correctness cases")


if __name__ == "__main__":
    main()
