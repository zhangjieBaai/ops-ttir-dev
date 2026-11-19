"""Paired total two-kernel CUDA graph timing; raw samples and same-process A/A."""

import argparse
import json
import math
import random
import statistics
from functools import partial
from pathlib import Path

import torch
import triton.testing
from common import (
    PERF_SIZES,
    ROOT,
    TUNING,
    allocate,
    frozen,
    hashes,
    kernel_info,
    launch,
    load_kernels,
    receipt,
    save_json,
    sha256,
    snapshot,
)


def interval(samples, rng, statistic=statistics.median):
    draws = sorted(statistic(rng.choices(samples, k=len(samples))) for _ in range(2000))
    return [draws[50], draws[1950]]


def summarize(rows):
    ratios = [row["median_speedup"] for row in rows]
    rng = random.Random(917)
    draws = sorted(
        math.exp(
            statistics.mean(
                math.log(
                    statistics.median(
                        rng.choices(
                            [p["speedup"] for p in r["pairs"]], k=len(r["pairs"])
                        )
                    )
                )
                for r in rows
            )
        )
        for _ in range(2000)
    )
    return dict(
        arithmetic=statistics.mean(ratios),
        geometric=math.exp(statistics.mean(map(math.log, ratios))),
        bootstrap_95=[draws[50], draws[1950]],
        worst=min(ratios),
        best=max(ratios),
        worst_n=rows[ratios.index(min(ratios))]["n"],
        wins=sum(r > 1 for r in ratios),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+")
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--suite", choices=["tuning", "full"], default="full")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    assert args.rounds > 0
    validation = json.loads(args.validation.read_text())
    pipeline = json.loads(args.pipeline.read_text())
    manifest = frozen()
    assert validation["passed"] and pipeline["passed"]
    assert pipeline["protocol_sha256"] == sha256(ROOT / "optimization_plan.md")
    assert pipeline["manifest_sha256"] == sha256(ROOT / "baseline.json")
    names = list(dict.fromkeys(["baseline", *args.names]))
    for name in names:
        assert validation["kernels"][name] == hashes(name)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    kernels = {name: load_kernels(name) for name in names}
    for name in names:
        for group, ck in kernels[name].items():
            evidence = pipeline["artifacts"][name][group]
            assert evidence["input_sha256"] == hashes(name)[group]
            assert evidence["stages"]["cubin"] == kernel_info(ck)["stages"]["cubin"]
    result = dict(
        **receipt(),
        metric="preallocated two-kernel total CUDA graph replay device latency",
        cache="not cleared, repeated graphs",
        clocks_locked="not set by experiment; queried state in snapshots",
        validation_sha256=sha256(args.validation),
        pipeline_sha256=sha256(args.pipeline),
        suite=args.suite,
        rounds=args.rounds,
        rep_ms=20,
        seed=args.seed,
        artifacts={
            name: {
                g: kernel_info(
                    ck, (ROOT / manifest["groups"][g]["path"]).with_name(f"{name}.ttir")
                )
                for g, ck in items.items()
            }
            for name, items in kernels.items()
        },
        records=[],
        controls=[],
    )
    for n in TUNING if args.suite == "tuning" else PERF_SIZES:
        tensors = allocate(n)
        calls = {
            name: partial(launch, items, tensors, manifest)
            for name, items in kernels.items()
        }
        for call in calls.values():
            for _ in range(20):
                call()
            triton.testing.do_bench_cudagraph(call, rep=100)
        torch.cuda.synchronize()
        pairs = {name: [] for name in args.names}
        controls = []
        for turn in range(args.rounds):
            candidates = [*args.names, "A/A"]
            rng.shuffle(candidates)
            for candidate in candidates:
                order = ["reference", "candidate"]
                rng.shuffle(order)
                raw, times = {}, {}
                for role in order:
                    name = (
                        "baseline"
                        if role == "reference" or candidate == "A/A"
                        else candidate
                    )
                    samples = triton.testing.do_bench_cudagraph(
                        calls[name], rep=20, return_mode="all"
                    )
                    assert samples and all(math.isfinite(t) and t > 0 for t in samples)
                    raw[role] = [t * 1000 for t in samples]
                    times[role] = statistics.mean(raw[role])
                row = dict(
                    round=turn,
                    order=order,
                    baseline_us=times["reference"],
                    candidate_us=times["candidate"],
                    speedup=times["reference"] / times["candidate"],
                    replay_samples_us=raw,
                )
                (controls if candidate == "A/A" else pairs[candidate]).append(row)
        control = statistics.median(p["speedup"] for p in controls)
        result["controls"].append(
            dict(
                n=n,
                pairs=controls,
                median_speedup=control,
                passed=0.97 <= control <= 1.03,
            )
        )
        for name, samples in pairs.items():
            row = dict(
                n=n,
                name=name,
                groups=manifest["shapes"][str(n)],
                pairs=samples,
                median_speedup=statistics.median(p["speedup"] for p in samples),
                bootstrap_95=interval([p["speedup"] for p in samples], rng),
                baseline_us=statistics.median(p["baseline_us"] for p in samples),
                candidate_us=statistics.median(p["candidate_us"] for p in samples),
            )
            result["records"].append(row)
            print(
                f"n={n} {name} {row['baseline_us']:.4f}/{row['candidate_us']:.4f} us: {row['median_speedup']:.5f}x",
                flush=True,
            )
    result["summaries"] = {
        name: summarize([r for r in result["records"] if r["name"] == name])
        for name in args.names
    }
    result["aa_passed"] = all(row["passed"] for row in result["controls"])
    result["gpu_snapshot_after"] = snapshot()
    save_json(args.output, result)
    print(json.dumps(result["summaries"], indent=2), flush=True)


if __name__ == "__main__":
    main()
