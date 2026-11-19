"""Randomized paired native-TTIR kernel timing, with A/A controls and raw samples."""

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
    compile_ir,
    kernel_info,
    launch,
    receipt,
    save_json,
    sha256,
    snapshot,
)


def summary(rows):
    ratios = [r["median_speedup"] for r in rows]
    return dict(
        arithmetic=statistics.mean(ratios),
        geometric=math.exp(statistics.mean(map(math.log, ratios))),
        worst=min(ratios),
        best=max(ratios),
        worst_n=rows[ratios.index(min(ratios))]["n"],
    )


def bootstrap(rows, seed=9182, count=5000):
    rng = random.Random(seed)
    draws = []
    for _ in range(count):
        medians = [
            statistics.median(
                rng.choices([s["speedup"] for s in r["pairs"]], k=len(r["pairs"]))
            )
            for r in rows
        ]
        draws.append(math.exp(statistics.mean(map(math.log, medians))))
    draws.sort()
    return [draws[int(count * 0.025)], draws[int(count * 0.975)]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+")
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--suite", choices=["tuning", "full"], default="full")
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and "baseline" not in args.names
    validation = json.loads(args.validation.read_text())
    pipeline = json.loads(args.pipeline.read_text())
    assert validation["passed"] and pipeline["passed"]
    assert pipeline["protocol_sha256"] == sha256(ROOT / "optimization_plan.md")
    names = ["baseline", *args.names]
    for name in names:
        assert validation["kernels"][name] == sha256(ROOT / f"ir/{name}.ttir")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    kernels = {name: compile_ir(name) for name in names}
    result = dict(
        **receipt(),
        metric="CUDA graph replay device kernel latency, preallocated output",
        cache="no clearing; repeated graphs",
        clocks_locked=False,
        validation_sha256=sha256(args.validation),
        pipeline_sha256=sha256(args.pipeline),
        suite=args.suite,
        rounds=args.rounds,
        rep_ms=20,
        seed=args.seed,
        artifacts={name: kernel_info(ck, name) for name, ck in kernels.items()},
        input_ttir={
            name: dict(path=f"ir/{name}.ttir", sha256=sha256(ROOT / f"ir/{name}.ttir"))
            for name in names
        },
        records=[],
        controls=[],
    )
    sizes = TUNING if args.suite == "tuning" else PERF_SIZES
    for n in sizes:
        x, y, out, _ = allocate(n)
        for ck in kernels.values():
            for _ in range(20):
                launch(ck, x, y, out)
            triton.testing.do_bench_cudagraph(partial(launch, ck, x, y, out), rep=20)
        torch.cuda.synchronize()
        pairs = {name: [] for name in args.names}
        controls = []
        for turn in range(args.rounds):
            candidates = [*args.names, "A/A"]
            rng.shuffle(candidates)
            for candidate in candidates:
                order = ["baseline", candidate]
                rng.shuffle(order)
                times = {}
                raw = {}
                for label in order:
                    ck = kernels["baseline" if label == "A/A" else label]
                    samples = triton.testing.do_bench_cudagraph(
                        partial(launch, ck, x, y, out), rep=20, return_mode="all"
                    )
                    assert samples and all(math.isfinite(t) and t > 0 for t in samples)
                    raw[label] = [t * 1000 for t in samples]
                    times[label] = statistics.mean(raw[label])
                row = dict(
                    round=turn,
                    order=order,
                    baseline_us=times["baseline"],
                    candidate_us=times[candidate],
                    speedup=times["baseline"] / times[candidate],
                    graph_replay_samples_us=raw,
                )
                (controls if candidate == "A/A" else pairs[candidate]).append(row)
        control_median = statistics.median(p["speedup"] for p in controls)
        result["controls"].append(
            dict(
                n=n,
                pairs=controls,
                median_speedup=control_median,
                passed=0.97 <= control_median <= 1.03,
            )
        )
        for name, samples in pairs.items():
            row = dict(
                n=n,
                name=name,
                pairs=samples,
                median_speedup=statistics.median(p["speedup"] for p in samples),
                baseline_us=statistics.median(p["baseline_us"] for p in samples),
                candidate_us=statistics.median(p["candidate_us"] for p in samples),
            )
            result["records"].append(row)
            print(f"N={n} {name}: {row['median_speedup']:.5f}x", flush=True)
        del x, y, out
    result["summaries"] = {}
    for name in args.names:
        rows = [r for r in result["records"] if r["name"] == name]
        result["summaries"][name] = dict(**summary(rows), bootstrap_95=bootstrap(rows))
    result["aa_passed"] = all(r["passed"] for r in result["controls"])
    result["gpu_snapshot_after"] = snapshot()
    save_json(args.output, result)
    print(json.dumps(result["summaries"], indent=2), flush=True)


if __name__ == "__main__":
    main()
