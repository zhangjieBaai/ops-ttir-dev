"""Fixed-suite graph replay, randomized pairs, A/A and raw evidence."""

import argparse
import json
import math
import random
import statistics
import subprocess
from functools import partial
from pathlib import Path

import torch
import triton.testing
from common import (
    PERF_SHAPES,
    ROOT,
    allocate,
    check_contention,
    compile_ir,
    group_for,
    kernel_info,
    key,
    launch,
    manifest,
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
        worst_shape=rows[ratios.index(min(ratios))]["shape"],
    )


def bootstrap(rows, count=5000):
    rng = random.Random(9182)
    draws = []
    for _ in range(count):
        ratios = [
            statistics.median(
                rng.choices([s["speedup"] for s in r["pairs"]], k=len(r["pairs"]))
            )
            for r in rows
        ]
        draws.append(math.exp(statistics.mean(map(math.log, ratios))))
    draws.sort()
    return [draws[int(count * 0.025)], draws[int(count * 0.975)]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*")
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--suite", choices=["tuning", "full"], default="full")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    validation = json.loads(args.validation.read_text())
    pipeline = json.loads(args.pipeline.read_text())
    assert validation["passed"] and pipeline["passed"]
    assert (
        validation["baseline_sha256"]
        == pipeline["baseline_sha256"]
        == sha256(ROOT / "baseline.json")
    )
    assert (
        validation["protocol_sha256"]
        == pipeline["protocol_sha256"]
        == sha256(ROOT / "optimization_plan.md")
    )
    names = list(dict.fromkeys(["baseline", *args.names]))
    shapes = [
        s
        for s in PERF_SHAPES
        if args.suite == "full"
        or manifest()["mapping"][key(s)]["partition"] == "tuning"
    ]
    groups = {group_for(s)[0] for s in shapes}
    kernels = {(g, n): compile_ir(g, n) for g in groups for n in names}
    for (group, name), ck in kernels.items():
        assert (
            validation["kernels"][group][name]["input_sha256"]
            == kernel_info(ck, group, name)["input_sha256"]
        )
        assert (
            pipeline["kernels"][group]["baseline"]["stages"]["cubin"]
            == kernel_info(kernels[group, "baseline"], group, "baseline")["stages"][
                "cubin"
            ]
        )
    check_contention()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    result = dict(
        **receipt(),
        suite=args.suite,
        rounds=args.rounds,
        rep_ms=20,
        seed=args.seed,
        metric="graph_replay one kernel device latency, allocation/loading/submission excluded",
        cache="repeated same input; no clearing; no cold-cache claim",
        clocks_locked="UNKNOWN; not actively set",
        validation_sha256=sha256(args.validation),
        pipeline_sha256=sha256(args.pipeline),
        artifacts={
            g: {n: kernel_info(kernels[g, n], g, n) for n in names} for g in groups
        },
        records=[],
        controls=[],
        telemetry_during=[],
    )
    telemetry_path = args.output.with_suffix(".telemetry.csv")
    with telemetry_path.open("x") as stream:
        telemetry = subprocess.Popen(
            [
                "nvidia-smi",
                "-i",
                __import__("os").environ["CUDA_VISIBLE_DEVICES"],
                "--query-gpu=timestamp,uuid,clocks.sm,clocks.mem,temperature.gpu,power.draw,"
                "utilization.gpu,clocks_event_reasons.active",
                "--format=csv",
                "-lms",
                "200",
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        try:
            for shape in shapes:
                group, _ = group_for(shape)
                tensors = allocate(shape)
                for name in names:
                    ck = kernels[group, name]
                    for _ in range(20):
                        launch(ck, shape, tensors)
                    triton.testing.do_bench_cudagraph(
                        partial(launch, ck, shape, tensors), rep=100
                    )
                torch.cuda.synchronize()
                pairs = {name: [] for name in args.names}
                controls = []
                for turn in range(args.rounds):
                    candidates = [*args.names, "A/A"]
                    rng.shuffle(candidates)
                    for candidate in candidates:
                        order = ["baseline", candidate]
                        rng.shuffle(order)
                        times, raw = {}, {}
                        for label in order:
                            ck = kernels[group, "baseline" if label == "A/A" else label]
                            samples = triton.testing.do_bench_cudagraph(
                                partial(launch, ck, shape, tensors),
                                rep=20,
                                return_mode="all",
                            )
                            assert samples and all(
                                math.isfinite(t) and t > 0 for t in samples
                            )
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
                        (controls if candidate == "A/A" else pairs[candidate]).append(
                            row
                        )
                cm = statistics.median(p["speedup"] for p in controls)
                result["controls"].append(
                    dict(
                        shape=shape,
                        group=group,
                        pairs=controls,
                        median_speedup=cm,
                        passed=0.97 <= cm <= 1.03,
                    )
                )
                for name, samples in pairs.items():
                    row = dict(
                        shape=shape,
                        group=group,
                        name=name,
                        pairs=samples,
                        median_speedup=statistics.median(p["speedup"] for p in samples),
                        baseline_us=statistics.median(
                            p["baseline_us"] for p in samples
                        ),
                        candidate_us=statistics.median(
                            p["candidate_us"] for p in samples
                        ),
                    )
                    result["records"].append(row)
                    print(key(shape), name, f"{row['median_speedup']:.5f}x", flush=True)
                result["telemetry_during"].append(snapshot())
                print("A/A", key(shape), f"{cm:.5f}", flush=True)
                save_json(
                    args.output.parent
                    / (args.output.stem + "_points")
                    / (key(shape) + ".json"),
                    dict(
                        control=result["controls"][-1],
                        records=[r for r in result["records"] if r["shape"] == shape],
                    ),
                )
        finally:
            telemetry.terminate()
            telemetry.wait()
    result["summaries"] = {}
    for name in args.names:
        rows = [r for r in result["records"] if r["name"] == name]
        result["summaries"][name] = dict(
            **summary(rows),
            bootstrap_95=bootstrap(rows),
            by_path={
                path: summary(
                    [r for r in rows if (r["shape"][2] == 1) == (path == "inner")]
                )
                for path in ("inner", "non_inner")
            },
            validation=(
                summary(
                    [
                        r
                        for r in rows
                        if manifest()["mapping"][key(r["shape"])]["partition"]
                        == "validation"
                    ]
                )
                if args.suite == "full"
                else None
            ),
        )
    result["aa_passed"] = all(r["passed"] for r in result["controls"])
    result["contention_after"] = check_contention()
    result["gpu_snapshot_after"] = snapshot()
    result["telemetry_sha256"] = sha256(telemetry_path)
    save_json(args.output, result)
    print(
        json.dumps(result["summaries"], indent=2), "AA", result["aa_passed"], flush=True
    )


if __name__ == "__main__":
    main()
