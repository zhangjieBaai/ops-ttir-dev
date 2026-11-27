"""Randomized complete-block factorial ablation using the frozen TTIR launcher."""

import argparse
import json
import math
import os
import random
import statistics
import subprocess
from functools import partial
from pathlib import Path

import torch
import triton.testing
from ablation_common import EDGES, NAMES
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
    receipt,
    save_json,
    sha256,
    snapshot,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accuracy", type=Path, required=True)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    assert not args.output.exists()
    accuracy = json.loads(args.accuracy.read_text())
    pipeline = json.loads(args.pipeline.read_text())
    design = json.loads((ROOT / "results/ablation_manifest.json").read_text())
    plan = sha256(ROOT / "ablation_plan.md")
    assert accuracy["passed"] and pipeline["passed"]
    assert design["plan_sha256"] == plan and design["variants"] == NAMES
    assert (
        accuracy["baseline_sha256"]
        == pipeline["baseline_sha256"]
        == sha256(ROOT / "baseline.json")
    )
    check_contention()
    groups = {group_for(shape)[0] for shape in PERF_SHAPES}
    kernels = {
        (group, name): compile_ir(group, name) for group in groups for name in NAMES
    }
    identities = {}
    for (group, name), ck in kernels.items():
        info = kernel_info(ck, group, name)
        for source in (accuracy, pipeline):
            expected = source["kernels"][group][name]
            assert info["input_sha256"] == expected["input_sha256"]
            assert info["stages"]["cubin"] == expected["stages"]["cubin"]
        identities.setdefault(group, {})[name] = info
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    result = dict(
        **receipt(),
        plan_sha256=plan,
        manifest_sha256=sha256(ROOT / "results/ablation_manifest.json"),
        accuracy_sha256=sha256(args.accuracy),
        pipeline_sha256=sha256(args.pipeline),
        names=NAMES,
        contrasts=EDGES,
        rounds=args.rounds,
        seed=args.seed,
        rep_ms=20,
        metric="CUDA graph replay device kernel latency; preallocated outputs",
        pairing="same randomized complete block; one timing per label; A/A baseline alias",
        clocks_locked="UNKNOWN; no setting command",
        cache="same-input replay; no flush",
        identities=identities,
        points=[],
    )
    telemetry_path = args.output.with_suffix(".telemetry.csv")
    with telemetry_path.open("x") as stream:
        monitor = subprocess.Popen(
            [
                "nvidia-smi",
                "-i",
                os.environ["CUDA_VISIBLE_DEVICES"],
                (
                    "--query-gpu=timestamp,uuid,clocks.sm,clocks.mem,temperature.gpu,power.draw,"
                    "utilization.gpu,clocks_event_reasons.active"
                ),
                "--format=csv",
                "-lms",
                "200",
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        try:
            for shape in PERF_SHAPES:
                check_contention()
                group, _ = group_for(shape)
                tensors = allocate(shape)
                calls = {
                    name: partial(launch, kernels[group, name], shape, tensors)
                    for name in NAMES
                }
                calls["A/A"] = calls["baseline"]
                for call in calls.values():
                    for _ in range(20):
                        call()
                    triton.testing.do_bench_cudagraph(call, rep=20)
                torch.cuda.synchronize()
                blocks = []
                for turn in range(args.rounds):
                    check_contention()
                    order = list(calls)
                    rng.shuffle(order)
                    samples = {}
                    for name in order:
                        raw = triton.testing.do_bench_cudagraph(
                            calls[name], rep=20, return_mode="all"
                        )
                        assert raw and all(math.isfinite(t) and t > 0 for t in raw)
                        samples[name] = [t * 1000 for t in raw]
                    times = {
                        name: statistics.mean(raw) for name, raw in samples.items()
                    }
                    blocks.append(
                        dict(
                            round=turn, order=order, samples_us=samples, times_us=times
                        )
                    )
                    check_contention()
                contrasts = {}
                for label, before, after in EDGES:
                    ratios = [
                        block["times_us"][before] / block["times_us"][after]
                        for block in blocks
                    ]
                    contrasts[label] = dict(
                        reference=before,
                        candidate=after,
                        paired_ratios=ratios,
                        median=statistics.median(ratios),
                    )
                aa = [
                    block["times_us"]["baseline"] / block["times_us"]["A/A"]
                    for block in blocks
                ]
                aa_median = statistics.median(aa)
                point = dict(
                    shape=shape,
                    group=group,
                    blocks=blocks,
                    contrasts=contrasts,
                    aa=dict(
                        paired_ratios=aa,
                        median=aa_median,
                        passed=0.97 <= aa_median <= 1.03,
                    ),
                    telemetry=snapshot(),
                )
                result["points"].append(point)
                save_json(
                    args.output.parent
                    / (args.output.stem + "_points")
                    / (key(shape) + ".json"),
                    point,
                )
                print(
                    key(shape),
                    "total",
                    round(contrasts["all_total"]["median"], 4),
                    "AA",
                    round(aa_median, 4),
                    "PASS",
                    point["aa"]["passed"],
                    flush=True,
                )
        finally:
            monitor.terminate()
            monitor.wait()
    result["aa_passed"] = all(point["aa"]["passed"] for point in result["points"])
    result["contention_after"] = check_contention()
    result["telemetry_sha256"] = sha256(telemetry_path)
    save_json(args.output, result)
    print("COMPLETE", len(result["points"]), "A/A", result["aa_passed"], flush=True)


if __name__ == "__main__":
    main()
