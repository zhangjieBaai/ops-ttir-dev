"""Same-round original/parent/candidate timings on the expanded 32-point matrix."""

import argparse
import json
import random
import statistics
from pathlib import Path

import triton.testing
from common_round2 import (
    MATRIX,
    allocate,
    candidates,
    compile_ir,
    digest,
    group,
    input_hashes,
    launch,
    save,
    snapshot,
)
from warmup import GraphWarmup


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--accuracy", required=True)
    p.add_argument("--pipeline", required=True)
    p.add_argument("--selection")
    p.add_argument("--seed", type=int, default=20)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--rep", type=int, default=10)
    p.add_argument("--measure-warmup", type=int, default=100)
    a = p.parse_args()
    out = Path(a.output)
    assert not out.exists()
    out.mkdir(parents=True)
    accuracy = json.loads(Path(a.accuracy).read_text())
    pipeline = json.loads(Path(a.pipeline).read_text())
    assert accuracy["passed"] and pipeline["passed"] and len(pipeline["points"]) == 32
    requested = (
        candidates()
        if not a.selection
        else {
            g: [v]
            for g, v in json.loads(Path(a.selection).read_text())[
                "tested_candidate"
            ].items()
        }
    )
    names = {
        g: [name for name in gs if accuracy["valid"][g + "/" + name]]
        for g, gs in requested.items()
    }
    assert all(names.values()), names
    rng = random.Random(a.seed)
    result = {
        "environment": snapshot(),
        "seed": a.seed,
        "rounds": a.rounds,
        "rep_ms": a.rep,
        "warmup_ms": 500,
        "per_measurement_warmup_ms": a.measure_warmup,
        "warmup_protocol": (
            "v3: fixed1200MHz SM; reusable32-launch warmup graph with CUDA-event "
            "duration; timed do_bench_cudagraph unchanged"
        ),
        "accuracy_sha256": digest(Path(a.accuracy)),
        "pipeline_sha256": digest(Path(a.pipeline)),
        "points": [],
    }
    kernels = {}
    for shape in MATRIX:
        b, h, n, d, c = shape
        g = group(d, c)
        t = allocate(*shape, seed=a.seed)
        hashes = input_hashes(t)
        for name in ["baseline", "parent"] + names[g]:
            key = (name, d, c)
            if key not in kernels:
                kernels[key] = compile_ir(name, d, c)
            ck, meta = kernels[key]
            old = accuracy["compile"][str(key)]
            assert (
                old["status"] == "PASS" and old["metadata"]["hashes"] == meta["hashes"]
            )
            assert (
                meta["parsed_options"]
                == kernels[("baseline", d, c)][1]["parsed_options"]
            )
            checks = [
                r
                for r in accuracy["checks"]
                if r["shape"] == list(shape)
                and r["name"] == name
                and r["distribution"] == "normal"
            ]
            assert (
                len(checks) == 1
                and checks[0]["input_sha256"] == hashes
                and checks[0]["status"] == "PASS"
            )

        warmers = {}
        warmup_records = []
        for name in ["baseline", "parent"] + names[g]:
            ck = kernels[(name, d, c)][0]

            def warm_fn(ck=ck, tensors=t):
                launch(ck, tensors)

            warmers[name] = GraphWarmup(warm_fn)
            warmup_records.append(
                dict(route=name, phase="initial", **warmers[name].run(500))
            )

        def measure(name, rep=a.rep):
            def fn():
                launch(kernels[(name, d, c)][0], t)

            if a.measure_warmup:
                warmup_records.append(
                    dict(
                        route=name,
                        phase="before_measurement",
                        **warmers[name].run(a.measure_warmup)
                    )
                )
            return triton.testing.do_bench_cudagraph(fn, rep=rep) * 1000

        aa = []
        samples = {name: [] for name in names[g]}
        for turn in range(a.rounds):
            times = [measure("parent"), measure("parent")]
            aa.append({"round": turn, "us": times, "ratio": times[0] / times[1]})
            order_candidates = names[g].copy()
            rng.shuffle(order_candidates)
            for name in order_candidates:
                order = ["baseline", "parent", name]
                rng.shuffle(order)
                times = {k: measure(k) for k in order}
                samples[name].append(
                    {
                        "round": turn,
                        "order": order,
                        "us": times,
                        "baseline_over_candidate": times["baseline"] / times[name],
                        "parent_over_candidate": times["parent"] / times[name],
                        "baseline_over_parent": times["baseline"] / times["parent"],
                    }
                )
        row = {
            "shape": shape,
            "group": g,
            "input_sha256": hashes,
            "aa": aa,
            "aa_median": statistics.median(x["ratio"] for x in aa),
            "warmup_records": warmup_records,
            "candidates": {},
        }
        for name, values in samples.items():
            row["candidates"][name] = {
                "samples": values,
                "ratios": {
                    k: statistics.median(x[k] for x in values)
                    for k in [
                        "baseline_over_candidate",
                        "parent_over_candidate",
                        "baseline_over_parent",
                    ]
                },
                "median_us": {
                    k: statistics.median(x["us"][k] for x in values)
                    for k in ["baseline", "parent", name]
                },
                "metadata": {
                    k: kernels[(k, d, c)][1] for k in ["baseline", "parent", name]
                },
            }
        result["points"].append(row)
        save(out / "benchmark.json", result)
        print(
            "BENCH",
            shape,
            "AA",
            row["aa_median"],
            {k: v["ratios"] for k, v in row["candidates"].items()},
            flush=True,
        )
    result["aa_passed"] = all(
        0.97 <= row["aa_median"] <= 1.03 for row in result["points"]
    )
    result["passed"] = True
    save(out / "benchmark.json", result)


if __name__ == "__main__":
    main()
