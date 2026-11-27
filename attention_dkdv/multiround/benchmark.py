"""Four-route paired device timing with a fixed round2 anchor."""

import argparse
import json
import random
import statistics
from pathlib import Path

import triton.testing
from common_mr import (
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
    a = p.parse_args()
    out = Path(a.output)
    assert not out.exists()
    out.mkdir(parents=True)
    acc = json.loads(Path(a.accuracy).read_text())
    pipe = json.loads(Path(a.pipeline).read_text())
    assert acc["passed"] and pipe["passed"] and len(pipe["points"]) == 32
    requested = (
        candidates()
        if not a.selection
        else {
            g: [n]
            for g, n in json.loads(Path(a.selection).read_text())[
                "tested_candidate"
            ].items()
        }
    )
    names = {
        g: [n for n in ns if acc["valid"][g + "/" + n]] for g, ns in requested.items()
    }
    assert all(names.values())
    result = {
        "environment": snapshot(),
        "seed": a.seed,
        "rounds": a.rounds,
        "rep_ms": a.rep,
        "warmup_protocol": "event-verified500ms initial and100ms each measurement; native do_bench_cudagraph",
        "accuracy_sha256": digest(Path(a.accuracy)),
        "pipeline_sha256": digest(Path(a.pipeline)),
        "points": [],
    }
    rng = random.Random(a.seed)
    kernels = {}
    for shape in MATRIX:
        b, h, n, d, c = shape
        g = group(d, c)
        t = allocate(*shape, seed=a.seed)
        hashes = input_hashes(t)
        routes = ["baseline", "anchor", "parent"] + names[g]
        for name in routes:
            key = (name, d, c)
            if key not in kernels:
                kernels[key] = compile_ir(name, d, c)
            ck, meta = kernels[key]
            assert meta["hashes"] == acc["compile"][str(key)]["metadata"]["hashes"]
            assert (
                meta["parsed_options"]
                == kernels[("baseline", d, c)][1]["parsed_options"]
            )
            records = [
                r
                for r in acc["checks"]
                if r["shape"] == list(shape)
                and r["name"] == name
                and r["distribution"] == "normal"
            ]
            assert (
                len(records) == 1
                and records[0]["input_sha256"] == hashes
                and records[0]["status"] == "PASS"
            )
        warmers = {}
        warm_records = []
        for name in routes:
            ck = kernels[(name, d, c)][0]

            def fn(ck=ck, tensors=t):
                launch(ck, tensors)

            warmers[name] = GraphWarmup(fn)
            warm_records.append(
                {"name": name, "initial": True, **warmers[name].run(500)}
            )

        def measure(name):
            warm_records.append(
                {"name": name, "initial": False, **warmers[name].run(100)}
            )
            return (
                triton.testing.do_bench_cudagraph(
                    lambda: launch(kernels[(name, d, c)][0], t), rep=a.rep
                )
                * 1000
            )

        samples, aa = [], []
        for turn in range(a.rounds):
            times = [measure("parent"), measure("parent")]
            aa.append({"round": turn, "times_us": times, "ratio": times[0] / times[1]})
            order = routes.copy()
            rng.shuffle(order)
            times = {route: measure(route) for route in order}
            samples.append(
                {
                    "round": turn,
                    "order": order,
                    "times_us": times,
                    "parent_over": {
                        route: times["parent"] / value for route, value in times.items()
                    },
                    "anchor_over": {
                        route: times["anchor"] / value for route, value in times.items()
                    },
                    "baseline_over": {
                        route: times["baseline"] / value
                        for route, value in times.items()
                    },
                }
            )
        row = {
            "shape": shape,
            "group": g,
            "input_sha256": hashes,
            "samples": samples,
            "aa": aa,
            "aa_median": statistics.median(x["ratio"] for x in aa),
            "warmup_records": warm_records,
            "metadata": {name: kernels[(name, d, c)][1] for name in routes},
            "median_us": {
                name: statistics.median(x["times_us"][name] for x in samples)
                for name in routes
            },
            "ratios": {
                metric: {
                    name: statistics.median(x[metric][name] for x in samples)
                    for name in routes
                }
                for metric in ("parent_over", "anchor_over", "baseline_over")
            },
        }
        result["points"].append(row)
        save(out / "benchmark.json", result)
        print(
            "BENCH",
            shape,
            "AA",
            row["aa_median"],
            row["ratios"]["parent_over"],
            flush=True,
        )
    result["passed"] = True
    result["aa_passed"] = all(0.97 <= x["aa_median"] <= 1.03 for x in result["points"])
    save(out / "benchmark.json", result)


if __name__ == "__main__":
    main()
