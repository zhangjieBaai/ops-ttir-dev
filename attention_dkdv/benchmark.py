"""Paired CUDA graph replay; correctness and exact IR hashes are mandatory gates."""

import argparse
import json
import random
import statistics
from pathlib import Path

import triton.testing
from common import (
    MATRIX,
    ROOT,
    allocate,
    compile_ir,
    digest,
    group,
    launch,
    save,
    snapshot,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--accuracy", required=True)
    p.add_argument("--selection")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--rep", type=int, default=10)
    p.add_argument("--seed", type=int, default=20)
    a = p.parse_args()
    out = Path(a.output)
    assert not out.exists()
    out.mkdir(parents=True)
    validation = json.loads(Path(a.accuracy).read_text())
    assert validation["passed"]
    candidates = json.loads((ROOT / "candidates.json").read_text())
    if a.selection:
        selection = json.loads(Path(a.selection).read_text())["tested_candidate"]
    else:
        selection = {g: list(v) for g, v in candidates.items()}
    result = dict(
        environment=snapshot(),
        seed=a.seed,
        rounds=a.rounds,
        rep_ms=a.rep,
        accuracy_sha256=digest(Path(a.accuracy)),
        timing="preallocated outputs, CUDA graph replay; excludes forward, K prescale, delta, Python and allocation",
        points=[],
    )
    random.seed(a.seed)
    kernels = {}
    for b, h, n, d, c in MATRIX:
        g = group(d, c)
        names = selection[g]
        if isinstance(names, str):
            names = [names]
        t = allocate(b, h, n, d, c, seed=a.seed)
        for name in ["baseline"] + names:
            key = (name, d, c)
            if key not in kernels:
                ck, meta = compile_ir(*key)
                old = validation["compile"][str(key)]
                assert (
                    old["status"] == "PASS"
                    and old["metadata"]["ttir_sha256"] == meta["ttir_sha256"]
                )
                kernels[key] = (ck, meta)

        def measure(name, rep=a.rep):
            return triton.testing.do_bench_cudagraph(
                lambda: launch(kernels[(name, d, c)][0], t), rep=rep
            )

        for name in ["baseline"] + names:
            measure(name, 100)
        aa = []
        for _ in range(3):
            first = measure("baseline")
            second = measure("baseline")
            aa.append(dict(first_ms=first, second_ms=second, ratio=first / second))
        point = dict(shape=(b, h, n, d, c), group=g, aa=aa, candidates={})
        for name in names:
            pairs = []
            for _ in range(a.rounds):
                order = ["baseline", name]
                random.shuffle(order)
                measured = {k: measure(k) for k in order}
                pairs.append(
                    dict(
                        order=order,
                        ms=measured,
                        speedup=measured["baseline"] / measured[name],
                    )
                )
            point["candidates"][name] = dict(
                pairs=pairs,
                median_speedup=statistics.median(x["speedup"] for x in pairs),
                baseline_median_us=1000
                * statistics.median(x["ms"]["baseline"] for x in pairs),
                candidate_median_us=1000
                * statistics.median(x["ms"][name] for x in pairs),
                baseline=kernels[("baseline", d, c)][1],
                candidate=kernels[(name, d, c)][1],
            )
            print(
                "bench",
                b,
                h,
                n,
                d,
                c,
                name,
                point["candidates"][name]["median_speedup"],
                flush=True,
            )
        result["points"].append(point)
        save(out / "benchmark.json", result)


if __name__ == "__main__":
    main()
