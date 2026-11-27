"""Interleaved paired CUDA-graph measurement after exact-hash validation."""

import argparse
import json
import random
import statistics

import triton
from common import (
    ROOT,
    allocate,
    compile_ir,
    environment,
    launch,
    manifest,
    record,
    save,
    sha,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--rep", type=int, default=20)
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()
    env = environment()
    evidence = json.loads(open(args.validation).read())
    assert evidence["passed"]
    hashes = {
        (r["case"], r["name"]): r["kernel"]["ttir_sha256"] for r in evidence["checks"]
    }
    rng = random.Random(args.seed)
    results = []
    for case in manifest()["cases"]:
        ts = allocate(case["shape"], seed=args.seed)
        kernels = {n: compile_ir(case, n) for n in set(["baseline"] + args.names)}
        for n in kernels:
            assert hashes[case["id"], n] == sha(
                ROOT / "ir" / case["id"] / (n + ".ttir")
            )
            launch(kernels[n], ts, case["config"])
        for name in args.names:
            samples = []
            for ri in range(args.rounds):
                order = ["baseline", name]
                rng.shuffle(order)
                values = []
                for n in order:
                    ck = kernels[n]
                    ms = triton.testing.do_bench_cudagraph(
                        lambda: launch(ck, ts, case["config"]), rep=args.rep
                    )
                    values.append(dict(name=n, ms=ms))
                b = next(v["ms"] for v in values if v["name"] == "baseline")
                c = (
                    values[1]["ms"]
                    if name == "baseline"
                    else next(v["ms"] for v in values if v["name"] == name)
                )
                if name == "baseline":
                    b = values[0]["ms"]
                samples.append(
                    dict(round=ri, order=order, values=values, speedup=b / c)
                )
            entry = dict(
                case=case["id"],
                shape=case["shape"],
                name=name,
                config=case["config"],
                samples=samples,
                median_speedup=statistics.median(s["speedup"] for s in samples),
                kernels={
                    n: record(kernels[n], ROOT / "ir" / case["id"] / (n + ".ttir"))
                    for n in set(["baseline", name])
                },
            )
            results.append(entry)
            print(case["id"], name, entry["median_speedup"], flush=True)
    save(
        args.output,
        dict(
            environment=env,
            seed=args.seed,
            rounds=args.rounds,
            rep_ms=args.rep,
            validation_sha256=sha(args.validation),
            results=results,
        ),
    )


if __name__ == "__main__":
    main()
