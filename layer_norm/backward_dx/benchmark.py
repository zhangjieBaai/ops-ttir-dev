"""Paired same-family dX-only kernel timings with A/A controls."""

import argparse
import math
import os
import random
import statistics
import subprocess

import torch
import triton.testing
from common import ROOT, allocate, block_for, compile_ir, digest, launch, save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("names", nargs="+")
    p.add_argument("--validation", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--rep", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--full", action="store_true")
    p.add_argument(
        "--profile",
        action="store_true",
        help="One verified C1-gated launch per kernel/shape, for NCU",
    )
    args = p.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    assert not os.path.exists(args.output)
    import json

    validation = json.load(open(args.validation))
    assert validation["passed"]
    names = ["baseline"] + [n for n in args.names if n != "baseline"]
    ns = (
        list(range(1024, 15873, 512))
        if args.full
        else [1024, 3072, 4096, 8192, 8704, 15872]
    )
    kernels = {}
    for n in ns:
        block = block_for(n)
        for name in names:
            key = (name, block)
            assert validation["kernels"][f"{name}/b{block}"] == digest(
                ROOT / f"ir/b{block}/{name}.ttir"
            )
            if key not in kernels:
                kernels[key] = compile_ir(name, block)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    result = dict(
        rounds=args.rounds,
        rep_ms=args.rep,
        seed=args.seed,
        timer="preallocated CUDA graph device time",
        device=os.environ["CUDA_VISIBLE_DEVICES"],
        validation_sha256=digest(args.validation),
        gpu_snapshot=subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,utilization.gpu,clocks.sm,temperature.gpu",
                "--format=csv",
            ],
            text=True,
        ),
        records=[],
        aa=[],
        kernels={
            f"{n}/b{b}": dict(
                ttir_sha256=digest(ROOT / f"ir/b{b}/{n}.ttir"),
                cubin_sha256=digest(ROOT / f"artifacts/b{b}/{n}/kernel.cubin"),
                n_regs=ck.n_regs,
                n_spills=ck.n_spills,
                metadata=ck.metadata._asdict(),
            )
            for (n, b), ck in kernels.items()
        },
    )
    if not args.profile:
        ts = allocate(4096, 8192)
        for name in names:
            ck = kernels[(name, 8192)]
            triton.testing.do_bench_cudagraph(lambda: launch(ck, ts), rep=50)
        del ts
    for n in ns:
        b = block_for(n)
        ts = allocate(4096, n)
        if args.profile:
            for name in names:
                print("PROFILE", name, n, flush=True)
                launch(kernels[(name, b)], ts)
            continue
        for name in names:
            for _ in range(10):
                launch(kernels[(name, b)], ts)
        torch.cuda.synchronize()
        samples = {name: [] for name in names if name != "baseline"}
        aa = []

        def measure(name):
            ck = kernels[(name, b)]
            return (
                triton.testing.do_bench_cudagraph(lambda: launch(ck, ts), rep=args.rep)
                * 1000
            )

        for turn in range(args.rounds):
            candidates = list(samples)
            rng.shuffle(candidates)
            aa_times = [measure("baseline"), measure("baseline")]
            aa.append(
                dict(round=turn, times_us=aa_times, ratio=aa_times[0] / aa_times[1])
            )
            for name in candidates:
                order = ["baseline", name]
                rng.shuffle(order)
                times = {label: measure(label) for label in order}
                samples[name].append(
                    dict(
                        round=turn,
                        order=order,
                        baseline_us=times["baseline"],
                        candidate_us=times[name],
                        speedup=times["baseline"] / times[name],
                    )
                )
        aa_mid = statistics.median(v["ratio"] for v in aa)
        result["aa"].append(
            dict(N=n, median=aa_mid, passed=0.97 <= aa_mid <= 1.03, samples=aa)
        )
        for name, values in samples.items():
            row = dict(
                kernel=name,
                M=4096,
                N=n,
                block=b,
                samples=values,
                baseline_us=statistics.median(v["baseline_us"] for v in values),
                candidate_us=statistics.median(v["candidate_us"] for v in values),
                speedup=statistics.median(v["speedup"] for v in values),
            )
            result["records"].append(row)
            print(
                name,
                n,
                f"{row['baseline_us']:.3f} -> {row['candidate_us']:.3f} us; {row['speedup']:.4f}x; AA={aa_mid:.4f}",
                flush=True,
            )
        save(args.output, result)
    result["summary"] = {}
    for name in names:
        if name == "baseline":
            continue
        ratios = [r["speedup"] for r in result["records"] if r["kernel"] == name]
        if ratios:
            result["summary"][name] = dict(
                geometric_mean=math.exp(statistics.mean(math.log(v) for v in ratios)),
                arithmetic_mean=statistics.mean(ratios),
                worst=min(ratios),
                best=max(ratios),
            )
    result["aa_passed"] = all(r["passed"] for r in result["aa"])
    save(args.output, result)
    print(result["summary"], flush=True)


if __name__ == "__main__":
    main()
