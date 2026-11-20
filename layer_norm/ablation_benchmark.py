"""Forward ablation timings with per-shape A/A controls."""

import argparse
import json
import math
import os
import random
import statistics
import subprocess

import torch
import triton
import triton.testing
from common import EPS, OPTIONS, ROOT, allocate, compile_ir, launch, save_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--rep", type=int, default=30, help="Target milliseconds per graph measurement"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stabilize-ms",
        type=int,
        default=0,
        help="Untimed graph warmup per kernel before the matrix",
    )
    parser.add_argument("--validation", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    assert not __import__("pathlib").Path(args.output).exists()
    assert os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    ), "Isolate a GPU explicitly before running"
    assert "baseline" not in args.names and len(set(args.names)) == len(args.names)
    validation = json.loads(open(args.validation).read())
    assert validation["passed"], "Correctness gate failed"
    for name in ["baseline"] + args.names:
        assert validation["kernels"][name] == sha256(
            ROOT / f"ir/{name}.ttir"
        ), "Validation does not match TTIR"
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    kernels = {name: compile_ir(name) for name in ["baseline"] + args.names}
    environment = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,clocks.sm,clocks.mem,"
            "temperature.gpu,power.draw,utilization.gpu",
            "--format=csv",
        ],
        text=True,
    )
    result = dict(
        timer="CUDA graph replay; preallocated output; no allocation or CPU launch in timed graph",
        stabilization_rep_ms=args.stabilize_ms,
        rounds=args.rounds,
        rep_ms=args.rep,
        seed=args.seed,
        options=OPTIONS,
        triton=triton.__version__,
        torch=torch.__version__,
        device=torch.cuda.get_device_name(),
        visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
        gpu_snapshot=environment,
        validation_sha256=sha256(args.validation),
        records=[],
        aa=[],
        kernels={
            name: dict(
                ttir_sha256=sha256(ROOT / f"ir/{name}.ttir"),
                cubin_sha256=sha256(ROOT / f"artifacts/{name}/kernel.cubin"),
                n_regs=ck.n_regs,
                n_spills=ck.n_spills,
            )
            for name, ck in kernels.items()
        },
    )
    if args.stabilize_ms:
        warm_tensors = allocate(4096, 8192)
        for ck in kernels.values():
            triton.testing.do_bench_cudagraph(
                lambda: launch(ck, warm_tensors, EPS), rep=args.stabilize_ms
            )
        torch.cuda.synchronize()
        del warm_tensors
        print("Untimed stabilization complete", flush=True)
    ns = (
        list(range(1024, 15873, 512))
        if args.full
        else [1024, 4096, 8192, 8704, 12288, 15872]
    )
    for n in ns:
        tensors = allocate(4096, n)
        tensors[0].mul_(0.5).add_(-2.3)
        for ck in kernels.values():
            for _ in range(10):
                launch(ck, tensors)
        torch.cuda.synchronize()
        samples = {name: [] for name in args.names}
        aa_samples = []
        for turn in range(args.rounds):
            aa_times = [
                triton.testing.do_bench_cudagraph(
                    lambda: launch(kernels["baseline"], tensors, EPS), rep=args.rep
                )
                * 1000
                for _ in range(2)
            ]
            aa_samples.append(
                dict(round=turn, times_us=aa_times, ratio=aa_times[0] / aa_times[1])
            )
            candidates = list(args.names)
            rng.shuffle(candidates)
            for name in candidates:
                order = ["baseline", name]
                rng.shuffle(order)
                times = {}
                for label in order:
                    ck = kernels[label]
                    ms = triton.testing.do_bench_cudagraph(
                        lambda: launch(ck, tensors, EPS), rep=args.rep
                    )
                    times[label] = ms * 1000
                samples[name].append(
                    dict(
                        round=turn,
                        order=order,
                        baseline_us=times["baseline"],
                        candidate_us=times[name],
                        speedup=times["baseline"] / times[name],
                    )
                )
        aa_ratio = statistics.median(v["ratio"] for v in aa_samples)
        result["aa"].append(
            dict(
                N=n,
                median=aa_ratio,
                passed=0.97 <= aa_ratio <= 1.03,
                samples=aa_samples,
            )
        )
        for name in args.names:
            pairs = samples[name]
            ratios = [s["speedup"] for s in pairs]
            row = dict(
                kernel=name,
                M=4096,
                N=n,
                samples=pairs,
                baseline_us=statistics.median(s["baseline_us"] for s in pairs),
                candidate_us=statistics.median(s["candidate_us"] for s in pairs),
                speedup=statistics.median(ratios),
                min_speedup=min(ratios),
                max_speedup=max(ratios),
            )
            result["records"].append(row)
            print(
                f"{name:24} N={n:5} {row['baseline_us']:8.3f} -> {row['candidate_us']:8.3f} us "
                f"{row['speedup']:.3f}x",
                flush=True,
            )
        save_json(args.output, result)
    result["aa_passed"] = all(v["passed"] for v in result["aa"])
    result["summary"] = {}
    for name in args.names:
        ratios = [r["speedup"] for r in result["records"] if r["kernel"] == name]
        result["summary"][name] = dict(
            arithmetic_mean=statistics.mean(ratios),
            geometric_mean=math.exp(statistics.mean(math.log(s) for s in ratios)),
            min=min(ratios),
            max=max(ratios),
        )
    save_json(args.output, result)
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
