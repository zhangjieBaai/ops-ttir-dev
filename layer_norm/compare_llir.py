"""Compare retained LLIR/TTIR binaries under one paired graph timing protocol."""

import argparse
import importlib.util
import json
import math
import os
import random
import statistics
import subprocess
from pathlib import Path

import torch
import triton.testing
from common import EPS, ROOT, allocate, compile_ir, launch, save_json, sha256
from finalize import text_section_hashes

OLD = Path("/root/ir-ops-dev/layer_norm/ir_opt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--rep", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    assert not args.output.exists(), "Choose a new output to preserve evidence"
    spec = importlib.util.spec_from_file_location(
        "retained_llir_bench", OLD / "bench_ir.py"
    )
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    old_base = old.make_kernel_from_cubin(str(OLD / "baseline.cubin"))
    old_opt = old.make_kernel_from_cubin(str(OLD / "opt_cached.cubin"))
    new_base = compile_ir("baseline")
    new_opt = compile_ir("tile_dispatch_math_v2")

    def legacy(fn, ts):
        x, y, w, b, mean, rstd = ts[:6]
        fn(x.shape[0], x, y, w, b, mean, rstd, x.stride(0), x.shape[1], EPS, 8192, 8)

    functions = {
        "llir_baseline": lambda ts: legacy(old_base, ts),
        "llir_optimized": lambda ts: legacy(old_opt, ts),
        "ttir_baseline": lambda ts: launch(new_base, ts),
        "ttir_optimized": lambda ts: launch(new_opt, ts),
    }
    binaries = {
        "llir_baseline": OLD / "baseline.cubin",
        "llir_optimized": OLD / "opt_cached.cubin",
        "ttir_baseline": ROOT / "artifacts/baseline/kernel.cubin",
        "ttir_optimized": ROOT / "artifacts/tile_dispatch_math_v2/kernel.cubin",
    }
    result = dict(
        rounds=args.rounds,
        rep_ms=args.rep,
        visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
        timer="Same input and preallocated output; randomized four-kernel blocks; CUDA graph replay",
        script_sha256=sha256(__file__),
        records=[],
        accuracy=[],
        binaries={
            label: dict(
                path=str(path),
                sha256=sha256(path),
                text=text_section_hashes(path),
                resource_usage=subprocess.check_output(
                    ["cuobjdump", "--dump-resource-usage", str(path)], text=True
                ),
            )
            for label, path in binaries.items()
        },
    )
    torch.manual_seed(2026)
    rng = random.Random(2026)
    # Validate every binary on all 30 benchmark shapes before any measurements.
    for n in range(1024, 15873, 512):
        ts = allocate(4096, n)
        x, y, w, b, mean, rstd = ts[:6]
        x.mul_(0.5).add_(-2.3)
        ref = torch.nn.functional.layer_norm(x, (n,), w, b, EPS)
        xf = x.double()
        mean_ref = xf.mean(-1).float()
        rstd_ref = torch.rsqrt(xf.var(-1, unbiased=False) + EPS).float()
        del xf
        for label, fn in functions.items():
            y.fill_(float("nan"))
            mean.fill_(float("nan"))
            rstd.fill_(float("nan"))
            fn(ts)
            error = (y.float() - ref.float()).abs().max().item()
            passed = error < 1e-2 and torch.allclose(
                mean, mean_ref, rtol=2e-5, atol=1e-5
            )
            passed &= torch.allclose(rstd, rstd_ref, rtol=2e-4, atol=1e-5)
            result["accuracy"].append(
                dict(kernel=label, N=n, y_error=error, passed=bool(passed))
            )
            assert passed, (label, n, error)
    print("Accuracy: 120 kernel cases PASS", flush=True)
    for n in range(1024, 15873, 512):
        ts = allocate(4096, n)
        ts[0].mul_(0.5).add_(-2.3)
        for fn in functions.values():
            for _ in range(10):
                fn(ts)
        torch.cuda.synchronize()
        blocks = []
        for turn in range(args.rounds):
            order = list(functions)
            rng.shuffle(order)
            times = {}
            for label in order:
                fn = functions[label]
                times[label] = (
                    triton.testing.do_bench_cudagraph(lambda: fn(ts), rep=args.rep)
                    * 1000
                )
            blocks.append(
                dict(
                    round=turn,
                    order=order,
                    times_us=times,
                    llir_speedup=times["llir_baseline"] / times["llir_optimized"],
                    ttir_speedup=times["ttir_baseline"] / times["ttir_optimized"],
                    baseline_ratio=times["llir_baseline"] / times["ttir_baseline"],
                    optimized_ratio=times["ttir_optimized"] / times["llir_optimized"],
                )
            )
        row = dict(
            N=n,
            M=4096,
            samples=blocks,
            times_us={
                label: statistics.median(b["times_us"][label] for b in blocks)
                for label in functions
            },
        )
        for key in [
            "llir_speedup",
            "ttir_speedup",
            "baseline_ratio",
            "optimized_ratio",
        ]:
            row[key] = statistics.median(b[key] for b in blocks)
        result["records"].append(row)
        save_json(args.output, result)
        print(
            f"N={n:5}: LLIR {row['llir_speedup']:.3f}x; TTIR {row['ttir_speedup']:.3f}x; "
            f"baseline ratio {row['baseline_ratio']:.3f}; TTIRopt/LLIRopt {row['optimized_ratio']:.3f}",
            flush=True,
        )
    result["summary"] = {}
    for key in ["llir_speedup", "ttir_speedup", "baseline_ratio", "optimized_ratio"]:
        vals = [r[key] for r in result["records"]]
        result["summary"][key] = dict(
            arithmetic_mean=statistics.mean(vals),
            geometric_mean=math.exp(statistics.mean(math.log(v) for v in vals)),
        )
    save_json(args.output, result)
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
