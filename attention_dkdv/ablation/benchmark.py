"""Paired D128 decomposition on the sixteen frozen D128 shapes."""

import argparse
import os
import random
import sys
from pathlib import Path

import torch
import triton
import triton.testing

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "round2"))
from common_round2 import (  # noqa: E402
    MATRIX,
    OPTIONS,
    _attn_bwd_dkdv_only,
    allocate,
    digest,
    group,
    input_hashes,
    launch,
    launch_source,
    save,
    snapshot,
)
from warmup import GraphWarmup  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rng = random.Random(args.seed)
    result = dict(
        environment=snapshot(), seed=args.seed, rounds=args.rounds, rep_ms=20, points=[]
    )
    if os.environ.get("ATTENTION_ABLATION_SM_MHZ"):
        result["environment"].update(
            requested_sm_clock_mhz=int(os.environ["ATTENTION_ABLATION_SM_MHZ"]),
            clocks_locked=True,
            clock_note="run.sh successfully requested locked SM clocks before launch",
        )
    for shape in MATRIX:
        b, h, n, d, causal = shape
        if d != 128:
            continue
        g = group(d, causal)
        r2 = ROOT / "round2/ir" / g
        r6 = ROOT / "multiround/r6/ir" / g
        structural = "rows64_unroll_v5" if causal else "rows64_remat_v3"
        paths = {
            "baseline": r2 / "baseline.ttir",
            "rows64": r2 / "rows64_v3.ttir",
            "structure": r2 / (structural + ".ttir"),
            "final": r6 / "optimized.ttir",
        }
        assert paths["structure"].read_bytes() == (r6 / "parent.ttir").read_bytes()
        tensors = allocate(*shape, seed=args.seed)
        hashes = input_hashes(tensors)
        source = launch_source(_attn_bwd_dkdv_only, tensors)
        expected = [tensors[k].view(torch.int16).clone() for k in ("dk", "dv")]
        kernels = {
            name: triton.compile(str(path), options=OPTIONS)
            for name, path in paths.items()
        }
        for ext in ("llir", "ptx", "cubin"):
            assert source.asm[ext] == kernels["baseline"].asm[ext]
        warmers = {}
        for name, kernel in kernels.items():
            for key in ("dk", "dv"):
                tensors[key].fill_(float("nan"))
            launch(kernel, tensors)
            assert all(
                torch.equal(tensors[k].view(torch.int16), e)
                for k, e in zip(("dk", "dv"), expected)
            )
            warmers[name] = GraphWarmup(lambda kernel=kernel: launch(kernel, tensors))
            warmers[name].run(500)
        assert input_hashes(tensors) == hashes
        samples = []
        for _ in range(args.rounds):
            order = list(kernels) + ["baseline_copy"]
            rng.shuffle(order)
            times = {}
            for label in order:
                name = "baseline" if label == "baseline_copy" else label
                warmers[name].run(100)
                times[label] = 1000 * triton.testing.do_bench_cudagraph(
                    lambda: launch(kernels[name], tensors), rep=20
                )
            samples.append(dict(order=order, us=times))
        result["points"].append(
            dict(
                shape=shape,
                structural=structural,
                samples=samples,
                input_hashes=hashes,
                ttir_sha256={k: digest(p) for k, p in paths.items()},
                source_pipeline_equal=True,
                outputs_bitwise_equal=True,
            )
        )
        save(args.output, result)
        print(shape, flush=True)


if __name__ == "__main__":
    main()
