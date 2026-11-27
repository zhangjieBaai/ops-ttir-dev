"""Diagnose parent-only A/A at the previously unstable 256-CTA D64 point."""

import argparse
import statistics
from pathlib import Path

import triton.testing
from common_round2 import allocate, compile_ir, input_hashes, launch, save, snapshot
from warmup import GraphWarmup


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    assert not out.exists()
    t = allocate(2, 8, 2048, 64, False, seed=20)
    ck, meta = compile_ir("parent", 64, False)

    def fn():
        launch(ck, t)

    warm = GraphWarmup(fn, launches=32)
    result = {
        "environment": snapshot(),
        "metadata": meta,
        "input_sha256": input_hashes(t),
        "shape": [2, 8, 2048, 64, False],
        "controls": {},
    }
    for mode in ("fresh_fn", "fixed_inner32", "fixed_events32"):
        rows = []
        warm.run(500)
        for turn in range(5):
            times = []
            for _ in range(2):
                w = warm.run(100)
                if mode == "fresh_fn":
                    us = triton.testing.do_bench_cudagraph(fn, rep=20) * 1000
                elif mode == "fixed_inner32":
                    us = (
                        triton.testing.do_bench_cudagraph(warm.graph.replay, rep=20)
                        * 1000
                        / 32
                    )
                else:
                    # Exact100 graph replays, each32 launches, timed with CUDA events.
                    us = warm.batch(100) * 1000 / (100 * 32)
                times.append({"us": us, "warmup": w})
            rows.append(
                {
                    "round": turn,
                    "times": times,
                    "ratio": times[0]["us"] / times[1]["us"],
                }
            )
        result["controls"][mode] = {
            "samples": rows,
            "aa_median": statistics.median(x["ratio"] for x in rows),
        }
        save(out, result)
        print(
            mode,
            result["controls"][mode]["aa_median"],
            [(x["times"][0]["us"], x["times"][1]["us"]) for x in rows],
            flush=True,
        )


if __name__ == "__main__":
    main()
