"""Check that graph timing detects doubled work; separate from performance suite."""

import argparse
import random
import statistics
from pathlib import Path

import triton.testing
from common import (
    allocate,
    check_contention,
    compile_ir,
    group_for,
    launch,
    receipt,
    save_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    shape = (256, 8192, 64)
    group, _ = group_for(shape)
    kernel = compile_ir(group)
    tensors = allocate(shape)
    check_contention()

    def one():
        launch(kernel, shape, tensors)

    def two():
        one()
        one()

    triton.testing.do_bench_cudagraph(one, rep=100)
    rng = random.Random(81)
    pairs = []
    for _ in range(5):
        order = ["one", "two"]
        rng.shuffle(order)
        raw = {
            name: triton.testing.do_bench_cudagraph(
                one if name == "one" else two, rep=20, return_mode="all"
            )
            for name in order
        }
        pairs.append(
            dict(
                order=order,
                raw_ms=raw,
                ratio=statistics.mean(raw["two"]) / statistics.mean(raw["one"]),
            )
        )
    ratio = statistics.median(p["ratio"] for p in pairs)
    passed = 1.8 <= ratio <= 2.2
    save_json(
        args.output,
        dict(**receipt(), shape=shape, pairs=pairs, median_ratio=ratio, passed=passed),
    )
    assert passed


if __name__ == "__main__":
    main()
