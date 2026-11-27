"""Freeze source TTIR, tune only baseline configurations, freeze per-shape mapping."""

import random

import common as c
import torch
from triton.testing import do_bench_cudagraph


def main():
    assert not (c.ROOT / "baseline.json").exists(), "Baseline is immutable"
    env = c.environment()
    tuning = []
    mapping = {}
    for m, e in c.PERF:
        t = c.allocate(m, e, seed=2026)
        choices = {}
        for w in (4, 8):
            src = c.source_launch(t, w)
            for stage, kernel in enumerate(src, 1):
                c.immutable(
                    c.ROOT / "ir" / c.group(e, w) / f"baseline_s{stage}.ttir",
                    kernel.asm["ttir"],
                )
            choices[w] = c.compile_pair(e, w)
            for s, k in zip(src, choices[w]):
                assert all(s.asm[ext] == k.asm[ext] for ext in ("llir", "ptx", "cubin"))
        rows = []
        for rnd in range(3):
            order = [4, 8]
            random.Random(71 + rnd + m).shuffle(order)
            for w in order:
                ms = do_bench_cudagraph(lambda: c.launch(choices[w], t), rep=10)
                rows.append(dict(round=rnd, warps=w, ms=ms))
        scores = {w: sum(r["ms"] for r in rows if r["warps"] == w) / 3 for w in (4, 8)}
        chosen = min(scores, key=scores.get)
        mapping[f"{m}x{e}"] = dict(
            tokens=m, experts=e, warps=chosen, group=c.group(e, chosen)
        )
        tuning.append(dict(shape=[m, e], samples=rows, means_ms=scores, chosen=chosen))
        print(m, e, scores, chosen, flush=True)
    torch.cuda.synchronize()
    c.save(
        c.ROOT / "results/baseline_tuning.json", dict(environment=env, points=tuning)
    )
    c.save(
        c.ROOT / "baseline.json",
        dict(mapping=mapping, protocol_sha256=c.sha(c.ROOT / "optimization_plan.md")),
    )


if __name__ == "__main__":
    main()
