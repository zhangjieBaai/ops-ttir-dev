"""Paired full-device-sequence timing after pipeline and numerical gates."""

import argparse
import json
import random
import statistics

import common as c
import torch
from triton.testing import do_bench_cudagraph
from validate import validate
from verify_pipeline import verify


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument(
        "--variants", nargs="+", default=["membership", "argmax", "combined"]
    )
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--rep", type=int, default=20)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    folder = c.ROOT / "results" / args.run
    assert not folder.exists(), "Do not overwrite results"
    folder.mkdir()
    verify(folder / "pipeline.json")
    names = sorted(set(["baseline", *args.variants]))
    numerical = validate(folder / "accuracy.json", names, args.seed)
    assert numerical["pass_all"], "No timing after failed numerical gate"
    manifest = json.loads((c.ROOT / "baseline.json").read_text())
    result = dict(
        environment=c.environment(),
        rounds=args.rounds,
        rep_ms=args.rep,
        seed=args.seed,
        points=[],
    )
    rng = random.Random(args.seed)
    for item in manifest["mapping"].values():
        m, e, w = (item[k] for k in ("tokens", "experts", "warps"))
        t = c.allocate(m, e, seed=args.seed)
        compiled = {name: c.compile_pair(e, w, name) for name in names}
        for pair in compiled.values():
            for _ in range(5):
                c.launch(pair, t)
        torch.cuda.synchronize()
        rows = []
        for rnd in range(args.rounds):
            order = list(args.variants)
            rng.shuffle(order)
            for variant in order:
                pair_order = ["baseline", "candidate"]
                rng.shuffle(pair_order)
                ms = {}
                for which in pair_order:
                    name = variant if which == "candidate" else "baseline"
                    ms[which] = do_bench_cudagraph(
                        lambda: c.launch(compiled[name], t), rep=args.rep
                    )
                rows.append(
                    dict(
                        round=rnd,
                        variant=variant,
                        order=pair_order,
                        ms=ms,
                        speedup=ms["baseline"] / ms["candidate"],
                    )
                )
        summary = {
            name: statistics.median(
                row["speedup"] for row in rows if row["variant"] == name
            )
            for name in args.variants
        }
        result["points"].append(
            dict(
                item,
                samples=rows,
                median_speedups=summary,
                artifacts={
                    name: c.artifacts(pair, e, w, name)
                    for name, pair in compiled.items()
                },
            )
        )
        c.save(folder / "performance.json", result)
        print(m, e, summary, flush=True)
    result["geomean_speedups"] = {
        name: statistics.geometric_mean(
            point["median_speedups"][name] for point in result["points"]
        )
        for name in args.variants
    }
    c.save(folder / "performance.json", result)
    print(result["geomean_speedups"], flush=True)


if __name__ == "__main__":
    main()
