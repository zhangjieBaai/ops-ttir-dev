"""Cross the frozen baseline and accepted r6 stages on the pilot or full frozen matrix."""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "multiround"))
import r6_run as r  # noqa: E402


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("output", type=Path)
    cli.add_argument("--seed", required=True, type=int)
    cli.add_argument("--rounds", required=True, type=int)
    cli.add_argument(
        "--full", action="store_true", help="Use the full 30-point r6 matrix"
    )
    cli.add_argument("--rep", type=int, default=10)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    r.ROOT = r.BASE / "r6"
    points = [dict(tokens=m, experts=e) for e in (256, 512) for m in (16, 512, 8192)]
    manifest = r.ROOT / "manifest30.json"
    if args.full:
        points = json.loads(manifest.read_text())["points"]
    env = r.environment()
    pairs, stages = {}, {}
    for e in (256, 512):
        base, best = [], []
        for name, group in [("baseline", base), ("batch4", best)]:
            group.extend(
                r.compile_file(
                    r.ROOT / "ir" / f"e{e}_w4" / f"{name}_s{stage}.ttir",
                    r.c.options(4),
                )
                for stage in (1, 2)
            )
            for i, ck in enumerate(group, 1):
                path = r.ROOT / "ir" / f"e{e}_w4" / f"{name}_s{i}.ttir"
                stages[f"{e}/{name}/{i}"] = {
                    "ir_sha256": r.digest(path),
                    "cubin_sha256": r.sha(ck.asm["cubin"]),
                }
        pairs[e, "baseline"] = tuple(base)
        pairs[e, "stage1"] = (best[0], base[1])
        pairs[e, "stage2"] = (base[0], best[1])
        pairs[e, "both"] = tuple(best)
    gate = r.verify(points, pairs)
    names = ["baseline", "stage1", "stage2", "both"]
    accuracy = r.validate(points, names, pairs, args.seed)
    r.save(args.output / "validation.json", dict(pipeline=gate, accuracy=accuracy))
    assert not accuracy["failed"]
    rows = []
    for item in points:
        m, e = item["tokens"], item["experts"]
        seed = args.seed + m * 1000003 + e
        t = r.c.allocate(m, e, seed=seed)
        # The existing paired timer uses the name "parent" for its A/A control.
        timed = {("parent" if n == "baseline" else n): pairs[e, n] for n in names}
        # The timer also computes ratios against baseline and anchor. Both names
        # denote frozen controls here. Preserve the unmodified timing protocol.
        timed["baseline"] = timed["parent"]
        timed["anchor"] = timed["parent"]
        timing = r.measure(t, timed, args.rounds, args.rep, seed)
        rows.append(
            dict(
                shape=[m, e],
                timing=timing,
                input_seed=seed,
                input_sha256={key: r.tensor_digest(t[key]) for key in ("x", "b")},
            )
        )
        print("bench", m, e, "AA", timing["aa_median"], flush=True)
    r.save(
        args.output / "performance.json",
        dict(
            environment=env,
            stages=stages,
            seed=args.seed,
            rounds=args.rounds,
            rep_ms=args.rep,
            manifest_sha256=r.digest(manifest),
            rows=rows,
        ),
    )


if __name__ == "__main__":
    main()
