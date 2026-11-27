"""Validate, then measure the frozen final-route ablations on all 30 shapes."""

import argparse
import json
import sys
from pathlib import Path

OP = Path(__file__).resolve().parent
sys.path.insert(0, str(OP / "multiround"))
import r6_run as r  # noqa: E402


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--seed", type=int, required=True)
    cli.add_argument("--rounds", type=int, required=True)
    cli.add_argument("--rep", type=int, default=10)
    cli.add_argument("--compile-only", action="store_true")
    cli.add_argument("--check-only", action="store_true")
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    r.ROOT = OP / "ablation_v1"
    manifest = json.loads((r.ROOT / "manifest.json").read_text())
    points = json.loads((r.ROOT / "manifest30.json").read_text())["points"]
    names = manifest["variants"]
    env = r.environment()
    pairs, meta = {}, {}
    for e in (256, 512):
        for n in names:
            pairs[e, n], meta[f"{e}/{n}"] = r.compile_pair(e, n)
            print("compiled", e, n, flush=True)
    r.save(args.output / "compile.json", meta)
    if args.compile_only:
        return
    gate = r.verify(points, pairs)
    r.save(args.output / "pipeline.json", gate)
    accuracy = r.validate(points, names, pairs, args.seed)
    r.save(args.output / "accuracy.json", accuracy)
    assert not accuracy["failed"]
    if args.check_only:
        return
    result = dict(
        environment=env,
        manifest_sha256=r.digest(r.ROOT / "manifest.json"),
        shape_manifest_sha256=r.digest(r.ROOT / "manifest30.json"),
        seed=args.seed,
        rounds=args.rounds,
        rep_ms=args.rep,
        names=names,
        records=[],
    )
    for item in points:
        m, e = item["tokens"], item["experts"]
        seed = args.seed + m * 1000003 + e
        t = r.c.allocate(m, e, seed=seed)
        timed = {n: pairs[e, n] for n in names}
        timed.update(parent=timed["baseline"], anchor=timed["baseline"])
        timing = r.measure(t, timed, args.rounds, args.rep, seed)
        result["records"].append(
            dict(
                shape=[m, e],
                timing=timing,
                input_seed=seed,
                input_sha256={k: r.tensor_digest(t[k]) for k in ("x", "b")},
            )
        )
        r.save(args.output / "performance.json", result)
        print(
            "bench",
            m,
            e,
            "final",
            timing["baseline_over_candidate"]["final"],
            "AA",
            timing["aa_median"],
            flush=True,
        )
    result["aa_passed"] = all(
        0.97 <= row["timing"]["aa_median"] <= 1.03 for row in result["records"]
    )
    r.save(args.output / "performance.json", result)
    assert result["aa_passed"]


if __name__ == "__main__":
    main()
