"""Freeze round-three proposals before independent confirmation; keep rejected IR archived."""

import argparse
import json
import math
import statistics

from close_round2 import group_ratios
from common import BLOCKS, ROOT, digest, save

R = ROOT / "results/round3"


def qualifies(data, name, block):
    parent = group_ratios(data, name, block, "r3_parent")
    rows = [r for r in data["records"] if r["kernel"] == name and r["block"] == block]
    baseline = math.exp(statistics.mean(math.log(r["speedup"]) for r in rows))
    return dict(
        vs_parent=parent,
        vs_baseline_geometric_mean=baseline,
        passed=parent["geometric_mean"] >= 1.02
        and parent["worst"] >= 0.95
        and baseline >= 1.02,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--final", action="store_true")
    args = p.parse_args()
    full = json.load(open(R / "full.json"))
    assert full["aa_passed"] and full["rounds"] == 7 and len(full["records"]) == 60
    proposal = {str(b): qualifies(full, "r3_group_center2048", b) for b in BLOCKS}
    path = R / "provisional_selection.json"
    if path.exists():
        assert json.load(open(path)) == proposal
    else:
        save(path, proposal)
    if not args.final:
        print(proposal)
        return
    repeat = json.load(open(R / "repeat.json"))
    assert (
        repeat["aa_passed"] and repeat["rounds"] == 5 and len(repeat["records"]) == 60
    )
    # No promotion is allowed here without all final acceptance checks.
    accepted = [
        b
        for b in BLOCKS
        if proposal[str(b)]["passed"]
        and qualifies(repeat, "r3_group_center2048", b)["passed"]
    ]
    assert (
        not accepted
    ), "A winning proposal requires explicit final artifact and sanitizer qualification"
    for data in [full, repeat]:
        for key, v in data["kernels"].items():
            name, group = key.split("/")
            assert v["ttir_sha256"] == digest(ROOT / f"ir/{group}/{name}.ttir")
    for b in BLOCKS:
        assert digest(ROOT / f"ir/b{b}/optimized.ttir") == digest(
            ROOT / f"ir/b{b}/r3_parent.ttir"
        )
    save(
        R / "decision.json",
        dict(
            promoted_blocks=[],
            main=proposal,
            repeat={
                str(b): qualifies(repeat, "r3_group_center2048", b) for b in BLOCKS
            },
            retained_selection_sha256=digest(ROOT / "results/selection.json"),
            unchanged_aliases=True,
        ),
    )
    print("No promotion; current optimized TTIR retained")


if __name__ == "__main__":
    main()
