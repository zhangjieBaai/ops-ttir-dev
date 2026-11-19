"""Freeze second-round dX family proposal, then qualify against the immutable parent."""

import argparse
import json
import math
import shutil
import statistics

from common import BLOCKS, ROOT, digest, save

R = ROOT / "results/round2"


def group_ratios(data, name, block, reference):
    rows = {
        v["N"]: v
        for v in data["records"]
        if v["kernel"] == name and v["block"] == block
    }
    parents = {
        v["N"]: v
        for v in data["records"]
        if v["kernel"] == reference and v["block"] == block
    }
    assert rows.keys() == parents.keys()
    ratios = [parents[n]["candidate_us"] / rows[n]["candidate_us"] for n in rows]
    return dict(
        geometric_mean=math.exp(statistics.mean(math.log(v) for v in ratios)),
        worst=min(ratios),
        provenance="ratio of same-session per-kernel median durations; each timing paired with common baseline",
    )


def proposal(full):
    names = [n for n in full["summary"] if n != "r1_parent"]
    out = {}
    for b in BLOCKS:
        stats = {n: group_ratios(full, n, b, "r1_parent") for n in names}
        best = max(names, key=lambda n: stats[n]["geometric_mean"])
        if (
            stats[best]["geometric_mean"] < 1.02
            or stats[best]["worst"] < 0.95
            or math.exp(
                statistics.mean(
                    math.log(v["speedup"])
                    for v in full["records"]
                    if v["kernel"] == best and v["block"] == b
                )
            )
            < 1.02
        ):
            best = "r1_parent"
        out[str(b)] = dict(candidate=best, vs_parent=stats)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--final", action="store_true")
    args = p.parse_args()
    full = json.load(open(R / "full.json"))
    assert full["aa_passed"] and len(full["records"]) == 90 and full["rounds"] == 7
    chosen = proposal(full)
    if not args.final:
        target = R / "provisional_selection.json"
        if target.exists():
            assert json.load(open(target)) == chosen
        save(target, chosen)
        print(chosen)
        return
    assert json.load(open(R / "provisional_selection.json")) == chosen
    repeat = json.load(open(R / "repeat.json"))
    assert (
        repeat["aa_passed"] and len(repeat["records"]) == 90 and repeat["rounds"] == 5
    )
    for record in [full, repeat]:
        for key, v in record["kernels"].items():
            name, group = key.split("/")
            assert v["ttir_sha256"] == digest(ROOT / f"ir/{group}/{name}.ttir")
    for filename in [
        "accuracy_ranges.json",
        "accuracy_repeat.json",
        "memcheck_accuracy.json",
        "racecheck_accuracy.json",
    ]:
        check = json.load(open(R / filename))
        assert check["passed"]
        for key, h in check["kernels"].items():
            name, group = key.split("/")
            assert h == digest(ROOT / f"ir/{group}/{name}.ttir")
    assert "ERROR SUMMARY: 0 errors" in (R / "memcheck.log").read_text()
    assert (
        "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
        in (R / "racecheck.log").read_text()
    )
    previous = json.load(open(R / "parent_selection.json"))
    families = {}
    promoted = []
    for b in BLOCKS:
        x = chosen[str(b)]
        name = x["candidate"]
        accepted = False
        rep = None
        if name != "r1_parent":
            rep = group_ratios(repeat, name, b, "r1_parent")
            ratios = [
                v["speedup"]
                for v in repeat["records"]
                if v["kernel"] == name and v["block"] == b
            ]
            accepted = (
                rep["geometric_mean"] >= 1.02
                and rep["worst"] >= 0.95
                and math.exp(statistics.mean(math.log(v) for v in ratios)) >= 1.02
            )
            if not accepted:
                name = "r1_parent"
        path = ROOT / f"ir/b{b}/{name}.ttir"
        if accepted:
            shutil.copyfile(path, ROOT / f"ir/b{b}/optimized.ttir")
            promoted.append(b)
            families[str(b)] = dict(
                candidate=name, sha256=digest(path), accepted=True, repeat=rep, main=x
            )
        else:
            assert (
                digest(ROOT / f"ir/b{b}/optimized.ttir")
                == previous["families"][str(b)]["sha256"]
            )
            families[str(b)] = previous["families"][str(b)]
    result = dict(
        passed=True,
        promoted_blocks=promoted,
        families=families,
        full_sha256=digest(R / "full.json"),
        repeat_sha256=digest(R / "repeat.json"),
        round2_proposals=chosen,
    )
    save(R / "decision.json", result)
    if promoted:
        save(ROOT / "results/selection.json", result)
    print("Promoted families:", promoted)


if __name__ == "__main__":
    main()
