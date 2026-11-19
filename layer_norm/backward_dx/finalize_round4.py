"""Freeze per-family winners from full timing, then qualify without reselecting on repeat."""

import argparse
import json
import shutil

from close_round3 import qualifies
from common import BLOCKS, ROOT, digest, save

R = ROOT / "results/round4"
NAMES = ["r4_evict", "r4_center_maskfold"]


def stats(data, name, block):
    # qualifies() uses round-three parent; the implementation is the same,
    # so adapt a copy of the records rather than changing archived evidence.
    renamed = dict(
        data,
        records=[
            dict(r, kernel="r3_parent") if r["kernel"] == "r4_parent" else r
            for r in data["records"]
        ],
    )
    return qualifies(renamed, name, block)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    full = json.load(open(R / "full.json"))
    assert full["aa_passed"] and full["rounds"] == 7 and len(full["records"]) == 90
    proposal = {}
    for block in BLOCKS:
        values = {name: stats(full, name, block) for name in NAMES}
        eligible = [name for name in NAMES if values[name]["passed"]]
        name = (
            max(eligible, key=lambda n: values[n]["vs_parent"]["geometric_mean"])
            if eligible
            else "r4_parent"
        )
        proposal[str(block)] = dict(candidate=name, stats=values)
    target = R / "provisional_selection.json"
    if args.freeze:
        if target.exists():
            assert json.load(open(target)) == proposal
        save(target, proposal)
        print(proposal)
        return
    assert json.load(open(target)) == proposal
    repeat = json.load(open(R / "repeat.json"))
    assert (
        repeat["aa_passed"] and repeat["rounds"] == 5 and len(repeat["records"]) == 90
    )
    checks = {}
    for filename in [
        "accuracy_finalists.json",
        "accuracy_repeat.json",
        "accuracy_boundaries.json",
        "memcheck_accuracy.json",
        "racecheck_accuracy.json",
        "pipeline_before.json",
    ]:
        data = json.load(open(R / filename))
        assert data["passed"]
        for key, sha in data.get("kernels", {}).items():
            name, group = key.split("/")
            assert sha == digest(ROOT / f"ir/{group}/{name}.ttir")
        checks[filename] = digest(R / filename)
    assert "ERROR SUMMARY: 0 errors" in (R / "memcheck.log").read_text()
    assert (
        "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
        in (R / "racecheck.log").read_text()
    )
    for filename in ["stress_batch2.json", "stress_batch3.json"]:
        data = json.load(open(R / filename))
        for name in NAMES:
            records = [r for r in data["records"] if r["kernel"] == name]
            if records:
                assert len(records) == 72 and all(r["passed"] for r in records)
    for data in [full, repeat]:
        for key, v in data["kernels"].items():
            name, group = key.split("/")
            assert v["ttir_sha256"] == digest(ROOT / f"ir/{group}/{name}.ttir")
    previous = json.load(open(R / "parent_selection.json"))
    assert digest(R / "parent_selection.json") == digest(
        ROOT / "results/selection.json"
    )
    promoted, families = [], {}
    for block in BLOCKS:
        key = str(block)
        name = proposal[key]["candidate"]
        accepted = name != "r4_parent" and stats(repeat, name, block)["passed"]
        if accepted:
            path = ROOT / f"ir/b{block}/{name}.ttir"
            shutil.copyfile(path, ROOT / f"ir/b{block}/optimized.ttir")
            families[key] = dict(
                candidate=name,
                sha256=digest(path),
                accepted=True,
                main=proposal[key],
                repeat=stats(repeat, name, block),
            )
            promoted.append(block)
        else:
            assert (
                digest(ROOT / f"ir/b{block}/optimized.ttir")
                == previous["families"][key]["sha256"]
            )
            families[key] = previous["families"][key]
    out = dict(
        passed=True,
        promoted_blocks=promoted,
        families=families,
        main=proposal,
        repeat={str(b): {n: stats(repeat, n, b) for n in NAMES} for b in BLOCKS},
        checks=checks,
        full_sha256=digest(R / "full.json"),
        repeat_sha256=digest(R / "repeat.json"),
        parent_selection_sha256=digest(R / "parent_selection.json"),
    )
    save(R / "selection_final.json", out)
    if promoted:
        save(ROOT / "results/selection.json", out)
    print("Promoted:", promoted)


if __name__ == "__main__":
    main()
