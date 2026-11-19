"""Select dX candidate families from the main run, then independently qualify or retain baseline."""

import argparse
import json
import math
import shutil
import statistics

from common import BLOCKS, ROOT, digest, save


def summary(rows, name, block):
    vals = [v["speedup"] for v in rows if v["kernel"] == name and v["block"] == block]
    return dict(
        geometric_mean=math.exp(statistics.mean(math.log(v) for v in vals)),
        worst=min(vals),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--final", action="store_true")
    args = p.parse_args()
    r = ROOT / "results"
    full = json.load(open(r / "full.json"))
    assert full["aa_passed"] and len(full["records"]) == 60 and full["rounds"] == 7
    validation = json.load(open(r / "accuracy_all.json"))
    assert validation["passed"]
    for key, entry in full["kernels"].items():
        assert entry["ttir_sha256"] == validation["kernels"][key]
        name, group = key.split("/")
        assert entry["ttir_sha256"] == digest(ROOT / f"ir/{group}/{name}.ttir")
    names = list(full["summary"])
    proposal = {}
    for block in BLOCKS:
        by_name = {name: summary(full["records"], name, block) for name in names}
        best = max(by_name, key=lambda n: by_name[n]["geometric_mean"])
        accept = (
            by_name[best]["geometric_mean"] >= 1.02 and by_name[best]["worst"] >= 0.95
        )
        proposal[str(block)] = dict(
            candidate=best if accept else "baseline", main=by_name
        )
    if not args.final:
        path = r / "provisional_selection.json"
        if path.exists():
            assert json.load(open(path)) == proposal
        save(path, proposal)
        print(proposal)
        return
    assert json.load(open(r / "provisional_selection.json")) == proposal
    repeat = json.load(open(r / "repeat.json"))
    assert (
        repeat["aa_passed"] and repeat["rounds"] == 5 and len(repeat["records"]) == 60
    )
    assert repeat["validation_sha256"] == digest(r / "accuracy_repeat.json")
    for key, entry in repeat["kernels"].items():
        assert entry["ttir_sha256"] == full["kernels"][key]["ttir_sha256"]
    for filename in [
        "accuracy_round1.json",
        "accuracy_grouped.json",
        "accuracy_repeat.json",
        "memcheck_accuracy.json",
        "racecheck_accuracy.json",
    ]:
        check = json.load(open(r / filename))
        assert check["passed"], filename
        for key, d in check["kernels"].items():
            name, group = key.split("/")
            assert d == digest(ROOT / f"ir/{group}/{name}.ttir"), (filename, key)
    assert "ERROR SUMMARY: 0 errors" in (r / "memcheck.log").read_text()
    assert (
        "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
        in (r / "racecheck.log").read_text()
    )
    selected = {}
    for block in BLOCKS:
        proposal_group = proposal[str(block)]
        name = proposal_group["candidate"]
        qualified = True
        if name != "baseline":
            s = summary(repeat["records"], name, block)
            qualified = s["geometric_mean"] >= 1.02 and s["worst"] >= 0.95
        else:
            s = None
        if not qualified:
            name = "baseline"
        src = ROOT / f"ir/b{block}/{name}.ttir"
        shutil.copyfile(src, ROOT / f"ir/b{block}/optimized.ttir")
        selected[str(block)] = dict(
            candidate=name,
            sha256=digest(src),
            main=proposal_group["main"],
            repeat=s,
            accepted=qualified and name != "baseline",
        )
    save(
        r / "selection.json",
        dict(
            passed=True,
            full_sha256=digest(r / "full.json"),
            repeat_sha256=digest(r / "repeat.json"),
            families=selected,
        ),
    )
    print(selected)


if __name__ == "__main__":
    main()
