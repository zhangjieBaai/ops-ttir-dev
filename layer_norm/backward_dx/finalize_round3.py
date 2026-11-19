"""Freeze the stream finalist before repeat, then accept families after all gates."""

import argparse
import json
import shutil

from close_round3 import qualifies
from common import BLOCKS, ROOT, digest, save

R = ROOT / "results/round3"
NAME = "r3_centered_stream"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    full = json.load(open(R / "full_stream.json"))
    assert full["aa_passed"] and full["rounds"] == 7 and len(full["records"]) == 60
    proposal = {str(b): qualifies(full, NAME, b) for b in BLOCKS}
    path = R / "provisional_stream.json"
    if args.freeze:
        if path.exists():
            assert json.load(open(path)) == proposal
        save(path, proposal)
        print(proposal)
        return
    assert json.load(open(path)) == proposal
    repeat = json.load(open(R / "repeat_stream.json"))
    assert (
        repeat["aa_passed"] and repeat["rounds"] == 5 and len(repeat["records"]) == 60
    )
    checks = {}
    for filename in [
        "accuracy_batch1.json",
        "accuracy_boundaries.json",
        "accuracy_stream_repeat.json",
        "memcheck_stream_accuracy.json",
        "racecheck_stream_accuracy.json",
        "pipeline_before.json",
    ]:
        check = json.load(open(R / filename))
        assert check["passed"]
        for key, h in check.get("kernels", {}).items():
            name, group = key.split("/")
            assert h == digest(ROOT / f"ir/{group}/{name}.ttir")
        checks[filename] = digest(R / filename)
    assert "ERROR SUMMARY: 0 errors" in (R / "memcheck_stream.log").read_text()
    assert (
        "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
        in (R / "racecheck_stream.log").read_text()
    )
    stress = json.load(open(R / "stress_saved_stats.json"))
    for key, h in stress["kernels"].items():
        name, group = key.split("/")
        assert h == digest(ROOT / f"ir/{group}/{name}.ttir")
    rows = [r for r in stress["records"] if r["kernel"] == NAME]
    assert len(rows) == 72 and all(r["passed"] for r in rows)
    for data in [full, repeat]:
        for key, metadata in data["kernels"].items():
            name, group = key.split("/")
            assert metadata["ttir_sha256"] == digest(ROOT / f"ir/{group}/{name}.ttir")
    previous = json.load(open(R / "parent_selection.json"))
    promoted = []
    families = {}
    for block in BLOCKS:
        key = str(block)
        rep = qualifies(repeat, NAME, block)
        accepted = proposal[key]["passed"] and rep["passed"]
        if accepted:
            candidate = ROOT / f"ir/b{block}/{NAME}.ttir"
            shutil.copyfile(candidate, ROOT / f"ir/b{block}/optimized.ttir")
            families[key] = dict(
                candidate=NAME,
                sha256=digest(candidate),
                main=proposal[key],
                repeat=rep,
                accepted=True,
            )
            promoted.append(block)
        else:
            assert (
                digest(ROOT / f"ir/b{block}/optimized.ttir")
                == previous["families"][key]["sha256"]
            )
            families[key] = previous["families"][key]
    result = dict(
        passed=True,
        promoted_blocks=promoted,
        families=families,
        full_sha256=digest(R / "full_stream.json"),
        repeat_sha256=digest(R / "repeat_stream.json"),
        main=proposal,
        repeat={str(b): qualifies(repeat, NAME, b) for b in BLOCKS},
        checks=checks,
        parent_selection_sha256=digest(R / "parent_selection.json"),
        stress_sha256=digest(R / "stress_saved_stats.json"),
    )
    save(R / "selection_final.json", result)
    if promoted:
        save(ROOT / "results/selection.json", result)
    print("Promoted families:", promoted)


if __name__ == "__main__":
    main()
