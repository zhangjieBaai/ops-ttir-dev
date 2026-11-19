"""Freeze a single uniform combined candidate using only round2 tuning evidence."""

import json

from common import ROOT, ir_path, manifest, save_json, sha256


def main():
    screen = ROOT / "results/runs/round2/screen_combined.json"
    evidence = json.loads(screen.read_text())
    assert evidence["suite"] == "tuning" and evidence["aa_passed"]
    assert evidence["round2_plan_sha256"] == sha256(
        ROOT / "optimization_plan_round2.md"
    )
    name = max(
        evidence["summaries"], key=lambda n: evidence["summaries"][n]["geometric"]
    )
    assert evidence["summaries"][name]["geometric"] > 1
    # Small noise is disclosed; selection is uniform, with no per-shape lookup.
    records = {}
    for group in manifest()["groups"]:
        source = ir_path(group, name)
        dest = ir_path(group, "r2_final")
        with dest.open("x") as stream:
            stream.write(source.read_text())
        records[group] = dict(candidate=name, sha256=sha256(dest))
    save_json(
        ROOT / "results/round2_finalist.json",
        dict(
            selected=name,
            plan_sha256=sha256(ROOT / "optimization_plan_round2.md"),
            screen_sha256=sha256(screen),
            groups=records,
            rule=(
                "Uniform candidate; original paths; N<8 parent; "
                "tail_remaining=1 parallel reduce; no shape lookup"
            ),
            tuning_summary=evidence["summaries"][name],
        ),
    )
    print("selected", name, evidence["summaries"][name], flush=True)


if __name__ == "__main__":
    main()
