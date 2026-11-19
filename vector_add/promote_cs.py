"""Apply the user's explicit streaming-store selection and archive its predecessor."""

import json
import os
import shutil

from common import ROOT, save_json, sha256

RUN = ROOT / "results/runs/promote_cs_20261119"
EXPECTED = "ddd8ec671f57abe0e0538e70b6e38dcdfc3f494ec33b73b87073cb3fe6e25c0e"


def main():
    candidate = ROOT / "ir/r1_store_cs.ttir"
    assert sha256(candidate) == EXPECTED
    history = RUN / "history"
    history.mkdir(parents=True)
    for name, source in (
        ("selection.json", ROOT / "results/selection.json"),
        ("optimized.ttir", ROOT / "ir/optimized.ttir"),
    ):
        dest = history / name
        assert not dest.exists()
        shutil.copyfile(source, dest)
    previous = json.loads((history / "selection.json").read_text())
    assert previous["selected"] == "baseline"
    save_json(
        RUN / "promotion.json",
        dict(
            candidate="r1_store_cs",
            ttir_sha256=EXPECTED,
            previous_selection_sha256=sha256(history / "selection.json"),
            previous_optimized_sha256=sha256(history / "optimized.ttir"),
            policy="user explicitly accepts the existing local gain; v1 1.02 aggregate threshold is waived",
            plan_sha256=sha256(ROOT / "promotion_plan_cs.md"),
        ),
    )
    selected = ROOT / "ir/optimized.ttir"
    temporary = selected.with_suffix(".ttir.promoting")
    shutil.copyfile(candidate, temporary)
    os.replace(temporary, selected)
    current = dict(
        selected="r1_store_cs",
        accepted=True,
        status="PROMOTED_FROM_EXISTING_EVIDENCE; new alias measurement pending",
        selected_path="ir/optimized.ttir",
        selected_sha256=sha256(selected),
        selected_matches="ir/r1_store_cs.ttir",
        selection_reason="Explicit user acceptance of verified local speedup; modest overall improvement retained",
        original_v1_aggregate_threshold_passed=False,
        predecessor="results/runs/promote_cs_20261119/history/selection.json",
        promotion_record="results/runs/promote_cs_20261119/promotion.json",
        prior_main="results/runs/r1_20261119/full.json",
        prior_repeat="results/runs/r1_20261119/repeat.json",
    )
    dest = ROOT / "results/selection.json"
    temporary = dest.with_suffix(".json.promoting")
    save_json(temporary, current)
    os.replace(temporary, dest)
    print("Promoted r1_store_cs to optimized.ttir; previous state archived")


if __name__ == "__main__":
    main()
