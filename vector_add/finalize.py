"""Select the best TTIR only when frozen promotion criteria are satisfied."""

import json
import shutil

from common import ROOT, save_json, sha256


def main():
    result = json.loads((ROOT / "results/runs/r1_20261119/full.json").read_text())
    assert result["aa_passed"]
    candidates = result["summaries"]
    best = max(candidates, key=lambda name: candidates[name]["geometric"])
    summary = candidates[best]
    # This round closes only a failed promotion; a successful promotion needs
    # independent timing and sanitizer gates before a different finalizer.
    assert summary["geometric"] < 1.02, "Promotion requires all remaining gates"
    accepted = False
    source = ROOT / "ir/baseline.ttir"
    selected = ROOT / "ir/optimized.ttir"
    if selected.exists():
        assert sha256(selected) == sha256(
            source
        ), "Never silently replace optimized.ttir"
    else:
        shutil.copyfile(source, selected)
    save_json(
        ROOT / "results/selection.json",
        dict(
            selected="optimized" if accepted else "baseline",
            accepted=accepted,
            selection_reason=(
                f"{best} passed frozen performance gates"
                if accepted
                else f"best candidate {best} has {summary['geometric']:.6f}x geometric mean; "
                "below the preregistered 1.02x promotion threshold"
            ),
            selected_path="ir/optimized.ttir",
            selected_sha256=sha256(selected),
            selected_matches="ir/" + (best if accepted else "baseline") + ".ttir",
            best_candidate=best,
            best_summary=summary,
            candidate_summaries=candidates,
            performance_evidence="results/runs/r1_20261119/full.json",
            accuracy_evidence="results/runs/r1_20261119/accuracy.json",
        ),
    )
    print(
        "selected", "optimized" if accepted else "baseline", best, summary["geometric"]
    )


if __name__ == "__main__":
    main()
