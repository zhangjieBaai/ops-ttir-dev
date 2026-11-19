"""Freeze a candidate using the preregistered tuning criterion before confirmation."""

import argparse
import json
from pathlib import Path

from common import ROOT, digest, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    screen = json.loads(Path(args.screen).read_text())
    assert screen["valid"] and screen["suite"] == "tuning"
    eligible = {
        name: value
        for name, value in screen["summary"].items()
        if value["geomean"] > 1.02 and value["worst"] >= 0.90
    }
    assert eligible, "No candidate meets the frozen retention criterion"
    chosen = max(eligible, key=lambda name: eligible[name]["geomean"])
    source = ROOT / f"ir/{chosen}.ttir"
    save(
        args.output,
        dict(
            candidate=chosen,
            ttir_sha256=digest(source),
            screen_sha256=digest(args.screen),
            tuning_summary=screen["summary"],
            reason="Largest equal-weight tuning geometric mean among worst>=0.90 candidates",
            validation_exposure=(
                "All 30 points previously exposed in superseded int32 trial; "
                "confirmation, not fresh holdout"
            ),
        ),
    )
    print(chosen)


if __name__ == "__main__":
    main()
