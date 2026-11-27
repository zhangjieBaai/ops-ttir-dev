"""Freeze a finalist before independent confirmation."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main(stage):
    folder = ROOT / stage
    data = json.loads((folder / "results/screen/results.json").read_text())
    assert data["passed"] and data["aa_passed"] and len(data["records"]) == 30
    names = [
        n for n in data["valid_names"] if n not in ("baseline", "anchor", "parent")
    ]
    eligible = [
        n for n in names if data["summary"][n]["incremental"]["minimum"] >= 0.95
    ]
    name = max(
        eligible or names, key=lambda n: data["summary"][n]["incremental"]["geomean"]
    )
    target = folder / "selection_pre_main.json"
    assert not target.exists()
    target.write_text(
        json.dumps(
            {
                "candidate": name,
                "rule": (
                    "highest full30 incremental geo with worst>=.95 when available; "
                    "no repeat reselection"
                ),
                "screen_summary": data["summary"][name],
            },
            indent=2,
        )
        + "\n"
    )
    print(name)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["r3", "r4", "r5"])
    main(p.parse_args().stage)
