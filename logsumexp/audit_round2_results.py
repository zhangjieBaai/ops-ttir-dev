"""Recompute every saved replay pair and verify execution identities for round2."""

import argparse
import json
import math
import statistics
from pathlib import Path

from benchmark import summary
from common import ROOT, ir_path, save_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = ROOT / "results/runs/r2_final"
    result = {"passed": True, "runs": {}}
    for name in ("main_parent", "main_baseline", "repeat_parent", "repeat_baseline"):
        path = root / (name + ".json")
        data = json.loads(path.read_text())
        for group, variants in data["artifacts"].items():
            for variant, artifact in variants.items():
                assert artifact["input_sha256"] == sha256(ir_path(group, variant))
                folder = ROOT / "artifacts" / group / variant / artifact["hash"]
                for stage, expected in artifact["stages"].items():
                    assert sha256(folder / ("kernel." + stage)) == expected
        for row in [*data["records"], *data["controls"]]:
            assert len(row["pairs"]) == data["rounds"]
            for pair in row["pairs"]:
                raw = pair["graph_replay_samples_us"]
                reference = data["performance_reference"]
                other = next(k for k in raw if k != reference)
                assert math.isclose(
                    pair["baseline_us"], statistics.mean(raw[reference]), rel_tol=1e-12
                )
                assert math.isclose(
                    pair["candidate_us"], statistics.mean(raw[other]), rel_tol=1e-12
                )
                assert math.isclose(
                    pair["speedup"],
                    pair["baseline_us"] / pair["candidate_us"],
                    rel_tol=1e-12,
                )
            assert row["median_speedup"] == statistics.median(
                p["speedup"] for p in row["pairs"]
            )
        assert data["aa_passed"] and all(
            0.97 <= c["median_speedup"] <= 1.03 for c in data["controls"]
        )
        rows = [r for r in data["records"] if r["name"] == "r2_select_after"]
        assert len(rows) == 30
        recomputed = summary(rows)
        assert (
            recomputed["geometric"] == data["summaries"]["r2_select_after"]["geometric"]
        )
        result["runs"][name] = {
            "sha256": sha256(path),
            "recomputed": recomputed,
            "physical_uuid": data["physical_uuid"],
        }
    save_json(args.output, result)


if __name__ == "__main__":
    main()
