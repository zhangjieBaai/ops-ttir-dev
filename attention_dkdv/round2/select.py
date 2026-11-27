"""Freeze one candidate per existing D/causal group from full-matrix screens."""

import json
import math
import statistics

from common_round2 import ROOT, digest, save


def main():
    sources = [ROOT / "results/screen32_locked/benchmark.json"]
    result = {
        "screens": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "tested_candidate": {},
        "screen_summary": {},
    }
    for path in sources:
        data = json.loads(path.read_text())
        assert data["passed"] and data["aa_passed"] and len(data["points"]) == 32
        for group in ("d64_c0", "d64_c1", "d128_c0", "d128_c1"):
            rows = [r for r in data["points"] if r["group"] == group]
            assert len(rows) == 8
            summary = result["screen_summary"].setdefault(group, {})
            for name in rows[0]["candidates"]:
                values = [
                    r["candidates"][name]["ratios"]["parent_over_candidate"]
                    for r in rows
                ]
                summary[name] = {
                    "incremental_geo": math.exp(
                        statistics.mean(math.log(v) for v in values)
                    ),
                    "min": min(values),
                    "max": max(values),
                }
    for group, values in result["screen_summary"].items():
        result["tested_candidate"][group] = max(
            values, key=lambda n: values[n]["incremental_geo"]
        )
    path = ROOT / "selection_pre_main.json"
    assert not path.exists()
    save(path, result)
    print(result["tested_candidate"])


if __name__ == "__main__":
    main()
