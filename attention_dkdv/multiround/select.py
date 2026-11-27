"""Freeze round finalists from complete group screens."""

import json
import math
import statistics

from common_mr import ROOT, digest, save


def main():
    path = ROOT / "results/screen/benchmark.json"
    data = json.loads(path.read_text())
    assert data["passed"] and data["aa_passed"] and len(data["points"]) == 32
    output = ROOT / "selection_pre_main.json"
    assert not output.exists()
    result = {
        "screen_sha256": digest(path),
        "tested_candidate": {},
        "screen_summary": {},
    }
    for g in ("d64_c0", "d64_c1", "d128_c0", "d128_c1"):
        rows = [r for r in data["points"] if r["group"] == g]
        assert len(rows) == 8
        names = [
            n
            for n in rows[0]["ratios"]["parent_over"]
            if n not in ("baseline", "anchor", "parent")
        ]
        result["screen_summary"][g] = {}
        for name in names:
            ratios = [r["ratios"]["parent_over"][name] for r in rows]
            result["screen_summary"][g][name] = {
                "geo": math.exp(statistics.mean(math.log(x) for x in ratios)),
                "min": min(ratios),
                "max": max(ratios),
            }
        result["tested_candidate"][g] = max(
            names, key=lambda n: result["screen_summary"][g][n]["geo"]
        )
    save(output, result)
    print(result["tested_candidate"], flush=True)


if __name__ == "__main__":
    main()
