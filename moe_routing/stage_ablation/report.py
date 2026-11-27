"""Audit the two full-matrix runs and print the Markdown table for the operator report."""

import hashlib
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
OP = HERE.parent
MANIFEST = OP / "multiround/r6/manifest30.json"
STAGE_NAMES = ("stage1", "stage2", "both")


def read(path):
    return json.loads(path.read_text())


def load_run(name, rounds):
    result = read(HERE / name / "performance.json")
    validation = read(HERE / name / "validation.json")
    expected = {(item["tokens"], item["experts"]) for item in read(MANIFEST)["points"]}
    assert (
        result["manifest_sha256"] == hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    )
    for key, hashes in result["stages"].items():
        e, name, i = key.split("/")
        ir = MANIFEST.parent / "ir" / f"e{e}_w4" / f"{name}_s{i}.ttir"
        assert hashes["ir_sha256"] == hashlib.sha256(ir.read_bytes()).hexdigest()
    checks = validation["accuracy"]["checks"]
    assert len(checks) == len(expected) * 4 * 8
    assert len({(*x["shape"], x["variant"], x["distribution"]) for x in checks}) == len(
        checks
    )
    assert all(row["status"] == "PASS" for row in checks)
    assert {tuple(x["shape"]) for x in checks} == expected
    assert {x["variant"] for x in checks} == {"baseline", *STAGE_NAMES}
    assert len({x["distribution"] for x in checks}) == 8
    assert {tuple(x["shape"]) for x in validation["pipeline"]} == expected
    assert result["rep_ms"] == 20 and result["rounds"] == rounds
    ratios = {}
    for row in result["rows"]:
        timing = row["timing"]
        assert len(timing["samples"]) == rounds and 0.97 <= timing["aa_median"] <= 1.03
        ratios[tuple(row["shape"])] = {
            n: statistics.median(
                s["us"]["baseline"] / s["us"][n] for s in timing["samples"]
            )
            for n in STAGE_NAMES
        }
        assert ratios[tuple(row["shape"])] == timing["baseline_over_candidate"]
    assert len(result["rows"]) == len(expected) and set(ratios) == expected
    return ratios, result["stages"]


def main():
    primary, stages = load_run("main30", 7)
    repeat, repeat_stages = load_run("repeat30", 5)
    assert stages == repeat_stages
    means = [
        {n: statistics.geometric_mean(v[n] for v in run.values()) for n in STAGE_NAMES}
        for run in (primary, repeat)
    ]
    print("| Shape (tokens, experts) | stage 1 | stage 2 | both |")
    print("|---|---:|---:|---:|")
    for shape in primary:
        cells = [
            f"{primary[shape][n]:.4f}× / {repeat[shape][n]:.4f}×" for n in STAGE_NAMES
        ]
        print(f"| {shape} | " + " | ".join(cells) + " |")
    cells = [f"{means[0][n]:.4f}× / {means[1][n]:.4f}×" for n in STAGE_NAMES]
    print("| **30 点几何平均** | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
