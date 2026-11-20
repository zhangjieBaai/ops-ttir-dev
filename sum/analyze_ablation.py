"""Independently recompute ablation ratios from all retained replay samples."""

import argparse
import json
import math
import statistics
from pathlib import Path

from ablation_common import BASE, DIRECT, EDGES, PUBLIC, text_sections
from benchmark import summarize
from common import PERF_SIZES, ROOT, save_json, sha256


def recompute(path):
    data = json.loads(path.read_text())
    assert data["uncontended"] and data["aa_passed"]
    assert len(data["controls"]) == len(PERF_SIZES)
    assert [r["n"] for r in data["controls"]] == PERF_SIZES
    rows = []
    for original in [*data["records"], *data["controls"]]:
        ratios = []
        for pair in original["pairs"]:
            raw = pair["replay_samples_us"]
            left, right = [
                statistics.mean(raw[role]) for role in ["reference", "candidate"]
            ]
            ratio = left / right
            assert math.isclose(pair["baseline_us"], left, rel_tol=1e-12)
            assert math.isclose(pair["candidate_us"], right, rel_tol=1e-12)
            assert math.isclose(pair["speedup"], ratio, rel_tol=1e-12)
            ratios.append(ratio)
        assert len(ratios) == data["rounds"]
        assert math.isclose(
            original["median_speedup"], statistics.median(ratios), rel_tol=1e-12
        )
        if "name" in original:
            rows.append(original)
        else:
            assert 0.97 <= statistics.median(ratios) <= 1.03
    summaries = {}
    for left, right in EDGES:
        label = left + "->" + right
        selected = [row for row in rows if row["name"] == label]
        assert [row["n"] for row in selected] == PERF_SIZES
        summaries[label] = {}
        for scope, subset in [
            ("full", selected),
            ("small", [r for r in selected if r["n"] <= 16384]),
            ("large", [r for r in selected if r["n"] > 16384]),
        ]:
            metrics = summarize(subset)
            assert metrics == data["summaries"][label][scope]
            summaries[label][scope] = dict(
                **metrics,
                geometric_gain_percent=100 * (metrics["geometric"] - 1),
                cumulative_latency_ratio=sum(r["baseline_us"] for r in subset)
                / sum(r["candidate_us"] for r in subset),
            )
    return dict(
        path=str(path.relative_to(ROOT)),
        sha256=sha256(path),
        gpu_uuid=data["gpu_uuid"],
        rounds=data["rounds"],
        summaries=summaries,
        raw_pairs_checked=len([*data["records"], *data["controls"]]) * data["rounds"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, default=BASE / "results/main.json")
    parser.add_argument("--repeat", type=Path, default=BASE / "results/repeat.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    results = {"main": recompute(args.main), "repeat": recompute(args.repeat)}
    assert results["main"]["gpu_uuid"] == results["repeat"]["gpu_uuid"]
    manifest = json.loads((BASE / "manifest.json").read_text())
    results["public_equivalence"] = {}
    for name in PUBLIC:
        checks = [r for r in manifest["checks"] if r["name"] == name]
        results["public_equivalence"][name] = dict(
            stage_instances=len(checks),
            executable_text_identical=sum(
                r["executable_text_identical"] for r in checks
            ),
            source_vs_file_all_stages_equal=all(
                all(r["file_pipeline_identical"].values()) for r in checks
            ),
        )
    results["public_failures"] = manifest["public_failures"]
    accuracy = json.loads((BASE / "results/accuracy.json").read_text())
    results["accuracy_checks"] = len(accuracy["records"])
    results["exact_public_output_comparisons"] = sum(
        r["output_exact"] for r in accuracy["public_vs_direct"]
    )
    results["public_comparisons"] = len(accuracy["public_vs_direct"])
    # Check the .cs attribute's executable effect against actual retained cubins.
    pipeline = json.loads((BASE / "results/pipeline.json").read_text())
    identical = 0
    for n in map(str, PERF_SIZES):
        for left, right in zip(
            pipeline["artifacts"]["baseline"][n], pipeline["artifacts"]["cs"][n]
        ):
            binaries = [
                (ROOT / "artifacts/round2" / r["hash"] / "kernel.cubin").read_bytes()
                for r in [left, right]
            ]
            identical += text_sections(binaries[0]) == text_sections(binaries[1])
    results["cs_vs_baseline_text_identical_instances"] = identical
    results["measured_direct_variants"] = DIRECT
    save_json(args.output, results)
    for name, summary in results["main"]["summaries"].items():
        print(
            name,
            summary["full"]["geometric"],
            results["repeat"]["summaries"][name]["full"]["geometric"],
        )


if __name__ == "__main__":
    main()
