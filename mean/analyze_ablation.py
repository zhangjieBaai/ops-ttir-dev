"""Derive factorial effects, interactions and raw-sample checks from blocked timings."""

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np
from ablation_common import EDGES, NAMES
from common import PERF_SHAPES, ROOT, ir_path, save_json, sha256
from make_ablation import expected


def summarize(points, label):
    ratios = np.array([point["contrasts"][label]["paired_ratios"] for point in points])
    medians = np.median(ratios, axis=1)
    rng = np.random.default_rng(73564)
    selections = rng.integers(
        0, ratios.shape[1], size=(5000, ratios.shape[0], ratios.shape[1])
    )
    draws = np.take_along_axis(ratios[None, :, :], selections, axis=2)
    geo = np.exp(np.log(np.median(draws, axis=2)).mean(axis=1))
    return dict(
        geomean=float(np.exp(np.log(medians).mean())),
        ci95=np.quantile(geo, [0.025, 0.975]).tolist(),
        worst=float(medians.min()),
        best=float(medians.max()),
        points=len(points),
        worst_shape=points[int(np.argmin(medians))]["shape"],
    )


def validate_run(data, accuracy, pipeline):
    assert data["names"] == NAMES
    assert data["contrasts"] == [list(edge) for edge in EDGES]
    assert [tuple(point["shape"]) for point in data["points"]] == PERF_SHAPES
    assert data["aa_passed"] and all(point["aa"]["passed"] for point in data["points"])
    assert data["plan_sha256"] == sha256(ROOT / "ablation_plan.md")
    assert data["manifest_sha256"] == sha256(ROOT / "results/ablation_manifest.json")
    for group, versions in data["identities"].items():
        for name, info in versions.items():
            assert info["input_sha256"] == sha256(ir_path(group, name))
            assert expected(group, name) == ir_path(group, name).read_text()
            for receipt in (accuracy, pipeline):
                expected_info = receipt["kernels"][group][name]
                assert info["input_sha256"] == expected_info["input_sha256"]
                assert info["stages"]["cubin"] == expected_info["stages"]["cubin"]
                assert info["metadata"] == expected_info["metadata"]
    for point in data["points"]:
        assert len(point["blocks"]) == data["rounds"]
        for block in point["blocks"]:
            assert sorted(block["order"]) == sorted([*NAMES, "A/A"])
            for name in block["order"]:
                raw = block["samples_us"][name]
                assert raw and all(math.isfinite(t) and t > 0 for t in raw)
                assert statistics.mean(raw) == block["times_us"][name]
        for label, before, after in EDGES:
            ratios = [
                block["times_us"][before] / block["times_us"][after]
                for block in point["blocks"]
            ]
            assert ratios == point["contrasts"][label]["paired_ratios"]
            assert statistics.median(ratios) == point["contrasts"][label]["median"]
        aa = [
            block["times_us"]["baseline"] / block["times_us"]["A/A"]
            for block in point["blocks"]
        ]
        assert aa == point["aa"]["paired_ratios"]
        assert point["aa"]["median"] == statistics.median(aa)
        assert 0.97 <= point["aa"]["median"] <= 1.03


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/runs/ablation_v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def read(name):
        return json.loads((args.run / f"{name}.json").read_text())

    pipeline = read("pipeline")
    assert pipeline["passed"] and len(pipeline["points"]) == 42
    runs = {name: read(name) for name in ("main", "repeat")}
    accuracies = {name: read(name) for name in ("accuracy", "accuracy_repeat")}
    assert all(
        item["passed"] and len(item["records"]) == 42 * 8 * 13
        for item in accuracies.values()
    )
    results = {}
    for label, run in runs.items():
        accuracy = accuracies["accuracy" if label == "main" else "accuracy_repeat"]
        validate_run(run, accuracy, pipeline)
        partitions = {
            "all30": run["points"],
            "vector18": [point for point in run["points"] if point["shape"][2] >= 1024],
            "oddK": [
                point
                for point in run["points"]
                if point["shape"][2] >= 1024 and point["shape"][2] % 16
            ],
            "aligned_vector": [
                point
                for point in run["points"]
                if point["shape"][2] >= 1024 and point["shape"][2] % 16 == 0
            ],
        }
        results[label] = {
            part: {edge[0]: summarize(points, edge[0]) for edge in EDGES}
            for part, points in partitions.items()
        }
    assert runs["main"]["rounds"] == 7 and runs["repeat"]["rounds"] == 5
    assert (
        len(
            {
                item["physical_uuid"]
                for item in [pipeline, *runs.values(), *accuracies.values()]
            }
        )
        == 1
    )
    comparisons = []
    for group, versions in pipeline["kernels"].items():
        if (
            "@mean_dim_kernel_non_inner_vec"
            not in ir_path(group, "baseline").read_text()
        ):
            continue
        comparisons.append(
            dict(
                group=group,
                flat64_equals_baseline=versions["ab_f1i0"]["stages"]["cubin"]
                == versions["baseline"]["stages"]["cubin"],
                flat32_equals_narrow32=versions["ab_f1i1"]["stages"]["cubin"]
                == versions["ab_f0i1"]["stages"]["cubin"],
                narrow32_equals_round1=versions["ab_f0i1"]["stages"]["cubin"]
                == versions["optimized"]["stages"]["cubin"],
                variants={
                    name: dict(
                        registers=info["n_regs"],
                        spills=info["n_spills"],
                        shared=info["metadata"]["shared"],
                        cubin_sha256=info["stages"]["cubin"],
                    )
                    for name, info in versions.items()
                    if name in NAMES
                },
            )
        )
    per_shape = []
    for main_point, repeat_point in zip(
        runs["main"]["points"], runs["repeat"]["points"]
    ):
        per_shape.append(
            dict(
                shape=main_point["shape"],
                main={
                    key: value["median"]
                    for key, value in main_point["contrasts"].items()
                },
                repeat={
                    key: value["median"]
                    for key, value in repeat_point["contrasts"].items()
                },
            )
        )
    save_json(
        args.output,
        dict(
            passed=True,
            plan_sha256=sha256(ROOT / "ablation_plan.md"),
            evidence={
                str(path.relative_to(ROOT)): sha256(path)
                for path in args.run.glob("*.json")
            },
            effects=results,
            resources_and_binary=comparisons,
            per_shape=per_shape,
            interpretation="Each ratio is paired before/after; effects depend on context; not additive percentages",
        ),
    )
    for part in ("all30", "vector18", "oddK"):
        print(part)
        for label, _, _ in EDGES:
            print(
                label,
                results["main"][part][label]["geomean"],
                results["repeat"][part][label]["geomean"],
            )


if __name__ == "__main__":
    main()
