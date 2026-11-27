"""Print measured per-stage gains for the D128 decomposition."""

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    ("baseline", "rows64", "串行两个 64-key 子块"),
    ("rows64", "structure", "指针重算 / 两次子块静态展开"),
    ("structure", "final", "在既有结构上后移 dV"),
    ("baseline", "final", "原始基线至最终实现（同场）"),
]


def load(name):
    data = json.loads((ROOT / f"{name}.json").read_text())
    points = data["points"]
    assert len(points) == len({tuple(row["shape"]) for row in points}) == 16
    assert all(len(row["samples"]) == data["rounds"] for row in points)
    assert all(
        row["outputs_bitwise_equal"] and row["source_pipeline_equal"] for row in points
    )
    return points


def ratio(point, a, b):
    return statistics.median(s["us"][a] / s["us"][b] for s in point["samples"])


if __name__ == "__main__":
    runs = [load("main"), load("repeat")]
    for a, b, label in STEPS:
        cells = []
        for causal in (False, True):
            for run in runs:
                values = [ratio(row, a, b) for row in run if row["shape"][-1] == causal]
                assert len(values) == 8
                cells.append(f"{statistics.geometric_mean(values):.4f}×")
        print("| " + " | ".join([label] + cells) + " |")
    for run in runs:
        aa = [ratio(row, "baseline", "baseline_copy") for row in run]
        print(f"A/A {min(aa):.4f}–{max(aa):.4f}")
