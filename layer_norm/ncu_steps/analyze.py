"""Audit launch ordering, units and immutable IR before using NCU evidence."""

import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OP = ROOT.parent
R = ROOT / "results"
NAMES = [
    "baseline",
    "abl_arithmetic",
    "abl_grouped",
    "abl_unroll",
    "abl_ca",
    "abl_cs",
    "abl_final",
]
SIZES = [1024, 2560, 4096, 15872]
METRICS = [
    "gpu__time_duration.sum",
    "launch__registers_per_thread",
    "launch__occupancy_limit_registers",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "dram__bytes.sum.per_second",
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    csv_path = R / "steps_20261124.csv"
    raw = list(csv.DictReader(csv_path.open()))
    units = raw[0]
    assert units["ID"] == ""
    rows = [r for r in raw if r["ID"].isdigit()]
    assert len(rows) == 28 and [int(r["ID"]) for r in rows] == list(range(28))
    launches = re.findall(
        r"(?m)^(baseline|abl_\w+) (\d+) ([\d.]+)\s*$",
        (R / "steps_20261124.log").read_text(),
    )
    assert [(name, int(n)) for name, n, _ in launches] == [
        (name, n) for n in SIZES for name in NAMES
    ]
    assert all(float(error) < 1e-2 for _, _, error in launches)
    accuracy = json.loads((R / "accuracy_20261124.json").read_text())
    pipeline = json.loads((R / "pipeline_20261124.json").read_text())
    assert accuracy["passed"] and len(accuracy["records"]) == 56
    assert pipeline["passed"]
    for name in NAMES:
        assert accuracy["kernels"][name] == sha(OP / "ir" / f"{name}.ttir")
    records = []
    for row, (name, n, _) in zip(rows, launches):
        assert row["Block Size"] == "(256, 1, 1)" and row["Grid Size"] == "(4096, 1, 1)"
        records.append(
            dict(
                N=int(n),
                variant=name,
                metrics={
                    metric: dict(
                        value=float(row[metric].replace(",", "")), unit=units[metric]
                    )
                    for metric in METRICS
                },
            )
        )
    summary = dict(
        evidence_grade="C",
        purpose="Mechanism clues; not graph benchmark speedups",
        clock_control="base",
        cache_control="all",
        shapes=[[4096, n] for n in SIZES],
        variants=NAMES,
        verified_profile_launches=28,
        smoke_checks=56,
        records=records,
        ir_sha256={n: sha(OP / "ir" / f"{n}.ttir") for n in NAMES},
        sources={
            str(p.relative_to(OP)): sha(p)
            for p in [
                csv_path,
                R / "steps_20261124.ncu-rep",
                R / "steps_20261124.log",
                R / "accuracy_20261124.json",
                R / "pipeline_20261124.json",
                OP / "results/ablation/resources.json",
            ]
        },
    )
    output = R / "summary.json"
    if output.exists():
        assert json.loads(output.read_text()) == summary
    else:
        output.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        "Audited 28 NCU launches, units, 56 smoke checks, original pipeline and all input IR hashes"
    )


if __name__ == "__main__":
    main()
