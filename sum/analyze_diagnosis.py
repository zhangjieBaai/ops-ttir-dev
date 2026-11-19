"""Recompute diagnostic tables from retained NCU CSV, timing JSON and nsys SQLite."""

import argparse
import csv
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

from common import ROOT, save_json

RUN = ROOT / "results/runs/diagnosis_20261119"


def metrics(path):
    records = defaultdict(dict)
    with Path(path).open() as stream:
        for row in csv.DictReader(stream):
            if row["Metric Name"]:
                raw = row["Metric Value"]
                try:
                    value = float(raw.replace(",", ""))
                except ValueError:
                    value = raw
                records[int(row["ID"])][row["Metric Name"]] = dict(
                    value=value,
                    unit=row["Metric Unit"],
                )
    return dict(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    timing = json.loads((RUN / "stage_timing.json").read_text())
    assert timing["uncontended"]
    stage_rows = []
    for record in timing["records"]:
        values = record["median_us"]
        stage_rows.append(
            dict(
                n=record["n"],
                block=record["block"],
                stage1_ctas=record["mid"],
                stage2_ctas=1,
                **values,
                isolated_stage2_fraction_of_total=values["stage2"] / values["total"],
                effective_stage1_read_TB_s=record["n"] * 4 / (values["stage1"] * 1e6),
            )
        )
    connection = sqlite3.connect(RUN / "small_timeline.sqlite")
    kernels = connection.execute(
        "SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
    ).fetchall()
    assert len(kernels) == 200
    span = (kernels[-1][1] - kernels[0][0]) / 1000
    active = sum(end - start for start, end in kernels) / 1000
    gaps = [(kernels[i + 1][0] - kernels[i][1]) / 1000 for i in range(199)]
    bindings = []
    for pattern in ["*_binding.json", "*_binding_*.json"]:
        for path in sorted(RUN.glob(pattern)):
            data = json.loads(path.read_text())
            assert data["uncontended"], path
            bindings.append(
                dict(path=path.name, gpu_uuid=data["gpu_uuid"], records=data["records"])
            )
    result = dict(
        evidence_grade="C: diagnostic counters/time decomposition, no new candidate promotion",
        normal_stage_timing=stage_rows,
        deep_case_order=[
            dict(id=i, n=n, name=name, stage=stage)
            for i, (n, name, stage) in enumerate(
                (n, name, stage)
                for n in [8388608, 67108864]
                for name in ["baseline", "r1_i32"]
                for stage in [1, 2]
            )
        ],
        deep_metrics=metrics(RUN / "deep_app.csv"),
        tail_case_order=[
            dict(id=i, name=name, stage=stage, n=16777215)
            for i, (name, stage) in enumerate(
                (name, stage) for name in ["baseline", "r1_cg"] for stage in [1, 2]
            )
        ],
        tail_metrics=metrics(RUN / "tail_app.csv"),
        tail_sector_metrics=metrics(RUN / "tail_sectors.csv"),
        small_nsys=dict(
            n=1024,
            calls=100,
            kernel_nodes=200,
            span_us=span,
            kernel_active_us=active,
            gap_total_us=span - active,
            gap_median_us=statistics.median(gaps),
            traced_device_us_per_call=span / 100,
        ),
        bindings=bindings,
    )
    save_json(args.output, result)
    print(json.dumps(result["normal_stage_timing"], indent=2))


if __name__ == "__main__":
    main()
