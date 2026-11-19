"""Archive static dX resource/SASS clues and NCU metrics without timing claims."""

import argparse
import csv
import json
import re
import subprocess
from collections import Counter

from common import BLOCKS, ROOT, digest, save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ncu-csv")
    p.add_argument("--traffic-csv")
    p.add_argument("--ncu-names", nargs="+")
    args = p.parse_args()
    r = ROOT / "results/round2"
    resources = {}
    names = [
        "baseline",
        "r1_parent",
        "r2_reload_x",
        "r2_reload_wdy",
        "r2_reload_all",
        "r2_loop2048",
        "r2_loop4096",
        "r2_fold_reload",
        "r2_range2048",
        "r2_range4096",
    ]
    for block in BLOCKS:
        for name in names:
            folder = ROOT / f"artifacts/b{block}/{name}"
            path = folder / "metadata.json"
            if not path.exists():
                continue
            m = json.load(open(path))
            sass = subprocess.check_output(
                ["cuobjdump", "--dump-sass", str(folder / "kernel.cubin")], text=True
            )
            ops = Counter(
                re.findall(r"/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9.]*)", sass)
            )
            resources[f"{name}/b{block}"] = dict(
                n_regs=m["n_regs"],
                n_spills=m["n_spills"],
                shared=m["shared"],
                ttir_sha256=digest(ROOT / f"ir/b{block}/{name}.ttir"),
                cubin_sha256=digest(folder / "kernel.cubin"),
                static_instruction_counts=dict(ops),
                note="Static counts include loop body once, not runtime executed counts",
            )
    save(r / "binary_analysis.json", resources)
    if args.ncu_csv:
        rows = list(csv.DictReader(open(args.ncu_csv)))
        unit = rows[0]
        keys = [
            "gpu__time_duration.sum",
            "launch__registers_per_thread",
            "launch__occupancy_limit_registers",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
            "dram__bytes.sum.per_second",
            "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
            "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
            "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
        ]
        out = []
        for row in rows:
            if not row["ID"].isdigit():
                continue
            i = len(out)
            names = ["baseline"] + args.ncu_names
            out.append(
                dict(
                    N=[1024, 3072, 4096, 8192, 8704, 15872][i // len(names)],
                    kernel=names[i % len(names)],
                    metrics={k: dict(value=row[k], unit=unit[k]) for k in keys},
                )
            )
        save(r / "ncu_summary.json", out)
    if args.traffic_csv:
        rows = list(csv.DictReader(open(args.traffic_csv)))
        units = rows[0]
        byte_scale = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9}
        time_scale = {"ns": 1e-3, "us": 1, "ms": 1e3}
        out = []
        for row in rows:
            if not row["ID"].isdigit():
                continue
            i = len(out)

            def number(key):
                return float(row[key].replace(",", ""))

            out.append(
                dict(
                    N=[1024, 3072, 4096, 8192, 8704, 15872][i // 3],
                    kernel=["baseline", "r1_parent", "r2_loop2048"][i % 3],
                    read_bytes=number("dram__bytes_read.sum")
                    * byte_scale[units["dram__bytes_read.sum"]],
                    write_bytes=number("dram__bytes_write.sum")
                    * byte_scale[units["dram__bytes_write.sum"]],
                    us=number("gpu__time_duration.sum")
                    * time_scale[units["gpu__time_duration.sum"]],
                    raw_units={
                        k: units[k]
                        for k in [
                            "gpu__time_duration.sum",
                            "dram__bytes_read.sum",
                            "dram__bytes_write.sum",
                        ]
                    },
                )
            )
        assert len(out) == 18
        save(r / "traffic_summary.json", out)
    print("Static and profiler evidence archived")


if __name__ == "__main__":
    main()
