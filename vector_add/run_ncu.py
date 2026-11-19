"""Collect a bounded, serialized diagnostic matrix and retain all raw reports."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from common import ROOT, gate, receipt, save_json, sha256

SECTIONS = [
    "SpeedOfLight",
    "LaunchStats",
    "Occupancy",
    "MemoryWorkloadAnalysis",
    "SchedulerStats",
    "WarpStateStats",
    "InstructionStats",
]
EXTRA_METRICS = ",".join(
    [
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "dram__bytes.sum.per_second",
        "lts__t_sector_hit_rate.pct",
    ]
)
MATRIX = [
    ("baseline", n, "none")
    for n in (4096, 65536, 1048576, 2097152, 4194304, 8388608, 134217728)
] + [
    ("r1_store_cs", 4194304, "none"),
    ("r1_store_evict_first", 4194304, "none"),
    ("r1_store_cs", 134217728, "none"),
    ("r1_split512_v2", 4096, "none"),
    ("r1_split512_v2", 1048576, "none"),
    ("r1_split512_v2", 2097152, "none"),
    ("baseline", 4194304, "all"),
    ("r1_store_cs", 4194304, "all"),
    ("baseline", 134217728, "all"),
]


def collect(output, name, n, cache):
    dest = output / f"{name}_n{n}_cache_{cache}"
    dest.mkdir()
    command = [
        "ncu",
        "--profile-from-start",
        "off",
        "--kernel-name",
        "add_kernel",
        "--replay-mode",
        "kernel",
        "--graph-profiling",
        "node",
        "--launch-skip",
        "16",
        "--launch-count",
        "1",
        "--cache-control",
        cache,
        "--clock-control",
        "none",
        "--export",
        str(dest / "profile.ncu-rep"),
    ]
    for section in SECTIONS:
        command.extend(["--section", section])
    command.extend(
        [
            "--metrics",
            EXTRA_METRICS,
            sys.executable,
            str(ROOT / "profile_ncu.py"),
            name,
            "--n",
            str(n),
            "--output",
            str(dest / "binding.json"),
        ]
    )
    start = time.monotonic()
    with (dest / "stdout.log").open("x") as stdout, (dest / "stderr.log").open(
        "x"
    ) as stderr:
        result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=240)
    report = dest / "profile.ncu-rep"
    status = (
        "COLLECTED" if result.returncode == 0 and report.exists() else "TOOL_FAILED"
    )
    if status == "COLLECTED":
        for page in ("raw", "details"):
            export = subprocess.run(
                ["ncu", "--import", str(report), "--csv", "--page", page],
                capture_output=True,
                text=True,
                check=True,
            )
            (dest / f"{page}.csv").write_text(export.stdout)
            (dest / f"{page}.stderr").write_text(export.stderr)
    record = dict(
        name=name,
        n=n,
        cache_control=cache,
        clock_control="none",
        command=command,
        exit_code=result.returncode,
        status=status,
        elapsed_s=time.monotonic() - start,
        hashes={p.name: sha256(p) for p in dest.iterdir() if p.is_file()},
    )
    save_json(dest / "manifest.json", record)
    print(
        f"{status}: {name}, N={n}, cache={cache}, {record['elapsed_s']:.1f}s",
        flush=True,
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    gate()
    args.output.mkdir(parents=True)
    save_json(
        args.output / "environment.json",
        dict(
            **receipt(),
            ncu_version=subprocess.check_output(["ncu", "--version"], text=True),
            sections=SECTIONS,
            metrics=EXTRA_METRICS,
            scripts={p: sha256(ROOT / p) for p in ("profile_ncu.py", "run_ncu.py")},
        ),
    )
    matrix = MATRIX[:1] if args.pilot else MATRIX
    records = []
    for name, n, cache in matrix:
        record = collect(args.output, name, n, cache)
        records.append(record)
        if record["status"] != "COLLECTED":
            break
    save_json(
        args.output / "index.json",
        dict(
            records=records,
            complete=len(records) == len(matrix)
            and all(r["status"] == "COLLECTED" for r in records),
        ),
    )


if __name__ == "__main__":
    main()
