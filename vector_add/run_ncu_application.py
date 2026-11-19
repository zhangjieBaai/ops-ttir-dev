"""Re-establish cache priming on every pass for the cache-sensitive cases."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from common import ROOT, gate, receipt, save_json, sha256

METRICS = ",".join(
    [
        "gpu__time_duration.sum",
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "dram__bytes.sum.per_second",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        "lts__t_sector_hit_rate.pct",
        "lts__t_sectors_srcunit_tex_op_read.sum",
        "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
    ]
)
CASES = [
    ("baseline", 2097152),
    ("r1_split512_v2", 2097152),
    ("baseline", 4194304),
    ("r1_store_cs", 4194304),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gate()
    args.output.mkdir(parents=True)
    save_json(
        args.output / "environment.json",
        dict(
            **receipt(),
            metrics=METRICS,
            ncu_version=subprocess.check_output(["ncu", "--version"], text=True),
            scripts={
                p: sha256(ROOT / p)
                for p in (
                    "profile_ncu.py",
                    "profile_ncu_application.py",
                    "run_ncu_application.py",
                )
            },
        ),
    )
    records = []
    for name, n in CASES:
        dest = args.output / f"{name}_n{n}"
        dest.mkdir()
        report = dest / "profile.ncu-rep"
        command = [
            "ncu",
            "--profile-from-start",
            "off",
            "--kernel-name",
            "add_kernel",
            "--replay-mode",
            "application",
            "--app-replay-mode",
            "strict",
            "--graph-profiling",
            "node",
            "--launch-skip",
            "16",
            "--launch-count",
            "1",
            "--cache-control",
            "none",
            "--clock-control",
            "none",
            "--metrics",
            METRICS,
            "--export",
            str(report),
            sys.executable,
            str(ROOT / "profile_ncu_application.py"),
            name,
            "--n",
            str(n),
            "--output",
            str(dest / "binding.json"),
        ]
        # Each application replay writes a distinct process-specific receipt.
        command[-1] = str(dest / "binding_{pid}.json")
        start = time.monotonic()
        with (dest / "stdout.log").open("x") as out, (dest / "stderr.log").open(
            "x"
        ) as err:
            result = subprocess.run(command, stdout=out, stderr=err, timeout=240)
        status = (
            "COLLECTED" if result.returncode == 0 and report.exists() else "TOOL_FAILED"
        )
        if status == "COLLECTED":
            for page in ("raw", "details"):
                value = subprocess.check_output(
                    ["ncu", "--import", str(report), "--csv", "--page", page], text=True
                )
                (dest / f"{page}.csv").write_text(value)
        record = dict(
            name=name,
            n=n,
            cache_control="none",
            replay_mode="application",
            command=command,
            status=status,
            exit_code=result.returncode,
            elapsed_s=time.monotonic() - start,
            hashes={p.name: sha256(p) for p in dest.iterdir() if p.is_file()},
        )
        save_json(dest / "manifest.json", record)
        records.append(record)
        print(f"{status}: {name} N={n}, {record['elapsed_s']:.1f}s", flush=True)
        if status != "COLLECTED":
            break
    save_json(
        args.output / "index.json",
        dict(
            records=records,
            complete=len(records) == len(CASES)
            and all(r["status"] == "COLLECTED" for r in records),
        ),
    )


if __name__ == "__main__":
    main()
