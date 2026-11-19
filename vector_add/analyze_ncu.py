"""Normalize actual NCU CSV units and bind diagnostic summaries to raw evidence."""

import argparse
import csv
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

from common import ROOT, save_json, sha256

FIELDS = {
    "duration_s": "gpu__time_duration.sum",
    "dram_read_bytes": "dram__bytes_read.sum",
    "dram_write_bytes": "dram__bytes_write.sum",
    "dram_bytes_per_s": "dram__bytes.sum.per_second",
    "dram_peak_pct": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "compute_peak_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_total_hit_pct": "lts__t_sector_hit_rate.pct",
    "l2_read_sectors": "lts__t_sectors_srcunit_tex_op_read.sum",
    "l2_read_hit_sectors": "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
    "occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "eligible_warps_per_scheduler": "smsp__warps_eligible.avg.per_cycle_active",
    "long_scoreboard_per_issue": "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "dynamic_warp_instructions": "smsp__inst_executed.sum",
    "registers_per_thread": "launch__registers_per_thread",
    "registers_allocated_per_thread": "launch__registers_per_thread_allocated",
    "cta_count": "launch__grid_size",
    "sm_count": "device__attribute_multiprocessor_count",
    "l2_capacity_bytes": "device__attribute_l2_cache_size",
}
SCALE = {"us": 1e-6, "ns": 1e-9, "ms": 1e-3, "s": 1.0}
for prefix, factor in (("", 1), ("K", 1e3), ("M", 1e6), ("G", 1e9), ("T", 1e12)):
    for suffix in ("byte", "byte/s", "inst", "sector", "warp"):
        SCALE[prefix + suffix] = factor
SCALE.update({"%": 1, "": 1, "register/thread": 1, "block": 1, "SM": 1})


def csv_record(path):
    with path.open() as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 3, f"Expected exactly one collected launch: {path}"
    names, units, values = rows
    assert len(names) == len(units) == len(values)
    return dict(zip(names, values)), dict(zip(names, units))


def profile_record(directory, replay_mode):
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "COLLECTED"
    for name, expected in manifest["hashes"].items():
        assert sha256(directory / name) == expected
    values, units = csv_record(directory / "raw.csv")
    assert values["Kernel Name"] == "add_kernel"
    raw, metrics = {}, {}
    for label, metric in FIELDS.items():
        if metric not in values:
            metrics[label] = None
            continue
        unit = units[metric]
        assert unit in SCALE, (metric, unit)
        raw[label] = dict(
            metric=metric, value=values[metric], unit=unit, raw_csv_line=3
        )
        metrics[label] = float(values[metric].replace(",", "")) * SCALE[unit]
    if metrics["l2_read_sectors"]:
        metrics["l2_read_hit_pct"] = (
            100 * metrics["l2_read_hit_sectors"] / metrics["l2_read_sectors"]
        )
    binding_paths = sorted(directory.glob("binding*.json"))
    assert binding_paths
    bindings = [json.loads(p.read_text()) for p in binding_paths]
    for binding in bindings:
        assert binding["name"] == manifest["name"] and binding["n"] == manifest["n"]
        assert binding["correctness_before_after"]
        assert sha256(ROOT / binding["ttir_path"]) == binding["ttir_sha256"]
        assert (
            binding["artifact"]["stages"]["cubin"]
            == bindings[0]["artifact"]["stages"]["cubin"]
        )
    return dict(
        name=manifest["name"],
        n=manifest["n"],
        replay_mode=replay_mode,
        cache_control=manifest["cache_control"],
        metrics=metrics,
        raw=raw,
        evidence_dir=str(directory.resolve().relative_to(ROOT)),
        raw_csv_sha256=sha256(directory / "raw.csv"),
        manifest_sha256=sha256(directory / "manifest.json"),
        ttir_sha256=bindings[0]["ttir_sha256"],
        cubin_sha256=bindings[0]["artifact"]["stages"]["cubin"],
        binding_files=[p.name for p in binding_paths],
    )


def static_records(output):
    kernels = json.loads(
        (ROOT / "results/runs/r1_20261119/compile_v3.json").read_text()
    )["kernels"]
    records = {}
    for name in ("baseline", "r1_store_cs", "r1_store_evict_first", "r1_split512_v2"):
        kernel = kernels[name]
        directory = ROOT / "artifacts" / name / kernel["hash"]
        assert sha256(directory / "kernel.cubin") == kernel["stages"]["cubin"]
        sass = subprocess.check_output(
            ["cuobjdump", "--dump-sass", str(directory / "kernel.cubin")], text=True
        )
        (output / f"{name}.sass").write_text(sass)
        instructions = [
            match[1]
            for line in sass.splitlines()
            if (
                match := re.search(
                    r"/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)", line
                )
            )
        ]
        ptx = (directory / "kernel.ptx").read_text()
        records[name] = dict(
            registers=kernel["n_regs"],
            spills=kernel["n_spills"],
            shared=kernel["metadata"]["shared"],
            text_sections=kernel["text_sections"],
            cubin_sha256=kernel["stages"]["cubin"],
            opcodes=dict(Counter(instructions)),
            static_instruction_slots=len(instructions),
            ptx_memory=[
                line.strip()
                for line in ptx.splitlines()
                if re.search(r"\b(ld|st)\.global", line)
            ],
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-matrix", type=Path, required=True)
    parser.add_argument("--application-matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    profiles = []
    for directory, mode in (
        (args.kernel_matrix, "kernel"),
        (args.application_matrix, "application"),
    ):
        index = json.loads((directory / "index.json").read_text())
        assert index["complete"]
        profiles.extend(
            profile_record(p.parent, mode)
            for p in sorted(directory.glob("*/manifest.json"))
        )
    save_json(
        args.output / "analysis.json",
        dict(
            evidence_grade="C: selected single-launch counters; no performance acceptance claim",
            script_sha256=sha256(Path(__file__)),
            profiles=profiles,
            static=static_records(args.output),
        ),
    )
    print(f"Verified and normalized {len(profiles)} profiles")


if __name__ == "__main__":
    main()
