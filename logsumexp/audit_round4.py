"""Reproduce round-four TTIR candidates and independently audit paired evidence."""

import argparse
import json
from pathlib import Path

from audit_artifacts import text_sections
from audit_round3 import arithmetic_interval, audit_timing
from benchmark import bootstrap, summary
from common import PERF_SHAPES, ROOT, ir_path, manifest, save_json, sha256
from make_candidates import narrow_indices
from make_round4 import row_group, transpose_noninner
from make_round4_focus import focus
from make_round4_lane import lane32
from make_round4_streams import stream128 as stream128_failed
from make_round4_streams_v2 import stream128 as stream128_scope_failed
from make_round4_streams_v3 import stream128

FLAT_SHAPES = [(m, n, 1) for m in (1, 64, 1024) for n in (128, 1008, 4096, 8192)] + [
    (1, 8192, 64),
    (16, 2048, 64),
    (256, 128, 64),
]


def transforms():
    return {
        "r4_rows2": lambda t, c: row_group(t, c, 2, 1024),
        "r4_rows4": lambda t, c: row_group(t, c, 4, 1024),
        "r4_rows8": lambda t, c: row_group(t, c, 8, 1024),
        "r4_rows4wide": lambda t, c: row_group(t, c, 4, 8192),
        "r4_transpose": transpose_noninner,
        "r4_i32non": lambda t, c: narrow_indices(t) if "TILE_K" in c["meta"] else t,
        "r4_lane32": lane32,
        "r4_focus": focus,
        "r4_stream128": stream128_failed,
        "r4_stream128_v2": stream128_scope_failed,
        "r4_stream128_v3": stream128,
    }


def static_audit():
    result = {}
    for group, config in manifest()["groups"].items():
        parent = ir_path(group, "r3_mix").read_text()
        for name, fn in transforms().items():
            assert fn(parent, config) == ir_path(group, name).read_text()
        result[group] = {n: sha256(ir_path(group, n)) for n in transforms()}
    return result


def subset_summary(rows):
    subset = [r for r in rows if tuple(r["shape"]) in FLAT_SHAPES]
    assert len(subset) == 15
    return dict(
        **summary(subset),
        bootstrap_95=bootstrap(subset),
        arithmetic_bootstrap_95=arithmetic_interval(subset)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--candidate", default="r4_selected")
    args = parser.parse_args()
    result = dict(
        passed=True, candidate_reproduction=static_audit(), runs={}, resources={}
    )
    folder = ROOT / "results/runs/r4_final"
    if not args.static_only:
        for name in (
            "main_baseline",
            "main_parent",
            "repeat_baseline",
            "repeat_parent",
        ):
            path = folder / (name + ".json")
            result["runs"][name] = audit_timing(path, args.candidate)
            data = json.loads(path.read_text())
            rows = [r for r in data["records"] if r["name"] == args.candidate]
            assert {tuple(r["shape"]) for r in rows} == set(PERF_SHAPES)
            result["runs"][name]["flat_subset"] = subset_summary(rows)
        assert len({v["physical_uuid"] for v in result["runs"].values()}) == 1
    pipeline = json.loads(
        (ROOT / "results/runs/r4_setup/pipeline_all.json").read_text()
    )
    for group, variants in pipeline["kernels"].items():
        result["resources"][group] = {}
        for name, k in variants.items():
            cubin = ROOT / "artifacts" / group / name / k["hash"] / "kernel.cubin"
            result["resources"][group][name] = dict(
                registers=k["n_regs"],
                spills=k["n_spills"],
                shared=k["metadata"]["shared"],
                text=text_sections(cubin),
            )
    save_json(args.output, result)


if __name__ == "__main__":
    main()
