"""Reproduce every round-three input and audit real paired timing/stage evidence."""

import argparse
import json
import math
import random
import statistics
from pathlib import Path

from audit_artifacts import text_sections
from benchmark import bootstrap, summary
from common import ROOT, ir_path, manifest, save_json, sha256
from make_round3 import (
    cache_load,
    full_inner,
    full_inner65536,
    narrow_noninner,
    split_reduce,
)
from make_round3_half import half_max
from make_round3_mix import half_noninner, mixed


def arithmetic_interval(rows, count=5000):
    rng = random.Random(9182)
    draws = []
    for _ in range(count):
        ratios = [
            statistics.median(
                rng.choices([p["speedup"] for p in row["pairs"]], k=len(row["pairs"]))
            )
            for row in rows
        ]
        draws.append(statistics.mean(ratios))
    draws.sort()
    return [draws[int(count * 0.025)], draws[int(count * 0.975)]]


def candidate_transforms():
    return {
        "r3_full64": full_inner65536,
        "r3_full128": full_inner,
        "r3_narrow32": lambda t, c: narrow_noninner(t, c, 32),
        "r3_narrow64": lambda t, c: narrow_noninner(t, c, 64),
        "r3_cachecg": lambda t, c: cache_load(t),
        "r3_fold": split_reduce,
        "r3_halfmax": lambda t, c: half_max(t),
        "r3_full64_half": lambda t, c: half_max(full_inner65536(t, c)),
    }


def audit_static():
    output = {}
    for group, config in manifest()["groups"].items():
        parent = ir_path(group, "r2_select_after").read_text()
        for name, fn in candidate_transforms().items():
            assert fn(parent, config) == ir_path(group, name).read_text()
        mix = mixed(group, config)
        assert mix == ir_path(group, "r3_mix").read_text()
        assert half_noninner(mix) == ir_path(group, "r3_mix_halfnon").read_text()
        output[group] = {
            n: sha256(ir_path(group, n))
            for n in [*candidate_transforms(), "r3_mix", "r3_mix_halfnon"]
        }
    return output


def audit_timing(path, candidate):
    data = json.loads(path.read_text())
    assert data["aa_passed"] and len(data["controls"]) == 30
    for row in [*data["records"], *data["controls"]]:
        assert len(row["pairs"]) == data["rounds"]
        for pair in row["pairs"]:
            raw = pair["graph_replay_samples_us"]
            ref = data["performance_reference"]
            other = next(k for k in raw if k != ref)
            assert math.isclose(
                pair["baseline_us"], statistics.mean(raw[ref]), rel_tol=1e-12
            )
            assert math.isclose(
                pair["candidate_us"], statistics.mean(raw[other]), rel_tol=1e-12
            )
            assert math.isclose(
                pair["speedup"],
                pair["baseline_us"] / pair["candidate_us"],
                rel_tol=1e-12,
            )
        assert row["median_speedup"] == statistics.median(
            p["speedup"] for p in row["pairs"]
        )
    for group, variants in data["artifacts"].items():
        for name, kernel in variants.items():
            assert kernel["input_sha256"] == sha256(ir_path(group, name))
            folder = ROOT / "artifacts" / group / name / kernel["hash"]
            for stage, expected in kernel["stages"].items():
                assert sha256(folder / ("kernel." + stage)) == expected
    rows = [r for r in data["records"] if r["name"] == candidate]
    assert len(rows) == 30
    stats = summary(rows)
    assert stats["geometric"] == data["summaries"][candidate]["geometric"]
    return dict(
        path=str(path.relative_to(ROOT)),
        sha256=sha256(path),
        physical_uuid=data["physical_uuid"],
        reference=data["performance_reference"],
        summary=dict(**stats, bootstrap_95=bootstrap(rows)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    output = dict(
        passed=True, candidate_reproduction=audit_static(), runs={}, resources={}
    )
    folder = ROOT / "results/runs/r3_final"
    if not args.static_only:
        for name in (
            "main_baseline",
            "main_parent",
            "repeat_baseline",
            "repeat_parent",
        ):
            filename = name + "_v2" if name.startswith("repeat_") else name
            output["runs"][name] = audit_timing(folder / (filename + ".json"), "r3_mix")
        assert len({d["physical_uuid"] for d in output["runs"].values()}) == 1
    pipeline = json.loads((folder / "pipeline.json").read_text())
    for group, variants in pipeline["kernels"].items():
        output["resources"][group] = {}
        for name, kernel in variants.items():
            path = ROOT / "artifacts" / group / name / kernel["hash"] / "kernel.cubin"
            output["resources"][group][name] = dict(
                registers=kernel["n_regs"],
                spills=kernel["n_spills"],
                shared=kernel["metadata"]["shared"],
                text_sections=text_sections(path),
            )
    save_json(args.output, output)
    print("PASS deterministic TTIR reproduction / stage identity / paired sample audit")


if __name__ == "__main__":
    main()
