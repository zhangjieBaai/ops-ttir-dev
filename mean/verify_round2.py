"""Read-only reconstruction of round2 candidates from immutable round1 TTIR."""

import argparse
import json
from pathlib import Path

from common import ALL_SHAPES, ROOT, ir_path, manifest, save_json, sha256
from make_round2 import VARIANTS, branch, build
from make_round2_combined import combined


def expected(group, config):
    base = ir_path(group, "optimized").read_text()
    if "@mean_dim_kernel_non_inner_vec" not in base:
        return base
    result = build(base, **config)
    if "rows" in config:
        header = next(line for line in base.splitlines() if "tt.func public" in line)
        if "%N:" not in header:
            return base
        result = branch(
            base,
            result,
            "    %limit = arith.constant 128 : i32\n    %take = arith.cmpi sge, %N, %limit : i32",
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for m, n, k in ALL_SHAPES:
        # Conservative bound includes masked lanes of the 256-row tail path.
        max_offset = (m - 1) * n * k + (n + 255) * k + ((k + 1023) // 1024) * 1024 - 1
        assert 0 <= max_offset < 2**31, (m, n, k, max_offset)
    for group in manifest()["groups"]:
        for name, config in VARIANTS.items():
            path = ir_path(group, name)
            assert expected(group, config) == path.read_text(), (group, name)
            rows.append(dict(group=group, name=name, sha256=sha256(path)))
    for group in manifest()["groups"]:
        for factor in (8, 16):
            name = f"r2_combo{factor}_v2"
            path = ir_path(group, name)
            assert combined(group, factor) == path.read_text()
            rows.append(dict(group=group, name=name, sha256=sha256(path)))
    finalist = ROOT / "results/round2_finalist.json"
    if finalist.exists():
        selected = json.loads(finalist.read_text())
        for group, identity in selected["groups"].items():
            path = ir_path(group, "r2_final")
            assert (
                path.read_bytes() == ir_path(group, selected["selected"]).read_bytes()
            )
            assert sha256(path) == identity["sha256"]
    save_json(
        args.output,
        dict(
            passed=True,
            plan_sha256=sha256(ROOT / "optimization_plan_round2.md"),
            records=rows,
        ),
    )


if __name__ == "__main__":
    main()
