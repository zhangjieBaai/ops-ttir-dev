"""Measure baseline/final local and merge kernels in all four combinations."""

import argparse
import json
import os
import sys
from pathlib import Path

OP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP.parent / "complex_suite"))
from multiround_runner import run  # noqa: E402


def link_stage_variants():
    # Both versions use the same sorted (value, index) workspace representation.
    expanded = OP / "round2/expanded"
    for item in json.loads((expanded / "manifest.json").read_text())["records"]:
        group = item["group"]
        for selected_stage in ("local", "merge"):
            for stage in ("local", "merge"):
                folder = OP / "multiround/r6/ir" / group / stage
                target = (
                    folder / "optimized.ttir"
                    if stage == selected_stage
                    else expanded / "ir" / group / stage / "baseline.ttir"
                )
                path = folder / f"final_{selected_stage}_only.ttir"
                if not path.is_symlink():
                    path.symlink_to(os.path.relpath(target, folder))
                assert path.resolve(strict=True) == target.resolve(strict=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--seed", type=int, default=234)
    args = parser.parse_args()
    args.op = "partial_topk"
    args.round = 6
    args.rounds = 5
    args.rep = 20
    args.names = "final_local_only,final_merge_only,merge_heads_packed_v2"
    args.check_only = False
    link_stage_variants()
    run(args)
