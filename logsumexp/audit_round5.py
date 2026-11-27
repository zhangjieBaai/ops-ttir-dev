"""Reproduce TTIR inputs and recompute raw paired evidence without GPU work."""

import argparse
from pathlib import Path

from audit_round3 import audit_timing
from common import ROOT, ir_path, manifest, save_json, sha256
from make_round5 import base2, fast_log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = dict(passed=True, groups={}, runs={})
    for group in manifest()["groups"]:
        parent = ir_path(group, "r4_focus").read_text()
        for name, fn in [("r5_fastlog", fast_log), ("r5_log2", base2)]:
            assert fn(parent) == ir_path(group, name).read_text()
        out["groups"][group] = {
            n: sha256(ir_path(group, n)) for n in ("r4_focus", "r5_fastlog", "r5_log2")
        }
    for name in ("main_parent", "repeat_parent", "main_baseline", "repeat_baseline"):
        out["runs"][name] = audit_timing(
            ROOT / "results/runs/r5_final" / (name + ".json"), "r5_fastlog"
        )
    assert len({x["physical_uuid"] for x in out["runs"].values()}) == 1
    save_json(args.output, out)


if __name__ == "__main__":
    main()
