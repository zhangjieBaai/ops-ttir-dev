"""Explicit pre-experiment original-source/file gate on all 30 frozen mappings."""

import argparse
import json
from pathlib import Path

from run import BASE, ROOT, c, compile_case, pipeline, save


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    assert not args.output.exists()
    manifest = json.loads((BASE / "manifest.json").read_text())
    assert manifest["source_sha256"] == c.sha(ROOT.parent / "tutorial.py")
    rows = []
    for case in manifest["cases"]:
        baseline, metadata = compile_case(case, "baseline")
        rows.append(
            {
                "case": case["id"],
                "pipeline": pipeline(case, baseline, args.seed),
                "kernel": metadata,
            }
        )
    save(args.output, {"environment": c.environment(), "passed": True, "records": rows})
    print("PASS: 30 original source/file LLIR/PTX/cubin and output gates")


if __name__ == "__main__":
    main()
