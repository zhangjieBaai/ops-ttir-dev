"""Verify every baseline source/file mapping before an ablation session."""

import argparse
import json
from pathlib import Path

from run import protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    assert not args.output.exists()
    manifest = json.loads((protocol.BASE / "manifest.json").read_text())
    assert manifest["source_sha256"] == protocol.c.sha(
        protocol.ROOT.parent / "tutorial.py"
    )
    records = []
    for case in manifest["cases"]:
        baseline, metadata = protocol.compile_case(case, "baseline")
        records.append(
            {
                "case": case["id"],
                "pipeline": protocol.pipeline(case, baseline, args.seed),
                "kernel": metadata,
            }
        )
    protocol.save(
        args.output,
        {"environment": protocol.c.environment(), "passed": True, "records": records},
    )
    print("PASS: 30 original source/file LLIR/PTX/cubin and output gates")


if __name__ == "__main__":
    main()
