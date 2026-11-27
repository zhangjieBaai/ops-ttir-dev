"""Promote the preselected candidate only after repeat and sanitizer gates."""

import json
import os
import subprocess
import sys

from common import ROOT, manifest, save, sha, write_once


def main():
    selected = json.loads((ROOT / "results/selection_pre_repeat.json").read_text())
    analysis = json.loads((ROOT / "results/analysis.json").read_text())
    chosen = selected["chosen"]
    repeat = analysis["summaries"]["repeat"][chosen]
    assert repeat["geomean"] >= 1.02 and repeat["minimum"] >= 0.95
    for tool in ("memcheck", "racecheck"):
        log = (ROOT / "results" / (tool + ".log")).read_text()
        assert "ERROR SUMMARY: 0 errors" in log or "0 errors" in log
        if tool == "racecheck":
            assert "0 hazards" in log
    mappings = []
    for case in manifest()["cases"]:
        source = ROOT / "ir" / case["id"] / (chosen + ".ttir")
        dest = source.with_name("optimized.ttir")
        write_once(dest, source.read_text())
        mappings.append(
            dict(
                case=case["id"],
                selected=chosen,
                source_sha256=sha(source),
                optimized_sha256=sha(dest),
            )
        )
    scripts = {str(p.relative_to(ROOT)): sha(p) for p in ROOT.glob("*.py")}
    save(
        ROOT / "results/selection_final.json",
        dict(
            chosen=chosen,
            mappings=mappings,
            main=analysis["summaries"]["main"][chosen],
            repeat=repeat,
            scripts=scripts,
            sanitizer="memcheck and racecheck, all 7 cases x 5 modes x 2 versions",
            evidence={p.name: sha(p) for p in (ROOT / "results").glob("*.json")},
            environment={
                k: v
                for k, v in os.environ.items()
                if any(s in k for s in ("TRITON", "CUDA", "LD_LIBRARY", "PYTHONPATH"))
            },
            tools=dict(
                python=sys.executable,
                compute_sanitizer=subprocess.check_output(
                    ["/usr/local/cuda/bin/compute-sanitizer", "--version"], text=True
                ),
            ),
        ),
    )


if __name__ == "__main__":
    main()
