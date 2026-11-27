"""Execute sequential rounds on GPU0; never overlap benchmark and sanitizer."""

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PYTHON = "/root/venvs/ir-ops-dev/bin/python"


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    commands = []

    def run(stage, phase, argv, sanitizer=None):
        cmd = [PYTHON, *map(str, argv)]
        if sanitizer:
            cmd = [
                "compute-sanitizer",
                "--tool",
                sanitizer,
                "--error-exitcode",
                "99",
                "--kernel-name",
                "regex=_weight_int4pack_mm_kernel",
                *cmd,
            ]
        log = ROOT / stage / f"{phase}.log"
        assert not log.exists()
        print("START", stage, phase, flush=True)
        with log.open("w") as f:
            p = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
        commands.append(
            {"stage": stage, "phase": phase, "argv": cmd, "exit_code": p.returncode}
        )
        (ROOT / "workflow_commands.json").write_text(
            json.dumps(commands, indent=2) + "\n"
        )
        p.check_returncode()
        print("DONE", stage, phase, flush=True)

    # r3 main was launched before this sequencer; refuse to overlap or reuse incomplete evidence.
    existing = json.loads((ROOT / "r3/results/main/results.json").read_text())
    assert (
        existing["passed"] and existing["aa_passed"] and len(existing["records"]) == 30
    )
    for stage in ("r3", "r4", "r5"):
        folder = ROOT / stage
        if stage != "r3":
            folder.mkdir(parents=True, exist_ok=True)
            run(stage, "generate", [ROOT / "generate.py", stage])
            candidates = json.loads((folder / "candidates.json").read_text())["names"]
            run(
                stage,
                "screen",
                [
                    ROOT / "run.py",
                    "--stage",
                    stage,
                    "--run",
                    "screen",
                    "--names",
                    ",".join(["baseline", "anchor", "parent", *candidates]),
                ],
            )
            screen = json.loads((folder / "results/screen/results.json").read_text())
            assert screen["passed"] and screen["aa_passed"]
            run(stage, "select", [ROOT / "select.py", stage])
        chosen = json.loads((folder / "selection_pre_main.json").read_text())[
            "candidate"
        ]
        base_args = [
            ROOT / "run.py",
            "--stage",
            stage,
            "--names",
            f"baseline,anchor,parent,{chosen}",
        ]
        if stage != "r3":
            run(
                stage,
                "main",
                [*base_args, "--run", "main", "--rounds", "7", "--rep", "20"],
            )
        run(
            stage,
            "repeat",
            [
                *base_args,
                "--run",
                "repeat",
                "--rounds",
                "5",
                "--rep",
                "20",
                "--seed",
                "123",
            ],
        )
        for phase in ("main", "repeat"):
            data = json.loads((folder / f"results/{phase}/results.json").read_text())
            assert data["passed"] and data["aa_passed"] and len(data["records"]) == 30
        for tool in ("memcheck", "racecheck"):
            run(
                stage, tool, [*base_args, "--run", tool, "--check-only"], sanitizer=tool
            )
        run(stage, "finalize", [ROOT / "finalize.py", stage])
    run("r5", "combined", [ROOT / "finalize.py", "combined"])


if __name__ == "__main__":
    main()
