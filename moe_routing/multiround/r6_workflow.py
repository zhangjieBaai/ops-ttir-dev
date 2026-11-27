"""Qualify the already-screened r6 finalist without overlapping GPU work."""

import json
import os
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE / "r6"
PYTHON = "/root/venvs/ir-ops-dev/bin/python"


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "2"
    screen = json.loads((ROOT / "results/screen/performance.json").read_text())
    assert screen["aa_passed"] and len(screen["records"]) == 30
    path = ROOT / "workflow.json"
    assert not path.exists()
    state = {"round": "r6", "commands": [], "completed": False}

    def run(script, log, args, sanitizer=None):
        assert not log.exists()
        cmd = [PYTHON, str(BASE / script), *map(str, args)]
        if sanitizer:
            cmd = [
                "compute-sanitizer",
                "--tool",
                sanitizer,
                "--error-exitcode",
                "99",
                "--kernel-name",
                "regex=topk",
                *cmd,
            ]
        print("START", log.name, flush=True)
        with log.open("w") as output:
            result = subprocess.run(
                cmd, cwd=BASE.parents[1], stdout=output, stderr=subprocess.STDOUT
            )
        state["commands"].append(
            {"argv": cmd, "log": str(log), "exit_code": result.returncode}
        )
        path.write_text(json.dumps(state, indent=2) + "\n")
        result.check_returncode()
        print("DONE", log.name, flush=True)

    run("r6_finalize.py", ROOT / "select.log", ["select", "--round", "r6"])
    candidate = json.loads((ROOT / "selection_pre_main.json").read_text())["candidate"]
    for name, seed, count in (("main", 42, 7), ("repeat", 123, 5)):
        run(
            "r6_run.py",
            ROOT / f"{name}.log",
            [
                "--round",
                "r6",
                "--run",
                name,
                "--variants",
                candidate,
                "--rounds",
                count,
                "--rep",
                20,
                "--seed",
                seed,
            ],
        )
        data = json.loads((ROOT / f"results/{name}/performance.json").read_text())
        assert data["aa_passed"] and len(data["records"]) == 30
    datasets = [
        json.loads((ROOT / f"results/{name}/performance.json").read_text())
        for name in ("main", "repeat")
    ]
    accepts = all(
        d["summary"][candidate]["parent_over_candidate"]["geomean"] >= 1.02
        and d["summary"][candidate]["parent_over_candidate"]["worst"] >= 0.95
        for d in datasets
    )
    if accepts:
        for tool in ("memcheck", "racecheck"):
            run(
                "r6_run.py",
                ROOT / f"{tool}.log",
                [
                    "--round",
                    "r6",
                    "--run",
                    tool,
                    "--variants",
                    candidate,
                    "--check-only",
                ],
                sanitizer=tool,
            )
    run("r6_finalize.py", ROOT / "finalize.log", ["finish", "--round", "r6"])
    state["completed"] = True
    path.write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
