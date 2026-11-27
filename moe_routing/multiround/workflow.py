"""Sequential subprocess orchestration; no same-device benchmark overlap."""

import json
import os
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1]
PY = "/root/venvs/ir-ops-dev/bin/python"


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "2"
    status = {"rounds": [], "commands": []}
    status_path = BASE / "workflow_status.json"
    assert not status_path.exists()

    def save_status():
        status_path.write_text(json.dumps(status, indent=2) + "\n")

    def run(script, log, *args, sanitizer=None):
        cmd = [PY, str(BASE / script), *map(str, args)]
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
        print("START", log, flush=True)
        with log.open("w") as f:
            result = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
        status["commands"].append(
            {"argv": cmd, "log": str(log), "exit_code": result.returncode}
        )
        save_status()
        result.check_returncode()
        print("DONE", log, flush=True)

    for rnd in ("r3", "r4", "r5"):
        root = BASE / rnd
        root.mkdir(exist_ok=True)
        screen = "screen_v2" if rnd == "r3" else "screen"
        if rnd != "r3":
            run("generate.py", root / "generate.log", rnd)
            run("run.py", root / "screen.log", "--round", rnd, "--run", screen)
        screen_data = json.loads(
            (root / f"results/{screen}/performance.json").read_text()
        )
        assert screen_data["aa_passed"] and len(screen_data["records"]) == 30
        run(
            "finalize.py",
            root / "select.log",
            "select",
            "--round",
            rnd,
            "--screen",
            screen,
        )
        candidate = json.loads((root / "selection_pre_main.json").read_text())[
            "candidate"
        ]
        for label, seed, rounds in [("main", 42, 7), ("repeat", 123, 5)]:
            run(
                "run.py",
                root / f"{label}.log",
                "--round",
                rnd,
                "--run",
                label,
                "--variants",
                candidate,
                "--rounds",
                rounds,
                "--rep",
                20,
                "--seed",
                seed,
            )
            data = json.loads((root / f"results/{label}/performance.json").read_text())
            assert data["aa_passed"] and len(data["records"]) == 30
        datasets = [
            json.loads((root / f"results/{label}/performance.json").read_text())
            for label in ("main", "repeat")
        ]
        accepts = all(
            d["summary"][candidate]["parent_over_candidate"]["geomean"] >= 1.02
            and d["summary"][candidate]["parent_over_candidate"]["worst"] >= 0.95
            for d in datasets
        )
        if accepts:
            for tool in ("memcheck", "racecheck"):
                run(
                    "run.py",
                    root / f"{tool}.log",
                    "--round",
                    rnd,
                    "--run",
                    tool,
                    "--variants",
                    candidate,
                    "--check-only",
                    sanitizer=tool,
                )
        run("finalize.py", root / "finalize.log", "finish", "--round", rnd)
        selection = json.loads((root / "selection.json").read_text())
        status["rounds"].append(
            {
                "round": rnd,
                "candidate": candidate,
                "accepted": selection["accepted"],
                "repeat": selection["repeat"],
                "selected_repeat": selection["selected_repeat"],
            }
        )
        save_status()
    status["completed"] = True
    save_status()


if __name__ == "__main__":
    main()
