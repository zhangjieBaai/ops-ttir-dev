"""Run a bounded new INT4 round without modifying prior round artifacts."""

import importlib.util
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "r6"
REPO = ROOT.parents[1]
PYTHON = "/root/venvs/ir-ops-dev/bin/python"


def load_script(filename):
    spec = importlib.util.spec_from_file_location("r6_" + filename, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(path.read_text())


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    validation = read(OUT / "results/validate/results.json")
    assert validation["passed"] and len(validation["records"]) == 30
    identities = {
        n: sum(
            r["kernels"][n]["elf_text"] == r["kernels"]["parent"]["elf_text"]
            for r in validation["records"]
        )
        for n in validation["valid_names"]
        if n not in ("baseline", "anchor", "parent")
    }
    (OUT / "identity_check.json").write_text(json.dumps(identities, indent=2) + "\n")
    candidates = [n for n, count in identities.items() if count != 30]
    commands = []

    def run(phase, names, rounds=3, rep=10, seed=42, tool=None):
        cmd = [
            PYTHON,
            str(ROOT / "r6_run.py"),
            "--stage",
            "r6",
            "--run",
            phase,
            "--names",
            ",".join(names),
        ]
        cmd += ["--rounds", str(rounds), "--rep", str(rep), "--seed", str(seed)]
        if tool:
            cmd += ["--check-only"]
            cmd = [
                "compute-sanitizer",
                "--tool",
                tool,
                "--error-exitcode",
                "99",
                "--kernel-name",
                "regex=_weight_int4pack_mm_kernel",
                *cmd,
            ]
        log = OUT / (phase + ".log")
        assert not log.exists()
        print("START", phase, flush=True)
        with log.open("w") as f:
            result = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
        commands.append({"phase": phase, "argv": cmd, "exit_code": result.returncode})
        (OUT / "workflow_commands.json").write_text(
            json.dumps(commands, indent=2) + "\n"
        )
        result.check_returncode()
        data = read(OUT / f"results/{phase}/results.json")
        assert data["passed"] and len(data["records"]) == 30
        if not tool:
            assert data["aa_passed"], phase
        print("DONE", phase, flush=True)

    assert (
        candidates
    ), "No new executable candidate; document identity disposition without timing."
    run("screen", ["baseline", "anchor", "parent", *candidates])
    load_script("select.py").main("r6")
    chosen = read(OUT / "selection_pre_main.json")["candidate"]
    names = ["baseline", "anchor", "parent", chosen]
    run("main", names, rounds=7, rep=20)
    run("repeat", names, rounds=5, rep=20, seed=123)
    for tool in ("memcheck", "racecheck"):
        run(tool, names, tool=tool)
    load_script("finalize.py").main("r6")
    result = subprocess.run([PYTHON, str(ROOT / "r6_report.py")], cwd=REPO)
    result.check_returncode()


if __name__ == "__main__":
    main()
