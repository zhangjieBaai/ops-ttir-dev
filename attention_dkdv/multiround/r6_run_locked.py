"""One bounded r6 search with guaranteed child/clock cleanup."""

import json
import os
import signal
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1]
PYTHON = "/root/venvs/ir-ops-dev/bin/python"


def main():
    path = BASE / "r6/clock_run.json"
    assert not path.exists()
    record = {"gpu": 4, "requested_sm_mhz": 1200, "commands": []}
    active = None
    locked = False

    def save():
        path.write_text(json.dumps(record, indent=2) + "\n")

    def smi(*args):
        return subprocess.run(
            ["nvidia-smi", "-i", "4", *args], capture_output=True, text=True
        )

    def stop(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    record["before"] = smi("-q", "-d", "CLOCK").stdout

    def run(round_name, script, log, *args, sanitizer=None):
        nonlocal active
        folder = BASE / round_name
        folder.mkdir(exist_ok=True)
        cmd = [PYTHON, str(BASE / script), *map(str, args)]
        if sanitizer:
            cmd = [
                "compute-sanitizer",
                "--tool",
                sanitizer,
                "--error-exitcode",
                "99",
                "--kernel-name",
                "regex=_attn_bwd_dkdv_only",
                *cmd,
            ]
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="4",
            ATTENTION_MR_ROUND=round_name,
            ATTENTION_MR_LOCK_SM_MHZ="1200",
        )
        print("START", round_name, log, flush=True)
        with (folder / log).open("w") as handle:
            active = subprocess.Popen(
                cmd,
                cwd=REPO,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            record["active_pid"] = active.pid
            save()
            code = active.wait()
        record["commands"].append(
            {"round": round_name, "argv": cmd, "log": log, "exit_code": code}
        )
        record["active_pid"] = None
        active = None
        save()
        assert code == 0, (round_name, script, log, code)
        print("DONE", round_name, log, flush=True)

    try:
        p = smi("--lock-gpu-clocks=1200,1200")
        record["lock"] = {
            "exit_code": p.returncode,
            "stdout": p.stdout,
            "stderr": p.stderr,
        }
        p.check_returncode()
        locked = True
        record["after_lock"] = smi(
            "--query-gpu=uuid,clocks.sm,clocks.mem,power.draw", "--format=csv"
        ).stdout
        save()
        for rd in ("r6",):
            root = BASE / rd
            run(rd, "r6_generate.py", "generation.log")
            run(
                rd,
                "verify_pipeline.py",
                "pipeline.log",
                "--output",
                root / "results/pipeline",
            )
            run(
                rd,
                "validate.py",
                "accuracy_screen.log",
                "--output",
                root / "results/accuracy_screen",
            )
            run(
                rd,
                "benchmark.py",
                "screen.log",
                "--output",
                root / "results/screen",
                "--accuracy",
                root / "results/accuracy_screen/accuracy.json",
                "--pipeline",
                root / "results/pipeline/pipeline.json",
            )
            run(rd, "select.py", "select.log")
            selected = root / "selection_pre_main.json"
            for label, seed, rounds in (("main", 20, 7), ("repeat", 123, 5)):
                run(
                    rd,
                    "verify_pipeline.py",
                    f"pipeline_{label}.log",
                    "--output",
                    root / f"results/pipeline_{label}",
                )
                run(
                    rd,
                    "validate.py",
                    f"accuracy_{label}.log",
                    "--output",
                    root / f"results/accuracy_{label}",
                    "--selection",
                    selected,
                    "--seed",
                    seed,
                )
                run(
                    rd,
                    "benchmark.py",
                    f"{label}.log",
                    "--output",
                    root / f"results/{label}",
                    "--accuracy",
                    root / f"results/accuracy_{label}/accuracy.json",
                    "--pipeline",
                    root / f"results/pipeline_{label}/pipeline.json",
                    "--selection",
                    selected,
                    "--seed",
                    seed,
                    "--rounds",
                    rounds,
                    "--rep",
                    20,
                )
                timed = json.loads(
                    (root / f"results/{label}/benchmark.json").read_text()
                )
                assert timed["passed"] and timed["aa_passed"]
            for tool in ("memcheck", "racecheck"):
                run(
                    rd,
                    "validate.py",
                    tool + ".log",
                    "--output",
                    root / f"results/{tool}",
                    "--selection",
                    selected,
                    sanitizer=tool,
                )
            run(rd, "r6_finalize.py", "finalize.log")
        record["passed"] = True
    finally:
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM)
            try:
                active.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid, signal.SIGKILL)
                active.wait()
        if locked:
            p = smi("--reset-gpu-clocks")
            record["restore"] = {
                "exit_code": p.returncode,
                "stdout": p.stdout,
                "stderr": p.stderr,
            }
            record["restored"] = p.returncode == 0
        record["after_restore"] = smi("-q", "-d", "CLOCK").stdout
        save()
        print("RESTORE", record.get("restore"), flush=True)
    assert record["passed"] and record["restored"]


if __name__ == "__main__":
    main()
