"""Qualify all32 points at fixed GPU4 clocks, restoring automatic clocks in finally."""

import json
import os
import signal
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PYTHON = "/root/venvs/ir-ops-dev/bin/python"


def main():
    output = ROOT / "clock_lock_fullrun.json"
    assert not output.exists()
    record = {"physical_gpu": 4, "requested_sm_mhz": 1200, "commands": []}

    def smi(*args):
        return subprocess.run(
            ["nvidia-smi", "-i", "4", *args], capture_output=True, text=True
        )

    def persist():
        output.write_text(json.dumps(record, indent=2) + "\n")

    def stop(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    record["before"] = smi("-q", "-d", "CLOCK").stdout
    locked = False
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="4", ATTENTION_R2_LOCK_SM_MHZ="1200")

    def run(script, log, *args, sanitizer=None):
        cmd = [PYTHON, str(ROOT / script), *map(str, args)]
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
        print("START", log, flush=True)
        with (ROOT / log).open("w") as handle:
            proc = subprocess.run(
                cmd, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT
            )
        record["commands"].append(
            {"argv": cmd, "log": log, "exit_code": proc.returncode}
        )
        persist()
        proc.check_returncode()
        print("DONE", log, flush=True)

    try:
        x = smi("--lock-gpu-clocks=1200,1200")
        record["lock"] = {
            "exit_code": x.returncode,
            "stdout": x.stdout,
            "stderr": x.stderr,
        }
        x.check_returncode()
        locked = True
        record["after_lock"] = smi(
            "--query-gpu=uuid,clocks.sm,clocks.mem,power.draw", "--format=csv"
        ).stdout
        persist()
        run(
            "verify_pipeline.py",
            "pipeline_locked32.log",
            "--output",
            ROOT / "results/pipeline_locked32",
        )
        run(
            "validate.py",
            "accuracy_locked_all6.log",
            "--output",
            ROOT / "results/accuracy_locked_all6",
        )
        accuracy = json.loads(
            (ROOT / "results/accuracy_locked_all6/accuracy.json").read_text()
        )
        assert accuracy["passed"] and all(accuracy["valid"].values())
        run(
            "benchmark.py",
            "screen32_locked.log",
            "--output",
            ROOT / "results/screen32_locked",
            "--accuracy",
            ROOT / "results/accuracy_locked_all6/accuracy.json",
            "--pipeline",
            ROOT / "results/pipeline_locked32/pipeline.json",
        )
        run("select.py", "select.log")
        selection = ROOT / "selection_pre_main.json"
        run(
            "verify_pipeline.py",
            "pipeline_main32.log",
            "--output",
            ROOT / "results/pipeline_main32",
        )
        run(
            "validate.py",
            "accuracy_main32.log",
            "--output",
            ROOT / "results/accuracy_main32",
            "--selection",
            selection,
        )
        run(
            "benchmark.py",
            "main32.log",
            "--output",
            ROOT / "results/main32",
            "--selection",
            selection,
            "--accuracy",
            ROOT / "results/accuracy_main32/accuracy.json",
            "--pipeline",
            ROOT / "results/pipeline_main32/pipeline.json",
            "--rounds",
            "7",
            "--rep",
            "20",
        )
        run(
            "verify_pipeline.py",
            "pipeline_repeat32.log",
            "--output",
            ROOT / "results/pipeline_repeat32",
        )
        run(
            "validate.py",
            "accuracy_repeat32.log",
            "--output",
            ROOT / "results/accuracy_repeat32",
            "--selection",
            selection,
            "--seed",
            "123",
        )
        run(
            "benchmark.py",
            "repeat32.log",
            "--output",
            ROOT / "results/repeat32",
            "--selection",
            selection,
            "--accuracy",
            ROOT / "results/accuracy_repeat32/accuracy.json",
            "--pipeline",
            ROOT / "results/pipeline_repeat32/pipeline.json",
            "--seed",
            "123",
            "--rounds",
            "5",
            "--rep",
            "20",
        )
        for name in ("main32", "repeat32"):
            timed = json.loads((ROOT / f"results/{name}/benchmark.json").read_text())
            assert timed["passed"] and timed["aa_passed"]
        for tool in ("memcheck", "racecheck"):
            run(
                "validate.py",
                tool + ".log",
                "--output",
                ROOT / "results" / tool,
                "--selection",
                selection,
                sanitizer=tool,
            )
        record["passed"] = True
    finally:
        if locked:
            x = smi("--reset-gpu-clocks")
            record["restore"] = {
                "exit_code": x.returncode,
                "stdout": x.stdout,
                "stderr": x.stderr,
            }
            record["restored"] = x.returncode == 0
        record["after_restore"] = smi("-q", "-d", "CLOCK").stdout
        persist()
        print("RESTORE", record.get("restore"), flush=True)
    assert record["passed"] and record["restored"]


if __name__ == "__main__":
    main()
