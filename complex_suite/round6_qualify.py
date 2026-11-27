"""Freeze a round6 finalist and run full independent qualification."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from support import digest, freeze, save

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/root/venvs/ir-ops-dev/bin/python"


def read(path):
    return json.loads(path.read_text())


def main(args):
    rr = ROOT / args.op / "multiround/r6"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == (
        "6" if args.op == "norm_quant" else "7"
    )
    screen_path = rr / "results" / args.screen / "results.json"
    screen = read(screen_path)
    assert screen["passed"] and screen["aa_passed"]
    candidates = [
        n
        for n in screen["validated_names"]
        if n not in ("baseline", "anchor", "parent")
    ]
    candidate = max(
        candidates, key=lambda n: screen["summary"][n]["parent_over"]["geomean"]
    )
    freeze(
        rr / "selection_pre_main.json",
        json.dumps(
            {
                "candidate": candidate,
                "screen": args.screen,
                "screen_sha256": digest(screen_path),
            },
            indent=2,
        )
        + "\n",
    )
    journal = rr / "workflow.json"
    state = (
        read(journal)
        if journal.exists()
        else {"op": args.op, "round": 6, "commands": []}
    )
    base = [
        PYTHON,
        str(ROOT / "complex_suite/round6_runner.py"),
        "--op",
        args.op,
        "--names",
        candidate,
    ]

    def run(argv, log, marker):
        if marker.exists():
            assert read(marker)["passed"]
            return
        assert not log.exists(), f"Preserve failed log and use a versioned rerun: {log}"
        print("START", args.op, log.name, flush=True)
        with log.open("w") as stream:
            result = subprocess.run(
                argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT
            )
        state["commands"].append(
            {"argv": argv, "log": str(log), "exit_code": result.returncode}
        )
        save(journal, state)
        result.check_returncode()
        assert read(marker)["passed"]
        print("DONE", args.op, log.name, flush=True)

    for phase, rounds, seed in (("main", 7, 42), ("repeat", 5, 123)):
        run(
            [
                *base,
                "--run",
                phase,
                "--rounds",
                str(rounds),
                "--rep",
                "20",
                "--seed",
                str(seed),
            ],
            rr / f"{phase}.log",
            rr / f"results/{phase}/results.json",
        )
        assert read(rr / f"results/{phase}/results.json")["aa_passed"]
    if args.op == "norm_quant":
        run(
            [PYTHON, str(rr.parent / "stress.py"), "--round", "6", "--name", candidate],
            rr / "stress.log",
            rr / "results/extreme_stress.json",
        )
    for tool in ("memcheck", "racecheck"):
        kernel = "residual_rms_int8" if args.op == "norm_quant" else "partial_select"
        run(
            [
                "compute-sanitizer",
                "--tool",
                tool,
                "--error-exitcode",
                "99",
                "--kernel-name",
                f"regex={kernel}",
                *base,
                "--run",
                tool,
                "--check-only",
            ],
            rr / f"{tool}.log",
            rr / f"results/{tool}/results.json",
        )
    data = {
        phase: read(rr / f"results/{phase}/results.json")
        for phase in ("main", "repeat", "memcheck", "racecheck")
    }
    assert all(d["passed"] and candidate in d["validated_names"] for d in data.values())
    assert "ERROR SUMMARY: 0 errors" in (rr / "memcheck.log").read_text()
    assert "0 errors, 0 warnings" in (rr / "racecheck.log").read_text()
    accepted = all(
        data[phase]["summary"][candidate]["parent_over"]["geomean"] >= 1.02
        and data[phase]["summary"][candidate]["parent_over"]["worst"] >= 0.95
        for phase in ("main", "repeat")
    )
    selected = candidate if accepted else "parent"
    mapping = []
    for record in data["main"]["records"]:
        peers = [
            next(x for x in d["records"] if x["group"] == record["group"])
            for d in data.values()
        ]
        for stage, meta in record["kernels"][selected].items():
            path = Path(meta["ir_path"])
            assert digest(path) == meta["ir_sha256"]
            assert all(
                p["kernels"][selected][stage]["ir_sha256"] == meta["ir_sha256"]
                for p in peers
            )
            target = rr / "ir" / record["group"]
            if args.op == "partial_topk":
                target /= stage
            freeze(target / "optimized.ttir", path.read_text())
            mapping.append(
                {
                    "group": record["group"],
                    "stage": stage,
                    "selected_sha256": digest(path),
                }
            )
    result = {
        "round": 6,
        "shape_count": 30,
        "candidate_tested": candidate,
        "accepted": accepted,
        "selected": selected,
        "screen_run": args.screen,
        "main": data["main"]["summary"][candidate],
        "repeat": data["repeat"]["summary"][candidate],
        "selected_main": data["main"]["summary"][selected],
        "selected_repeat": data["repeat"]["summary"][selected],
        "mapping": mapping,
        "sanitizer": {"memcheck": "0 errors", "racecheck": "0 errors, 0 warnings"},
        "evidence": {
            phase: digest(rr / f"results/{phase}/results.json") for phase in data
        },
    }
    freeze(rr / "selection.json", json.dumps(result, indent=2) + "\n")
    state.update({"passed": True, "selection": result})
    save(journal, state)
    print("SELECT", args.op, selected, "accepted", accepted, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", required=True, choices=("norm_quant", "partial_topk"))
    parser.add_argument("--screen", default="screen")
    main(parser.parse_args())
