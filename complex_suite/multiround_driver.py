"""Sequential r3-r5 controller: no same-GPU overlap, no silent parent promotion."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from support import digest, freeze, save

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/root/venvs/ir-ops-dev/bin/python"


def completed(path):
    if not path.exists():
        return False
    assert json.loads(path.read_text()).get("passed"), f"Incomplete prior run: {path}"
    return True


def finalize(op, rid, candidate):
    rr = ROOT / op / "multiround" / f"r{rid}"
    data = {
        phase: json.loads((rr / f"results/{phase}/results.json").read_text())
        for phase in ("screen", "main", "repeat", "memcheck", "racecheck")
    }
    assert all(d["passed"] and candidate in d["validated_names"] for d in data.values())
    assert data["main"]["aa_passed"] and data["repeat"]["aa_passed"]
    for tool, marker in (
        ("memcheck", "ERROR SUMMARY: 0 errors"),
        ("racecheck", "0 errors, 0 warnings"),
    ):
        assert marker in (rr / f"{tool}.log").read_text()
    if op == "norm_quant":
        assert json.loads((rr / "results/extreme_stress.json").read_text())["passed"]
    accepted = all(
        d["summary"][candidate]["parent_over"]["geomean"] >= 1.02
        and d["summary"][candidate]["parent_over"]["worst"] >= 0.95
        for d in (data["main"], data["repeat"])
    )
    selected = candidate if accepted else "parent"
    records = []
    stages = ["kernel"] if op == "norm_quant" else ["local", "merge"]
    for main, repeat in zip(data["main"]["records"], data["repeat"]["records"]):
        assert main["shape"] == repeat["shape"]
        for stage in stages:
            meta = main["kernels"][selected][stage]
            path = Path(meta["ir_path"])
            assert (
                digest(path)
                == meta["ir_sha256"]
                == repeat["kernels"][selected][stage]["ir_sha256"]
            )
            chosen = rr / "ir" / main["group"]
            if op != "norm_quant":
                chosen /= stage
            freeze(chosen / "optimized.ttir", path.read_text())
            records.append(
                {
                    "group": main["group"],
                    "stage": stage,
                    "selected_sha256": meta["ir_sha256"],
                }
            )
    result = {
        "round": rid,
        "candidate_tested": candidate,
        "accepted": accepted,
        "selected": selected,
        "main": data["main"]["summary"][candidate],
        "repeat": data["repeat"]["summary"][candidate],
        "selected_main": data["main"]["summary"][selected],
        "selected_repeat": data["repeat"]["summary"][selected],
        "shape_count": 30,
        "mapping": records,
        "sanitizer": {"memcheck": "0 errors", "racecheck": "0 errors, 0 warnings"},
        "evidence": {
            p: digest(rr / f"results/{p}/results.json")
            for p in ("screen", "main", "repeat", "memcheck", "racecheck")
        },
    }
    freeze(rr / "selection.json", json.dumps(result, indent=2) + "\n")
    return result


def main(args):
    folder = ROOT / args.op / "multiround"
    journal = folder / "workflow.json"
    state = (
        json.loads(journal.read_text())
        if journal.exists()
        else {"op": args.op, "commands": [], "rounds": {}}
    )
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == (
        "6" if args.op == "norm_quant" else "7"
    )

    def run(argv, log, marker=None):
        if marker and completed(marker):
            return
        assert not log.exists(), f"Use a fresh log after failure: {log}"
        print("START", args.op, str(log.relative_to(folder)), flush=True)
        with log.open("w") as stream:
            proc = subprocess.run(
                argv,
                cwd=ROOT,
                env=os.environ.copy(),
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        state["commands"].append(
            {
                "argv": argv,
                "log": str(log.relative_to(folder)),
                "exit_code": proc.returncode,
            }
        )
        save(journal, state)
        proc.check_returncode()
        print("DONE", args.op, str(log.relative_to(folder)), flush=True)

    for rid in (3, 4, 5):
        rr = folder / f"r{rid}"
        rr.mkdir(parents=True, exist_ok=True)
        if (rr / "selection.json").exists():
            state["rounds"][str(rid)] = json.loads((rr / "selection.json").read_text())
            continue
        if not (rr / "candidates.json").exists():
            run(
                [PYTHON, str(folder / "generate.py"), "--round", str(rid)],
                rr / "generate.log",
            )
        names = sorted(
            {x["name"] for x in json.loads((rr / "candidates.json").read_text())}
        )
        base = [
            PYTHON,
            str(ROOT / "complex_suite/multiround_runner.py"),
            "--op",
            args.op,
            "--round",
            str(rid),
        ]
        run(
            [*base, "--run", "screen", "--names", ",".join(names)],
            rr / "screen.log",
            rr / "results/screen/results.json",
        )
        screen = json.loads((rr / "results/screen/results.json").read_text())
        assert screen[
            "aa_passed"
        ], "Invalid screening must be separately rerun and documented"
        valid = [n for n in names if n in screen["validated_names"]]
        assert valid, "All candidates rejected; disposition required"
        candidate = max(
            valid, key=lambda n: screen["summary"][n]["parent_over"]["geomean"]
        )
        freeze(
            rr / "selection_pre_main.json",
            json.dumps(
                {
                    "candidate": candidate,
                    "screen_sha256": digest(rr / "results/screen/results.json"),
                    "rule": "highest full30 incremental geomean; only final gates decide promotion",
                },
                indent=2,
            )
            + "\n",
        )
        for phase, rounds, seed in (("main", 7, 42), ("repeat", 5, 123)):
            run(
                [
                    *base,
                    "--run",
                    phase,
                    "--names",
                    candidate,
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
            d = json.loads((rr / f"results/{phase}/results.json").read_text())
            assert (
                d["aa_passed"] and candidate in d["validated_names"]
            ), "Final qualification failed; preserve and inspect"
        if args.op == "norm_quant":
            run(
                [
                    PYTHON,
                    str(folder / "stress.py"),
                    "--round",
                    str(rid),
                    "--name",
                    candidate,
                ],
                rr / "stress.log",
                rr / "results/extreme_stress.json",
            )
        for tool in ("memcheck", "racecheck"):
            kernel = (
                "residual_rms_int8" if args.op == "norm_quant" else "partial_select"
            )
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
                    "--names",
                    candidate,
                    "--check-only",
                ],
                rr / f"{tool}.log",
                rr / f"results/{tool}/results.json",
            )
        result = finalize(args.op, rid, candidate)
        state["rounds"][str(rid)] = result
        save(journal, state)
        print(
            "SELECT",
            args.op,
            rid,
            result["selected"],
            "accepted",
            result["accepted"],
            flush=True,
        )
    state["passed"] = True
    save(journal, state)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--op", choices=["norm_quant", "partial_topk"], required=True)
    main(p.parse_args())
