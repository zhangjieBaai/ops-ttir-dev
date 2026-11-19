"""Randomized paired CUDA graph measurements with immutable accuracy receipts."""

import argparse
import fcntl
import json
import math
import os
import random
import statistics
import subprocess
import threading
import time

import numpy as np
import torch
import triton.testing
from common import (
    CONTRACT,
    ROOT,
    SHAPES,
    allocate,
    compile_ir,
    digest,
    identity,
    launch,
    provenance,
    save,
    telemetry,
)


def summary(records):
    result = {}
    rng = np.random.default_rng(42)
    for name in sorted({r["name"] for r in records}):
        rows = [r for r in records if r["name"] == name]
        ratios = [r["speedup"] for r in rows]
        draws = np.array(
            [
                [
                    np.median(
                        rng.choice(
                            r["paired_ratios"], len(r["paired_ratios"]), replace=True
                        )
                    )
                    for r in rows
                ]
                for _ in range(2000)
            ]
        )
        ci = np.quantile(np.exp(np.log(draws).mean(axis=1)), [0.025, 0.975]).tolist()
        result[name] = dict(
            geomean=math.exp(statistics.mean(map(math.log, ratios))),
            arithmetic=statistics.mean(ratios),
            worst=min(ratios),
            ci95_conditional_bootstrap=ci,
            points=len(rows),
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--accuracy", required=True)
    p.add_argument("--pipeline", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--suite", choices=["tuning", "validation", "full", "controls"], default="full"
    )
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--rep", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    device = os.environ["CUDA_VISIBLE_DEVICES"]
    assert device.isdecimal(), "Select one physical GPU index"
    gpu_uuid = subprocess.check_output(
        ["nvidia-smi", "-i", device, "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    lock = open(f"/root/tmp/ttir-device-{gpu_uuid}.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    running = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
    )
    assert gpu_uuid not in running, f"Device busy before initialization: {running}"
    pipeline = json.loads(open(args.pipeline).read())
    assert pipeline["passed"] and pipeline["provenance"]["contract_sha256"] == digest(
        ROOT / "contract.json"
    )
    accuracy = json.loads(open(args.accuracy).read())
    assert accuracy["passed"] and accuracy["provenance"]["contract_sha256"] == digest(
        ROOT / "contract.json"
    )
    for name in set(["baseline", *args.names]) - {"aa", "double"}:
        assert accuracy["identities"][name]["ttir_sha256"] == digest(
            ROOT / f"ir/{name}.ttir"
        )
    names = args.names
    kernels = {
        name: compile_ir("baseline" if name in ["aa", "double"] else name)
        for name in set(["baseline", *names])
    }
    resources = {
        n: identity("baseline" if n in ["aa", "double"] else n, k)
        for n, k in kernels.items()
    }
    for name in set(["baseline", *args.names]) - {"aa", "double"}:
        assert (
            resources[name]["products"]["cubin"]
            == accuracy["identities"][name]["products"]["cubin"]
        )
    assert (
        resources["baseline"]["products"]["cubin"]
        == pipeline["baseline"]["products"]["cubin"]
    )
    indices = range(len(SHAPES))
    tuning = CONTRACT["performance"]["tuning_indices"]
    if args.suite == "tuning":
        indices = tuning
    if args.suite == "validation":
        indices = [i for i in indices if i not in tuning]
    if args.suite == "controls":
        indices = [0, 17, 29]
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    before = provenance()
    records = []
    load_samples = []
    stop_monitor = threading.Event()

    def monitor():
        while not stop_monitor.is_set():
            try:
                state = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "-i",
                        device,
                        "--query-gpu=timestamp,uuid,clocks.sm,clocks.mem,power.draw,"
                        "temperature.gpu,utilization.gpu,clocks_event_reasons.active",
                        "--format=csv,noheader",
                    ],
                    text=True,
                )
                processes = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=gpu_uuid,pid",
                        "--format=csv,noheader",
                    ],
                    text=True,
                )
                load_samples.append(dict(state=state, processes=processes))
            except Exception as exc:
                load_samples.append(dict(error=str(exc)))
            stop_monitor.wait(0.5)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    for i in indices:
        m, n = SHAPES[i]
        ts = allocate(m, n)
        calls = {name: (lambda ck=ck: launch(ck, ts)) for name, ck in kernels.items()}
        if "double" in names:

            def twice():
                launch(kernels["baseline"], ts)
                launch(kernels["baseline"], ts)

            calls["double"] = twice
        for call in calls.values():
            for _ in range(10):
                call()
            triton.testing.do_bench_cudagraph(call, rep=50)
        torch.cuda.synchronize()
        for name in names:
            pairs = []
            for j in range(args.rounds):
                order = ["baseline", name]
                random.shuffle(order)
                samples = {}
                for label in order:
                    samples[label] = (
                        triton.testing.do_bench_cudagraph(calls[label], rep=args.rep)
                        * 1000
                    )
                pairs.append(
                    dict(
                        round=j,
                        order=order,
                        baseline_us=samples["baseline"],
                        candidate_us=samples[name],
                        ratio=samples["baseline"] / samples[name],
                    )
                )
            ratios = [q["ratio"] for q in pairs]
            records.append(
                dict(
                    name=name,
                    shape=[m, n],
                    group="row_f32_b1024_w4",
                    pairs=pairs,
                    paired_ratios=ratios,
                    speedup=statistics.median(ratios),
                    baseline_us=statistics.median(q["baseline_us"] for q in pairs),
                    candidate_us=statistics.median(q["candidate_us"] for q in pairs),
                )
            )
            print(name, [m, n], round(statistics.median(ratios), 4), flush=True)
    torch.cuda.synchronize()
    stop_monitor.set()
    monitor_thread.join()
    running = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
    )
    foreign = [
        line
        for line in running.splitlines()
        if gpu_uuid in line and line.split(",")[-1].strip() != str(os.getpid())
    ]
    for sample in load_samples:
        foreign.extend(
            line
            for line in sample.get("processes", "").splitlines()
            if gpu_uuid in line and line.split(",")[-1].strip() != str(os.getpid())
        )
    save(
        args.output,
        dict(
            provenance=before,
            valid=not foreign and not any("error" in s for s in load_samples),
            foreign_process_samples=foreign,
            telemetry_after=telemetry(),
            load_samples=load_samples,
            processes_after=running,
            gpu_uuid=gpu_uuid,
            accuracy_sha256=digest(args.accuracy),
            pipeline_sha256=digest(args.pipeline),
            resources=resources,
            suite=args.suite,
            seed=args.seed,
            rounds=args.rounds,
            rep_ms=args.rep,
            metric=CONTRACT["performance"]["metric"],
            records=records,
            summary=summary(records),
            completed_at=time.time(),
        ),
    )
    print(json.dumps(summary(records), indent=2))
    assert not foreign, f"Interference detected. Invalid raw samples saved: {foreign}"
    assert not any("error" in s for s in load_samples), "GPU telemetry was unavailable"


if __name__ == "__main__":
    main()
