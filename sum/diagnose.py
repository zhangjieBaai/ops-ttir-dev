"""Profile frozen sum kernels, or measure each stage outside profilers.

This script does not create candidates or change the frozen experiment protocol.
Profiler durations and isolated-stage timings are diagnostic, not promotion data.
"""

import argparse
import fcntl
import json
import os
import random
import statistics
import subprocess
import threading
from functools import partial
from pathlib import Path

import torch
import triton.testing
from common import (
    ATOL,
    L1_FACTOR,
    ROOT,
    allocate,
    compile_ir,
    frozen,
    gate,
    geometry,
    kernel_info,
    launch,
    receipt,
    save_json,
    sha256,
)

SIZES = [1024, 65536, 1048576, 8388608, 16777216, 67108864]


def normalize_uuid(value):
    return value.strip().removeprefix("GPU-").lower()


def parse_other_processes(result, uuid, current_pid):
    return [
        int(line.split(",")[1])
        for line in result.splitlines()
        if normalize_uuid(line.split(",")[0]) == normalize_uuid(uuid)
        and int(line.split(",")[1]) != current_pid
    ]


def other_processes(uuid):
    result = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
    )
    return parse_other_processes(result, uuid, os.getpid())


def stage_calls(kernels, tensors, manifest):
    n = tensors[0].numel()
    _, size, _ = geometry(n)
    result = []
    for stage, arguments, grid in [
        (0, (tensors[0], tensors[1], n), (size, 1, 1)),
        (1, (tensors[1], tensors[2], size), (1, 1, 1)),
    ]:
        group = manifest["shapes"][str(n)][stage]
        config = manifest["groups"][group]
        runtime = tuple(arguments[i] for i in config["runtime_indices"])
        result.append(partial(kernels[group][grid], *runtime))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["timing", "ncu", "timeline"], required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    parser.add_argument("--names", nargs="+", default=["baseline"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--per-process-output", action="store_true")
    args = parser.parse_args()
    if args.per_process_output:
        args.output = args.output.with_name(
            f"{args.output.stem}_{os.getpid()}{args.output.suffix}"
        )
    assert not args.output.exists()
    gate()
    manifest = frozen()
    assert all(str(n) in manifest["shapes"] for n in args.sizes)
    uuid = "GPU-" + normalize_uuid(str(torch.cuda.get_device_properties(0).uuid))
    lock = open(f"/root/tmp/ttir-device-{uuid}.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not other_processes(uuid), "Selected device has another compute process"
    stop = threading.Event()
    telemetry = []

    def monitor():
        while not stop.is_set():
            try:
                telemetry.append(
                    dict(
                        other_pids=other_processes(uuid),
                        state=subprocess.check_output(
                            [
                                "nvidia-smi",
                                "-i",
                                os.environ["CUDA_VISIBLE_DEVICES"],
                                "--query-gpu=timestamp,uuid,clocks.sm,clocks.mem,power.draw,"
                                "temperature.gpu,utilization.gpu,clocks_event_reasons.active",
                                "--format=csv,noheader",
                            ],
                            text=True,
                        ),
                    )
                )
            except Exception as error:
                telemetry.append(dict(error=repr(error)))
            stop.wait(1)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        run(args, manifest, telemetry)
    finally:
        stop.set()
        thread.join()


def run(args, manifest, telemetry):
    torch.manual_seed(20261119)
    rng = random.Random(20261119)
    groups = {g for n in args.sizes for g in manifest["shapes"][str(n)]}
    kernels = {
        name: {g: compile_ir(g, name) for g in sorted(groups)} for name in args.names
    }
    # Reuse exact-hash full numerical and pipeline evidence from the completed round.
    validation = json.loads((ROOT / "results/runs/round1/accuracy.json").read_text())
    pipeline = json.loads((ROOT / "results/runs/round1/pipeline.json").read_text())
    assert validation["passed"] and pipeline["passed"]
    artifacts = {}
    for name, compiled in kernels.items():
        artifacts[name] = {}
        for group, ck in compiled.items():
            path = (ROOT / manifest["groups"][group]["path"]).with_name(f"{name}.ttir")
            info = kernel_info(ck, path)
            assert info["input_sha256"] == validation["kernels"][name][group]
            assert (
                info["stages"]["cubin"]
                == pipeline["artifacts"][name][group]["stages"]["cubin"]
            )
            artifacts[name][group] = info
    records = []
    for n in args.sizes:
        tensors = allocate(n)
        reference = tensors[0].double().sum()
        bound = ATOL + L1_FACTOR * tensors[0].double().abs().sum()
        for name, compiled in kernels.items():
            tensors[1].fill_(float("nan"))
            tensors[2].fill_(float("nan"))
            launch(compiled, tensors, manifest)
            error = (tensors[2].double() - reference).abs().item()
            assert error <= bound.item()
            stages = stage_calls(compiled, tensors, manifest)
            full = partial(launch, compiled, tensors, manifest)
            for _ in range(100):
                full()
            torch.cuda.synchronize()
            block, mid, tail = geometry(n)
            record = dict(
                n=n,
                name=name,
                block=block,
                mid=mid,
                tail=tail,
                groups=manifest["shapes"][str(n)],
                error=error,
            )
            if args.mode == "timing":
                calls = dict(stage1=stages[0], stage2=stages[1], total=full)
                for call in calls.values():
                    triton.testing.do_bench_cudagraph(call, rep=100)
                samples = {key: [] for key in calls}
                orders = []
                for turn in range(args.rounds):
                    order = list(calls)
                    rng.shuffle(order)
                    orders.append(order)
                    for key in order:
                        raw = triton.testing.do_bench_cudagraph(
                            calls[key], rep=20, return_mode="all"
                        )
                        samples[key].append([value * 1000 for value in raw])
                record.update(
                    replay_samples_us=samples,
                    orders=orders,
                    median_us={
                        key: statistics.median(statistics.mean(pair) for pair in raw)
                        for key, raw in samples.items()
                    },
                )
            else:
                graph = None
                if args.mode == "timeline":
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(100):
                            full()
                    graph.replay()
                    torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStart()
                torch.cuda.nvtx.range_push(f"n={n};name={name}")
                if graph is None:
                    full()
                else:
                    graph.replay()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
                torch.cuda.cudart().cudaProfilerStop()
                record["profiled_launches"] = 2 if graph is None else 200
            records.append(record)
            print(f"DIAG {record}", flush=True)
    save_json(
        args.output,
        dict(
            **receipt(),
            mode=args.mode,
            script_sha256=sha256(__file__),
            validation_sha256=sha256(ROOT / "results/runs/round1/accuracy.json"),
            pipeline_sha256=sha256(ROOT / "results/runs/round1/pipeline.json"),
            records=records,
            artifacts=artifacts,
            telemetry=list(telemetry),
            uncontended=all(
                not sample.get("other_pids") and "error" not in sample
                for sample in telemetry
            ),
            device_properties=str(torch.cuda.get_device_properties(0)),
            sm_count=torch.cuda.get_device_properties(0).multi_processor_count,
            l2_bytes=torch.cuda.get_device_properties(0).L2_cache_size,
            cache="warmup100 full calls; no clearing in driver; NCU cache policy set separately",
            timing="diagnostic isolated-stage and total graph; profiler data is separate",
        ),
    )


if __name__ == "__main__":
    main()
