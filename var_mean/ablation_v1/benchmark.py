"""Randomized complete blocks: every factor variant uses the same input each round."""

import argparse
import fcntl
import json
import os
import random
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402
import triton  # noqa: E402
import triton.testing  # noqa: E402
from common import (  # noqa: E402
    OPTIONS,
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
from triton.backends.nvidia.compiler import CUDABackend  # noqa: E402

HERE = Path(__file__).resolve().parent


def process_list():
    return subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accuracy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pipeline", type=Path, default=HERE / "results/pipeline_gpu4.json"
    )
    parser.add_argument("--shape-indices", nargs="+", type=int)
    args = parser.parse_args()
    source_before = digest(__file__)
    device = os.environ["CUDA_VISIBLE_DEVICES"]
    assert device.isdecimal()
    uuid = subprocess.check_output(
        ["nvidia-smi", "-i", device, "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    lock = open(f"/root/tmp/ttir-device-{uuid}.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert uuid not in process_list(), "Device occupied before initialization"
    manifest = json.loads((HERE / "manifest.json").read_text())
    accuracy = json.loads(Path(args.accuracy).read_text())
    pipeline = json.loads(args.pipeline.read_text())
    assert accuracy["passed"] and pipeline["passed"]
    assert (
        manifest["contract_sha256"]
        == accuracy["provenance"]["contract_sha256"]
        == digest(ROOT / "contract.json")
    )
    assert manifest["plan_sha256"] == digest(HERE / "plan.md")
    variants = manifest["variants"]
    kernels = {bits: compile_ir(v["name"]) for bits, v in variants.items()}
    original = compile_ir("baseline")
    assert (
        original.asm["cubin"] == kernels["000"].asm["cubin"]
    ), "canonicalization is not a free factor"
    assert original.asm["ptx"] == kernels["000"].asm["ptx"]
    assert (
        identity("baseline", original)["products"]["cubin"]
        == pipeline["baseline"]["products"]["cubin"]
    )
    backend = CUDABackend(triton.runtime.driver.active.get_current_target())
    parsed = backend.parse_options(OPTIONS).__dict__
    identities = {}
    for bits, ck in kernels.items():
        name = variants[bits]["name"]
        assert all(
            ck.metadata._asdict()[k] == original.metadata._asdict()[k] for k in parsed
        )
        ident = identity(name, ck)
        assert (
            ident["ttir_sha256"]
            == variants[bits]["sha256"]
            == accuracy["identities"][name]["ttir_sha256"]
        )
        assert (
            ident["products"]["cubin"]
            == accuracy["identities"][name]["products"]["cubin"]
        )
        identities[bits] = ident
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    meta = provenance()
    traces = []
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            try:
                state = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "-i",
                        device,
                        "--query-gpu=timestamp,uuid,clocks.sm,clocks.mem,power.draw,temperature.gpu,"
                        "utilization.gpu,clocks_event_reasons.active",
                        "--format=csv,noheader",
                    ],
                    text=True,
                )
                traces.append(dict(state=state, processes=process_list()))
            except Exception as exc:
                traces.append(dict(error=str(exc)))
            stop.wait(0.5)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    records = []
    indices = (
        args.shape_indices
        if args.shape_indices is not None
        else list(range(len(SHAPES)))
    )
    assert len(indices) == len(set(indices)) and all(
        0 <= index < len(SHAPES) for index in indices
    )
    for index in indices:
        m, n = SHAPES[index]
        ts = allocate(m, n)
        calls = {bits: (lambda ck=ck: launch(ck, ts)) for bits, ck in kernels.items()}
        for call in calls.values():
            ts[3].fill_(float("nan"))
            ts[4].fill_(float("nan"))
            for _ in range(10):
                call()
            triton.testing.do_bench_cudagraph(call, rep=50)
        torch.cuda.synchronize()
        blocks = []
        for iteration in range(args.rounds):
            order = list(calls)
            rng.shuffle(order)
            times = {
                bits: triton.testing.do_bench_cudagraph(calls[bits], rep=20) * 1000
                for bits in order
            }
            blocks.append(dict(block=iteration, order=order, times_us=times))
        records.append(dict(shape=[m, n], blocks=blocks))
        print([m, n], flush=True)
    torch.cuda.synchronize()
    stop.set()
    monitor_thread.join()
    foreign = []
    for sample in [*traces, dict(processes=process_list())]:
        foreign.extend(
            line
            for line in sample.get("processes", "").splitlines()
            if uuid in line and line.split(",")[-1].strip() != str(os.getpid())
        )
    source_unchanged = digest(__file__) == source_before
    valid = not foreign and not any("error" in s for s in traces) and source_unchanged
    save(
        args.output,
        dict(
            valid=valid,
            provenance=meta,
            manifest_sha256=digest(HERE / "manifest.json"),
            runner_sha256=digest(__file__),
            runner_before_sha256=source_before,
            runner_unchanged=source_unchanged,
            accuracy_sha256=digest(args.accuracy),
            pipeline_sha256=digest(args.pipeline),
            pipeline_path=str(args.pipeline),
            shape_indices=indices,
            identities=identities,
            original_baseline=identity("baseline", original),
            canonical_cubin_identical=True,
            canonical_ptx_identical=True,
            records=records,
            rounds=args.rounds,
            seed=args.seed,
            rep_ms=20,
            gpu_uuid=uuid,
            telemetry=traces,
            telemetry_after=telemetry(),
            foreign=foreign,
        ),
    )
    assert valid, "Invalid raw measurement retained"


if __name__ == "__main__":
    main()
