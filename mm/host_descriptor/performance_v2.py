"""Paired host-descriptor optimization with a user-authorized unresolved-race exception."""

# flake8: noqa: E402 -- standalone MM owners require explicit path setup.

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import triton.testing
from bridge import compile_file
from common import (
    allocate,
    compile_ir,
    dump_kernel,
    frozen,
    launch,
    runtime_args,
    save_json,
    sha256,
)
from experiment_round1 import confidence, idle
from probe import CONFIG, ROOT, descriptors
from triton.backends.nvidia.compiler import CUDABackend
from verify_pipeline import snapshot

BASE = "envelope_baseline"
NAMES = ["envelope_host_v1", "host_pid_v2"]
REGISTRY = ROOT / "results/prototype_inputs_v4.json"
WAIVER = "Unresolved racecheck warnings retained; user explicitly permits continued performance evaluation, not sync qualification"


def fingerprints():
    files = [
        ROOT / "optimization_plan_v2.md",
        ROOT / "performance_v2.py",
        ROOT / "bridge.py",
        REGISTRY,
    ]
    return {str(p): sha256(p) for p in files}


def prepare(ts):
    a, b, c = ts[:3]
    raw = runtime_args(ts)
    desc = descriptors(a, b, c)
    args = (
        *desc,
        a,
        b,
        c,
        raw[3],
        raw[4],
        raw[5],
        a.stride(0),
        b.stride(0),
        c.stride(0),
    )
    grid = ((a.shape[0] + 63) // 64 * ((b.shape[1] + 127) // 128), 1, 1)
    return grid, args


def invoke(ck, prepared):
    grid, args = prepared
    ck[grid](*args)


def load_all(names):
    kernels = {}
    info = {}
    audits = {}
    for name in [BASE, *names]:
        if name in kernels:
            continue
        ck, audit = compile_file(
            ROOT / "ir" / f"{name}.ttir", CONFIG["options"], REGISTRY
        )
        if kernels:
            base = kernels[BASE]
            assert ck.src.signature == base.src.signature
            assert ck.metadata.target == base.metadata.target
            options = CUDABackend(base.metadata.target).parse_options(CONFIG["options"])
            for key in vars(options):
                assert getattr(ck.metadata, key) == getattr(base.metadata, key), key
            assert (
                audit["native_expanded_signature"]
                == audits[BASE]["native_expanded_signature"]
            )
        assert ck.metadata.global_scratch_size == ck.metadata.profile_scratch_size == 0
        kernels[name] = ck
        info[name] = dump_kernel(ck, "host_descriptor/" + name)
        audits[name] = audit
    return kernels, info, audits


def wall_metrics(ts, kernels, original, seed):
    rng = random.Random(seed)
    # All cold setup and native descriptor encoder initialization precedes measurement.
    for ck in kernels.values():
        invoke(ck, prepare(ts))
    launch(original, ts, CONFIG)
    torch.cuda.synchronize()
    prep_samples = []
    for _ in range(50):
        t = time.perf_counter_ns()
        descriptors(*ts[:3])
        prep_samples.append((time.perf_counter_ns() - t) / 1000)
    records = {}
    for mode in [
        "preallocated_with_descriptor_prepare",
        "allocation_and_descriptor_prepare",
    ]:
        values = {name: [] for name in [*kernels, "original_pointer_baseline"]}
        orders = []
        for turn in range(10):
            order = list(values)
            rng.shuffle(order)
            orders.append(order)
            for name in order:
                torch.cuda.synchronize()
                began = time.perf_counter_ns()
                if mode.startswith("allocation"):
                    a, b = ts[:2]
                    c = torch.empty(
                        (a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype
                    )
                    tensors = (a, b, c)
                else:
                    tensors = ts
                if name == "original_pointer_baseline":
                    launch(original, tensors, CONFIG)
                else:
                    invoke(kernels[name], prepare(tensors))
                torch.cuda.synchronize()
                values[name].append((time.perf_counter_ns() - began) / 1000)
        records[mode] = dict(
            samples_us=values,
            orders=orders,
            median_us={k: statistics.median(v) for k, v in values.items()},
        )
    return dict(
        descriptor_object_prepare_us=prep_samples,
        descriptor_prepare_median_us=statistics.median(prep_samples),
        records=records,
        scope="CPUwall includes per-call preparation/native encoding/launch/completion sync; allocation mode also includes output torch.empty. Not kernel time.",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--sizes", type=int, nargs="+", default=CONFIG["square_sizes"])
    p.add_argument("--names", nargs="+", default=NAMES)
    p.add_argument("--no-wall", action="store_true")
    args = p.parse_args()
    assert not args.output.exists()
    idle()
    frozen()
    torch.manual_seed(args.seed)
    validation = json.loads(args.validation.read_text())
    assert validation["passed"]
    assert validation["bridge_sha256"] == sha256(ROOT / "bridge.py")
    kernels, resources, audits = load_all(args.names)
    for name in kernels:
        row = validation["records"][name]
        assert row["passed"] and len(row["cases"]) == 37
        assert row["artifacts"]["stages"]["cubin"] == resources[name]["stages"]["cubin"]
        assert row["audit"]["input_sha256"] == sha256(ROOT / "ir" / f"{name}.ttir")
    original = compile_ir("bm64_bn128_bk64_g8_w4_s4")
    result = dict(
        fingerprints=fingerprints(),
        validation_sha256=sha256(args.validation),
        baseline_manifest_sha256=sha256(ROOT.parent / "baseline.json"),
        seed=args.seed,
        rounds=args.rounds,
        warning_exception=WAIVER,
        sync_qualified=False,
        resources=resources,
        audits=audits,
        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
        gpu_before=snapshot(),
        records=[],
        wall_metrics={},
        metric="Preallocated descriptors+output CUDA graph device time; excludes CPU encoding/allocation; includes every device operation",
    )
    rng = random.Random(args.seed)
    for n in args.sizes:
        assert n in CONFIG["square_sizes"]
        idle()
        ts = allocate(n, n, n)
        prepared = prepare(ts)
        # Check the exact prepared entry and its current inputs before timing.
        launch(original, ts, CONFIG)
        reference = ts[2].clone()
        for name, ck in kernels.items():
            ts[2].fill_(float("nan"))
            invoke(ck, prepared)
            torch.testing.assert_close(ts[2], reference, atol=0.01, rtol=0.001)
            for _ in range(10):
                invoke(ck, prepared)
        torch.cuda.synchronize()
        samples = {label: [] for label in ["control", *args.names]}
        for turn in range(args.rounds):
            labels = list(samples)
            rng.shuffle(labels)
            for label in labels:
                order = [BASE, label]
                rng.shuffle(order)
                times = {}
                for who in order:
                    ck = kernels[BASE if who == "control" else who]
                    times[who] = (
                        triton.testing.do_bench_cudagraph(
                            lambda: invoke(ck, prepared), rep=20
                        )
                        * 1000
                    )
                samples[label].append(
                    dict(
                        round=turn,
                        order=order,
                        baseline_us=times[BASE],
                        candidate_us=times[label],
                        speedup=times[BASE] / times[label],
                    )
                )
        idle()
        control = statistics.median(p["speedup"] for p in samples["control"])
        for name, pairs in samples.items():
            ratio = statistics.median(p["speedup"] for p in pairs)
            drift = max(p["baseline_us"] for p in pairs) / min(
                p["baseline_us"] for p in pairs
            )
            row = dict(
                name=name,
                shape=[n, n, n],
                pairs=pairs,
                speedup=ratio,
                baseline_drift=drift,
                control_speedup=control,
                valid=abs(control - 1) <= 0.02 and drift <= 1.10,
                baseline_us=statistics.median(p["baseline_us"] for p in pairs),
                candidate_us=statistics.median(p["candidate_us"] for p in pairs),
            )
            result["records"].append(row)
            print(
                n,
                name,
                f"{ratio:.5f}x",
                "AA",
                control,
                "valid",
                row["valid"],
                flush=True,
            )
        if not args.no_wall:
            result["wall_metrics"][str(n)] = wall_metrics(
                ts, kernels, original, args.seed + n
            )
    result["summary"] = {
        name: confidence([x for x in result["records"] if x["name"] == name])
        for name in samples
    }
    result["gpu_after"] = snapshot()
    save_json(args.output, result)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
