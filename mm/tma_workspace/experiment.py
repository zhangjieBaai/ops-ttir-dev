"""Matched baseline/device-TMA experiments with native runtime workspace allocation."""

# flake8: noqa: E402 -- standalone MM helpers are imported after explicit path bootstrap.

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiment_round2 as tests
import torch
import triton.testing
from common import allocate, dump_kernel, frozen, launch_source, save_json, sha256
from engine import (
    CONFIG,
    FAMILY,
    MM_ROOT,
    OPTIONS,
    ROOT,
    NativeScratchAllocator,
    allocation_scope,
    compare_contract,
    compile_native,
    fingerprint,
    launch_native,
)
from experiment_round1 import confidence, idle
from triton.runtime import _allocation
from verify_pipeline import snapshot


def pipeline(args):
    manifest = frozen()
    before = _allocation._allocator.get()
    allocator = NativeScratchAllocator()
    base = compile_native("baseline")
    candidate = compile_native("device_tma_v1")
    compare_contract(base, candidate)
    assert (
        base.asm["cubin"]
        and sha256(ROOT / "ir/baseline.ttir")
        == manifest["families"][FAMILY]["ttir_sha256"]
    )
    records = []
    with allocation_scope(allocator):
        for n in CONFIG["square_sizes"]:
            idle()
            ts = allocate(n, n, n)
            source = launch_source(ts, CONFIG)
            source_output = ts[2].clone()
            assert source.asm["ttir"] == (ROOT / "ir/baseline.ttir").read_text()
            for stage in ["llir", "ptx", "cubin"]:
                assert source.asm[stage] == base.asm[stage], stage
            ts[3].fill_(float("nan"))
            launch_native(base, ts)
            torch.testing.assert_close(ts[2], source_output, rtol=0, atol=0)
            # Candidate is checked by the full numeric suite before any timing.
            records.append(dict(shape=[n, n, n], source_file_equivalence=True))
    assert _allocation._allocator.get() is before
    # Explicit exception-restoration counterexample; no kernel in this scope.
    try:
        with allocation_scope(allocator):
            raise ValueError("restoration probe")
    except ValueError:
        pass
    assert _allocation._allocator.get() is before
    out = dict(
        passed=True,
        experiment="device-tma-workspace-v1",
        family=FAMILY,
        records=records,
        fingerprints=fingerprint(),
        baseline_manifest_sha256=sha256(MM_ROOT / "baseline.json"),
        baseline=dump_kernel(base, "tma_workspace/baseline"),
        candidate=dump_kernel(candidate, "tma_workspace/device_tma_v1"),
        options=OPTIONS,
        allocator_restored=True,
        exception_restoration=True,
        gpu=snapshot(),
        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
    )
    save_json(args.output, out)
    print(
        "PASS: matched baseline source chain and allocator normal/exception restoration"
    )


def check(args):
    args.output.mkdir(parents=True, exist_ok=False)
    pipe = json.loads(args.pipeline.read_text())
    assert pipe["passed"] and pipe["fingerprints"] == fingerprint()
    allocator = NativeScratchAllocator()
    before = _allocation._allocator.get()
    out = dict(
        passed=True,
        seed=args.seed,
        fingerprints=fingerprint(),
        pipeline_sha256=sha256(args.pipeline),
        records={},
    )
    base = compile_native("baseline")
    with allocation_scope(allocator):
        for name in ["baseline", *args.names]:
            idle()
            ck = base if name == "baseline" else compile_native(name)
            if name != "baseline":
                compare_contract(base, ck)
            row = dict(
                ttir_sha256=sha256(ROOT / "ir" / f"{name}.ttir"),
                artifacts=dump_kernel(ck, f"tma_workspace/{name}"),
            )
            row["accuracy"] = tests.shared.validate(
                ck, CONFIG, args.seed, "sanitizer" if args.sanitizer else "full"
            )
            row["passed"] = row["accuracy"]["passed"]
            out["passed"] &= row["passed"]
            out["records"][name] = row
            save_json(args.output / f"{name}.json", row)
            print(
                name, "checks", len(row["accuracy"]["cases"]), row["passed"], flush=True
            )
            assert row["passed"], "Correctness failure blocks timing"
    assert _allocation._allocator.get() is before
    out.update(
        allocator_restored=True,
        allocator_calls=allocator.calls,
        allocator_samples=allocator.samples,
        max_workspace_bytes=allocator.max_bytes,
    )
    save_json(args.output / "summary.json", out)


def bench(args):
    pipe = json.loads(args.pipeline.read_text())
    validation = json.loads(args.validation.read_text())
    assert pipe["passed"] and validation["passed"]
    assert validation["fingerprints"] == pipe["fingerprints"] == fingerprint()
    assert validation["pipeline_sha256"] == sha256(args.pipeline)
    names = ["baseline", *args.names]
    kernels = {n: compile_native(n) for n in names}
    for n, ck in kernels.items():
        row = validation["records"][n]
        assert row["passed"]
        assert sha256(ROOT / "ir" / f"{n}.ttir") == row["ttir_sha256"]
        assert (
            dump_kernel(ck, f"tma_workspace/{n}")["stages"]["cubin"]
            == row["artifacts"]["stages"]["cubin"]
        )
    for n in args.names:
        compare_contract(kernels["baseline"], kernels[n])
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    allocator = NativeScratchAllocator()
    result = dict(
        experiment="device-tma-workspace-v1",
        fingerprints=fingerprint(),
        seed=args.seed,
        rounds=args.rounds,
        validation_sha256=sha256(args.validation),
        pipeline_sha256=sha256(args.pipeline),
        records=[],
        gpu_before=snapshot(),
        resources={
            n: dump_kernel(ck, f"tma_workspace/{n}") for n, ck in kernels.items()
        },
    )
    sizes = CONFIG["square_sizes"] if args.full else [768, 1024, 1152, 1280]
    before = _allocation._allocator.get()
    with allocation_scope(allocator):
        for n in sizes:
            idle()
            ts = allocate(n, n, n)
            for ck in kernels.values():
                for _ in range(10):
                    launch_native(ck, ts)
            torch.cuda.synchronize()
            samples = {label: [] for label in ["control", *args.names]}
            for turn in range(args.rounds):
                labels = list(samples)
                rng.shuffle(labels)
                for label in labels:
                    order = ["baseline", label]
                    rng.shuffle(order)
                    times = {}
                    for who in order:
                        ck = kernels["baseline" if who == "control" else who]
                        times[who] = (
                            triton.testing.do_bench_cudagraph(
                                lambda: launch_native(ck, ts), rep=20
                            )
                            * 1000
                        )
                    samples[label].append(
                        dict(
                            round=turn,
                            order=order,
                            baseline_us=times["baseline"],
                            candidate_us=times[label],
                            speedup=times["baseline"] / times[label],
                        )
                    )
            idle()
            control = statistics.median(p["speedup"] for p in samples["control"])
            for label, pairs in samples.items():
                ratio = statistics.median(p["speedup"] for p in pairs)
                drift = max(p["baseline_us"] for p in pairs) / min(
                    p["baseline_us"] for p in pairs
                )
                row = dict(
                    name=label,
                    shape=[n, n, n],
                    pairs=pairs,
                    speedup=ratio,
                    control_speedup=control,
                    baseline_drift=drift,
                    valid=abs(control - 1) <= 0.02 and drift <= 1.10,
                    baseline_us=statistics.median(p["baseline_us"] for p in pairs),
                    candidate_us=statistics.median(p["candidate_us"] for p in pairs),
                )
                result["records"].append(row)
                print(n, label, ratio, "valid", row["valid"], flush=True)
    assert _allocation._allocator.get() is before
    result.update(
        summary={
            label: confidence([r for r in result["records"] if r["name"] == label])
            for label in samples
        },
        allocator_restored=True,
        allocator_calls=allocator.calls,
        allocator_samples=allocator.samples,
        max_workspace_bytes=allocator.max_bytes,
        gpu_after=snapshot(),
    )
    save_json(args.output, result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["pipeline", "check", "bench"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pipeline", type=Path)
    p.add_argument("--validation", type=Path)
    p.add_argument("--names", nargs="+", default=["device_tma_v1"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--full", action="store_true")
    p.add_argument("--sanitizer", action="store_true")
    args = p.parse_args()
    assert not args.output.exists(), "New evidence path required"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    dict(pipeline=pipeline, check=check, bench=bench)[args.mode](args)


if __name__ == "__main__":
    main()
