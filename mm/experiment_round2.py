"""Bounded target-family experiments reusing the immutable native compiler and tests."""

import argparse
import json
import random
import statistics
from pathlib import Path

import experiment_round1 as shared
import torch
import triton
import triton.testing
from common import (
    ROOT,
    allocate,
    compile_ir,
    dump_kernel,
    frozen,
    gate,
    launch,
    save_json,
    sha256,
)
from make_round2 import CANDIDATES, FAMILY
from verify_pipeline import snapshot

TUNING = [768, 1024, 1152, 1280]
ORIGINAL_CASES = shared.test_cases


def extended_cases(config, suite):
    cases = ORIGINAL_CASES(config, suite)
    cases += [
        (m, n, k, pad, "normal")
        for m, n, k in [(704, 256, 80), (832, 256, 48), (960, 384, 192), (16, 16, 16)]
        for pad in [0, 16]
    ]
    return list(dict.fromkeys(cases))


# Extend only this process's test-case provider; production kernels, compiler,
# source files and the original round1 helper on disk remain untouched.
shared.test_cases = extended_cases


def compatible(base, ck, config):
    shared.compatible(base, ck, config)
    for name in ["global_scratch_size", "profile_scratch_size", "tensordesc_meta"]:
        assert getattr(base.metadata, name) == getattr(
            ck.metadata, name
        ), f"Workspace contract changed: {name}"


def check(args):
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = frozen()
    config = manifest["families"][FAMILY]
    shared.check_pipeline(args.pipeline, manifest)
    shared.idle()
    base = compile_ir(FAMILY)
    result = dict(
        passed=True,
        family=FAMILY,
        seed=args.seed,
        runner_sha256=sha256(__file__),
        baseline_manifest_sha256=sha256(ROOT / "baseline.json"),
        pipeline_sha256=sha256(args.pipeline),
        plan_sha256=sha256(ROOT / "optimization_round2.md"),
        records={},
    )
    for name in list(dict.fromkeys(["baseline", "r1_canonical", *args.names])):
        shared.idle()
        path = (ROOT / config["ttir_path"]).with_name(name + ".ttir")
        row = dict(ttir_path=str(path.relative_to(ROOT)), ttir_sha256=sha256(path))
        try:
            ck = base if name == "baseline" else compile_ir(FAMILY, name)
            compatible(base, ck, config)
            row["artifacts"] = dump_kernel(ck, f"{FAMILY}/{name}")
            row["executable_sections"] = shared.executable_hashes(ck)
            if name == "r1_canonical":
                assert row["executable_sections"] == shared.executable_hashes(base)
            row["accuracy"] = shared.validate(ck, config, args.seed)
            row["status"] = "PASS" if row["accuracy"]["passed"] else "CORRECTNESS_FAIL"
        except Exception as exc:
            row.update(
                status="COMPILE_OR_RUNTIME_FAIL",
                error_type=type(exc).__name__,
                error=str(exc),
            )
        result["records"][name] = row
        save_json(args.output / f"{name}.json", row)
        print(name, row["status"], row.get("error", ""), flush=True)
        shared.idle()
    save_json(args.output / "summary.json", result)


def diagnose_descriptor(args):
    gate()
    m = frozen()
    config = m["families"][FAMILY]
    path = ROOT / "results/runs/mm_opt_r2_20261119/device_descriptor_probe.ttir"
    record = dict(input_sha256=sha256(path), launched=False, functional_candidate=False)
    try:
        ck = triton.compile(str(path), options=config["options"])
        ck._init_handles()
        record.update(
            compiled=True,
            artifacts=dump_kernel(ck, "round2_descriptor_diagnostic"),
            requires_global_scratch=ck.metadata.global_scratch_size,
            reason="Compilation-only; new scratch allocator/workspace is outside frozen launch contract",
        )
    except Exception as exc:
        record.update(compiled=False, error_type=type(exc).__name__, error=str(exc))
    save_json(args.output, record)
    print(json.dumps(record, indent=2, default=str))


def bench(args):
    manifest = frozen()
    config = manifest["families"][FAMILY]
    shared.check_pipeline(args.pipeline, manifest)
    validation = json.loads(args.validation.read_text())
    assert validation["runner_sha256"] == sha256(__file__)
    assert validation["baseline_manifest_sha256"] == sha256(ROOT / "baseline.json")
    assert validation["pipeline_sha256"] == sha256(args.pipeline)
    names = list(dict.fromkeys(["r1_canonical", *args.names]))
    names = [n for n in names if n != "baseline"]
    excluded = []
    for name in list(names):
        if validation["records"][name]["status"] != "PASS":
            excluded.append(
                dict(name=name, status=validation["records"][name]["status"])
            )
            names.remove(name)
    kernels = {}
    infos = {}
    for name in ["baseline", *names]:
        r = validation["records"][name]
        assert r["status"] == "PASS"
        assert sha256(ROOT / r["ttir_path"]) == r["ttir_sha256"]
        ck = compile_ir(FAMILY, name)
        compatible(ck if not kernels else kernels["baseline"], ck, config)
        info = dump_kernel(ck, f"{FAMILY}/{name}")
        assert info["stages"]["cubin"] == r["artifacts"]["stages"]["cubin"]
        kernels[name] = ck
        infos[name] = dict(ttir_sha256=r["ttir_sha256"], **info)
    result = dict(
        family=FAMILY,
        runner_sha256=sha256(__file__),
        baseline_manifest_sha256=sha256(ROOT / "baseline.json"),
        validation_sha256=sha256(args.validation),
        pipeline_sha256=sha256(args.pipeline),
        seed=args.seed,
        rounds=args.rounds,
        rep_ms=20,
        resources=infos,
        excluded=excluded,
        records=[],
        gpu_before=snapshot(),
        metric="paired CUDA graph kernel time; same tensors/preallocated output; randomized pair order",
    )
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    sizes = args.sizes or (config["square_sizes"] if args.full else TUNING)
    assert all(n in config["square_sizes"] for n in sizes)
    for n in sizes:
        shared.idle()
        ts = allocate(n, n, n)
        for ck in kernels.values():
            for _ in range(10):
                launch(ck, ts, config)
        torch.cuda.synchronize()
        samples = {name: [] for name in names}
        for turn in range(args.rounds):
            order_candidates = list(names)
            rng.shuffle(order_candidates)
            for name in order_candidates:
                order = ["baseline", name]
                rng.shuffle(order)
                times = {}
                for label in order:
                    ck = kernels[label]
                    times[label] = (
                        triton.testing.do_bench_cudagraph(
                            lambda: launch(ck, ts, config), rep=20
                        )
                        * 1000
                    )
                samples[name].append(
                    dict(
                        round=turn,
                        order=order,
                        baseline_us=times["baseline"],
                        candidate_us=times[name],
                        speedup=times["baseline"] / times[name],
                    )
                )
        shared.idle()
        control = statistics.median(p["speedup"] for p in samples["r1_canonical"])
        for name, pairs in samples.items():
            ratio = statistics.median(p["speedup"] for p in pairs)
            drift = max(p["baseline_us"] for p in pairs) / min(
                p["baseline_us"] for p in pairs
            )
            valid = abs(control - 1) <= 0.02 and drift <= 1.10
            row = dict(
                name=name,
                shape=[n, n, n],
                pairs=pairs,
                speedup=ratio,
                baseline_us=statistics.median(p["baseline_us"] for p in pairs),
                candidate_us=statistics.median(p["candidate_us"] for p in pairs),
                control_speedup=control,
                baseline_drift=drift,
                valid=valid,
            )
            result["records"].append(row)
            print(
                f"{name} N={n}: {ratio:.5f}x AA={control:.4f} drift={drift:.4f} valid={valid}",
                flush=True,
            )
    result["summary"] = {
        name: shared.confidence([r for r in result["records"] if r["name"] == name])
        for name in names
    }
    result["gpu_after"] = snapshot()
    save_json(args.output, result)


def sanitize(args):
    manifest = frozen()
    config = manifest["families"][FAMILY]
    base = compile_ir(FAMILY)
    out = dict(family=FAMILY, seed=args.seed, records={}, passed=True)
    for name in args.names:
        shared.idle()
        ck = compile_ir(FAMILY, name)
        compatible(base, ck, config)
        row = shared.validate(ck, config, args.seed, "sanitizer")
        row["ttir_sha256"] = sha256(
            (ROOT / config["ttir_path"]).with_name(name + ".ttir")
        )
        out["records"][name] = row
        out["passed"] &= row["passed"]
        print(name, "sanitizer checks", len(row["cases"]), row["passed"], flush=True)
    save_json(args.output, out)
    assert out["passed"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["check", "bench", "descriptor", "sanitize"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pipeline", type=Path)
    p.add_argument("--validation", type=Path)
    p.add_argument("--names", nargs="+", default=list(CANDIDATES))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--full", action="store_true")
    p.add_argument("--sizes", type=int, nargs="+")
    args = p.parse_args()
    assert not args.output.exists(), "Use new evidence path"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    dict(check=check, bench=bench, descriptor=diagnose_descriptor, sanitize=sanitize)[
        args.mode
    ](args)


if __name__ == "__main__":
    main()
