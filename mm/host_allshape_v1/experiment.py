"""Native six-family TTIR host-TMA validation and paired measurement."""

# flake8: noqa: E402 -- standalone experiment imports the frozen MM helpers.
import argparse
import json
import random
import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MM = ROOT.parent
sys.path.insert(0, str(MM))
sys.path.insert(0, str(MM / "host_descriptor"))

import torch
import triton
import triton.testing
from bridge import DESC, descriptor_source_reader
from common import (
    allocate,
    compile_ir,
    dump_kernel,
    frozen,
    gate,
    grid,
    launch,
    runtime_args,
    save_json,
    sha256,
)
from experiment_round1 import confidence, idle, test_cases
from triton.backends.nvidia.compiler import CUDABackend
from triton.backends.nvidia.driver import expand_signature
from triton.tools.tensor_descriptor import TensorDescriptor
from verify_pipeline import snapshot

NAMES = ["baseline", "host_tma", "host_pid"]


def fingerprints():
    paths = [
        ROOT / "experiment.py",
        ROOT / "plan.md",
        ROOT / "inputs_v4.json",
        MM / "host_descriptor/bridge.py",
    ]
    return {str(p): sha256(p) for p in paths}


def compile_native(family, name, config):
    gate()
    registry = json.loads((ROOT / "inputs_v4.json").read_text())
    assert registry["baseline_manifest_sha256"] == sha256(MM / "baseline.json")
    source = registry["families"][family][name]
    path = ROOT / source["path"]
    assert sha256(path) == source["sha256"]
    raw = triton.compile(str(path), options=config["options"])
    with descriptor_source_reader(path):
        ck = triton.compile(str(path), options=config["options"])
    assert ck.src.src == raw.src.src == path.read_text()
    for stage in ["ttgir", "llir", "ptx", "cubin"]:
        assert ck.asm[stage] == raw.asm[stage]
    expected = [
        "nvTmaDesc" if DESC.fullmatch(t) else t for t in raw.src.signature.values()
    ]
    assert (
        expand_signature(ck.src.signature.values(), ck.metadata.tensordesc_meta)
        == expected
    )
    assert len(ck.src.signature) == 12 and len(expected) == 24
    assert ck.metadata.global_scratch_size == ck.metadata.profile_scratch_size == 0
    ck._init_handles()
    return ck, dict(
        ck.src.host_abi_audit,
        expanded_signature=expected,
        raw_adapted_bytes_identical=True,
    )


def load(family, config, names=NAMES):
    kernels, resources, audits, failures = {}, {}, {}, {}
    base = None
    for name in names:
        try:
            ck, audit = compile_native(family, name, config)
            if base is None:
                base = ck
            else:
                assert ck.src.signature == base.src.signature
                assert ck.metadata.target == base.metadata.target
                options = CUDABackend(base.metadata.target).parse_options(
                    config["options"]
                )
                for key in vars(options):
                    assert getattr(ck.metadata, key) == getattr(base.metadata, key), key
            kernels[name] = ck
            resources[name] = dump_kernel(ck, f"host_allshape_v1/{family}/{name}")
            audits[name] = audit
        except Exception as exc:
            failures[name] = dict(
                status="COMPILE_OR_LOAD_FAIL",
                error=str(exc),
                traceback=traceback.format_exc(),
            )
            print(family, name, "COMPILE_OR_LOAD_FAIL", str(exc)[:400], flush=True)
    assert "baseline" in kernels, failures
    return kernels, resources, audits, failures


def prepare(ts, config):
    a, b, c = ts[:3]
    runtime_args(ts)
    bm, bn, bk = [config["meta"]["BLOCK_SIZE_" + key] for key in "MNK"]
    desc = tuple(
        TensorDescriptor(x, list(x.shape), list(x.stride()), tile)
        for x, tile in [
            (a, [bm, bk]),
            (b, [bk, bn]),
            (c, [bm, bn]),
        ]
    )
    return grid(ts, config["meta"]), (
        *desc,
        a,
        b,
        c,
        a.shape[0],
        b.shape[1],
        a.shape[1],
        a.stride(0),
        b.stride(0),
        c.stride(0),
    )


def invoke(ck, prepared):
    launch_grid, args = prepared
    ck[launch_grid](*args)


def cases_for(config, suite):
    bm, bn, bk = [config["meta"]["BLOCK_SIZE_" + key] for key in "MNK"]
    if suite == "sanitize":
        return [
            (m, n, k, pad, "normal")
            for m, n, k, pad in [
                (16, 16, 16, 0),
                (bm + 16, bn + 16, bk - 16, 16),
                (bm + 16, bn + 16, bk, 0),
                (bm + 16, bn + 16, bk + 16, 16),
                (11 * bm, 2 * bn, bk + 16, 0),
                (11 * bm, 2 * bn, bk + 16, 16),
            ]
        ]
    cases = test_cases(config, "full")
    cases += [(11 * bm, 2 * bn, bk + 16, pad, "normal") for pad in [0, 16]]
    return list(dict.fromkeys(cases))


def correctness(kernels, config, seed, suite):
    torch.manual_seed(seed)
    result = {name: dict(passed=True, cases=[]) for name in kernels}
    for m, n, k, pad, dist in cases_for(config, suite):
        ts = allocate(m, n, k, pad)
        a, b, c, storage = ts
        if dist == "zero":
            a.zero_()
        elif dist == "constant":
            a.fill_(0.25)
            b.fill_(0.25)
        elif dist == "large":
            a.mul_(4)
            b.mul_(4)
        elif dist == "small":
            a.mul_(1e-3)
            b.mul_(1e-3)
        elif dist == "alternating":
            b[::2].copy_(b[1::2])
            a[:, 1::2].copy_(-a[:, ::2])
        before = [a.clone(), b.clone()]
        refs = {
            "native_fp16": torch.mm(a, b),
            "native_fp32": torch.mm(a.float(), b.float()).half(),
        }
        if max(m, n, k) <= 272:
            refs["cpu_fp64"] = (
                torch.mm(a.cpu().double(), b.cpu().double()).half().cuda()
            )
        prepared = prepare(ts, config)
        for name, ck in kernels.items():
            storage.fill_(float("nan"))
            invoke(ck, prepared)
            torch.cuda.synchronize()
            errors = {
                key: (c.float() - ref.float()).abs().max().item()
                for key, ref in refs.items()
            }
            passed = bool(torch.isfinite(c).all()) and bool(
                storage[:, n:].isnan().all()
            )
            passed &= all(torch.equal(x, y) for x, y in zip([a, b], before))
            passed &= all(
                torch.allclose(c, ref, atol=0.01, rtol=0.001) for ref in refs.values()
            )
            result[name]["cases"].append(
                dict(
                    shape=[m, n, k],
                    pad=pad,
                    distribution=dist,
                    errors=errors,
                    passed=passed,
                )
            )
            result[name]["passed"] &= passed
            if not passed:
                print("NUMERICAL_FAIL", name, m, n, k, pad, dist, errors, flush=True)
    return result


def validate(args):
    manifest = frozen()
    pipeline = json.loads((ROOT / "results/pipeline.json").read_text())
    assert pipeline["passed"] and pipeline["config_sha256"] == sha256(
        MM / "baseline.json"
    )
    assert pipeline["torch_device_uuid"] == str(
        torch.cuda.get_device_properties(0).uuid
    )
    result = dict(
        seed=args.seed,
        fingerprints=fingerprints(),
        pipeline_sha256=sha256(ROOT / "results/pipeline.json"),
        families={},
    )
    for family, config in manifest["families"].items():
        if args.families and family not in args.families:
            continue
        idle()
        kernels, resources, audits, failures = load(family, config, args.names)
        checks = correctness(
            kernels,
            config,
            args.seed,
            "sanitize" if args.mode == "sanitize" else "full",
        )
        assert checks["baseline"]["passed"]
        result["families"][family] = dict(
            resources=resources, audits=audits, failures=failures, validation=checks
        )
        print(
            family,
            {name: (r["passed"], len(r["cases"])) for name, r in checks.items()},
            flush=True,
        )
        idle()
    save_json(args.output, result)
    assert all(
        v["passed"]
        for group in result["families"].values()
        for v in group["validation"].values()
    ), "Numerical failure; no timing for failing candidate"


def wall_metrics(ts, kernels, original, config, rng):
    times = {name: [] for name in [*kernels, "original_pointer"]}
    orders = []
    for _ in range(10):
        order = list(times)
        rng.shuffle(order)
        orders.append(order)
        for name in order:
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            if name == "original_pointer":
                launch(original, ts, config)
            else:
                invoke(kernels[name], prepare(ts, config))
            torch.cuda.synchronize()
            times[name].append((time.perf_counter_ns() - start) / 1000)
    return dict(
        samples_us=times,
        orders=orders,
        median_us={k: statistics.median(v) for k, v in times.items()},
        scope="Prepared output; includes descriptor prepare/encode/launch/completion sync; no allocation",
    )


def benchmark(args):
    manifest = frozen()
    validation = json.loads(args.validation.read_text())
    assert validation["fingerprints"] == fingerprints()
    assert validation["pipeline_sha256"] == sha256(ROOT / "results/pipeline.json")
    result = dict(
        seed=args.seed,
        rounds=args.rounds,
        rep_ms=20,
        fingerprints=fingerprints(),
        validation_sha256=sha256(args.validation),
        gpu_before=snapshot(),
        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
        metric="Prepared-descriptor/output CUDA graph device time; CPU encoding/allocation excluded",
        warning_context="Previously unresolved TMA racecheck warnings retained; sync qualification not granted",
        sync_qualified=False,
        families={},
        records=[],
        wall_metrics={},
    )
    rng = random.Random(args.seed)
    for family, config in manifest["families"].items():
        if args.families and family not in args.families:
            continue
        idle()
        group = validation["families"][family]
        names = [
            name
            for name in NAMES
            if name in group["validation"] and group["validation"][name]["passed"]
        ]
        kernels, resources, audits, failures = load(family, config, names)
        assert not failures
        for name in names:
            assert (
                resources[name]["stages"]["cubin"]
                == group["resources"][name]["stages"]["cubin"]
            )
        original = compile_ir(family)
        result["families"][family] = dict(resources=resources, audits=audits)
        for n in config["square_sizes"]:
            if args.sizes and n not in args.sizes:
                continue
            idle()
            ts = allocate(n, n, n)
            prepared = prepare(ts, config)
            launch(original, ts, config)
            reference = ts[2].clone()
            for name, ck in kernels.items():
                ts[2].fill_(float("nan"))
                invoke(ck, prepared)
                torch.testing.assert_close(ts[2], reference, atol=0.01, rtol=0.001)
                for _ in range(10):
                    invoke(ck, prepared)
            torch.cuda.synchronize()
            labels = [
                "control",
                *[name for name in names if name != "baseline"],
                "original_pointer",
            ]
            samples = {name: [] for name in labels}
            for turn in range(args.rounds):
                order_labels = list(labels)
                rng.shuffle(order_labels)
                for label in order_labels:
                    order = ["baseline", label]
                    rng.shuffle(order)
                    times = {}
                    for who in order:
                        if who == "original_pointer":
                            fn = lambda: launch(original, ts, config)
                        else:
                            ck = kernels["baseline" if who == "control" else who]
                            fn = lambda: invoke(ck, prepared)
                        times[who] = (
                            triton.testing.do_bench_cudagraph(fn, rep=20) * 1000
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
            control = statistics.median(p["speedup"] for p in samples["control"])
            for name, pairs in samples.items():
                drift = max(p["baseline_us"] for p in pairs) / min(
                    p["baseline_us"] for p in pairs
                )
                record = dict(
                    family=family,
                    name=name,
                    shape=[n, n, n],
                    pairs=pairs,
                    speedup=statistics.median(p["speedup"] for p in pairs),
                    baseline_us=statistics.median(p["baseline_us"] for p in pairs),
                    candidate_us=statistics.median(p["candidate_us"] for p in pairs),
                    baseline_drift=drift,
                    control_speedup=control,
                    valid=abs(control - 1) <= 0.02 and drift <= 1.10,
                )
                result["records"].append(record)
                print(
                    n,
                    family,
                    name,
                    round(record["speedup"], 5),
                    "AA",
                    round(control, 5),
                    "valid",
                    record["valid"],
                    flush=True,
                )
            if not args.no_wall:
                result["wall_metrics"][str(n)] = wall_metrics(
                    ts, kernels, original, config, rng
                )
            idle()
    result["summary"] = {}
    for family in result["families"]:
        names = {r["name"] for r in result["records"] if r["family"] == family}
        result["summary"][family] = {
            name: confidence(
                [
                    r
                    for r in result["records"]
                    if r["family"] == family and r["name"] == name
                ]
            )
            for name in names
        }
    result["gpu_after"] = snapshot()
    save_json(args.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["check", "sanitize", "bench"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--sizes", type=int, nargs="+")
    parser.add_argument("--names", nargs="+", default=NAMES)
    parser.add_argument("--no-wall", action="store_true")
    args = parser.parse_args()
    assert not args.output.exists()
    gate()
    idle()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.manual_seed(args.seed)
    if args.mode == "bench":
        benchmark(args)
    else:
        validate(args)


if __name__ == "__main__":
    main()
