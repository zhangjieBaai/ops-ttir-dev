"""Validate and measure frozen sum ablation edges, retaining raw paired evidence."""

import argparse
import json
import math
import random
import statistics
from functools import partial
from pathlib import Path

import torch
import triton.testing
from ablation_common import (
    BASE,
    EDGES,
    PUBLIC,
    TIMED,
    VALIDATED,
    identities,
    isolated,
    launch,
    load,
    mapping,
)
from benchmark import interval, summarize
from common import PERF_SIZES, ROOT, allocate, geometry, receipt, save_json, sha256
from validate import CANARY, check_values, patterns, prepare


def stamp():
    return dict(
        **receipt(),
        plan_sha256=sha256(ROOT / "ablation_plan.md"),
        ablation_manifest_sha256=sha256(BASE / "manifest.json"),
        driver_sha256=sha256(__file__),
        common_sha256=sha256(ROOT / "ablation_common.py"),
        triton_source_sha256=sha256(ROOT / "ablation_triton.py"),
    )


def plain(data):
    return json.loads(json.dumps(data, default=str))


def bind(path, evidence):
    data = json.loads(path.read_text())
    assert data["passed"] and data["uncontended"]
    assert data["plan_sha256"] == sha256(ROOT / "ablation_plan.md")
    assert data["ablation_manifest_sha256"] == sha256(BASE / "manifest.json")
    for name, records in plain(evidence).items():
        assert data["artifacts"][name] == records, (path, name)
    return data


def pipeline(data, evidence):
    source_pipeline = BASE / "results/baseline_pipeline_gpu3.json"
    original = json.loads(source_pipeline.read_text())
    assert original["passed"]
    frozen = json.loads((ROOT / "baseline.json").read_text())
    for n, records in evidence["baseline"].items():
        for index, record in enumerate(records):
            group = frozen["shapes"][n][index]
            assert (
                record["stages"] == original["artifacts"]["baseline"][group]["stages"]
            )
    return dict(
        **stamp(),
        passed=True,
        artifacts=evidence,
        routes=data["routes"],
        source_file_pipeline_sha256=sha256(source_pipeline),
        public_checks=data["checks"],
        public_failures=data["public_failures"],
    )


def validate(args, data, kernels, evidence):
    torch.manual_seed(args.seed)
    rows, equivalent = [], []
    for n in PERF_SIZES:
        distributions = ["random"] if args.sanitizer else patterns(n)
        if n <= 16384:
            distributions = list(
                dict.fromkeys(
                    [
                        *distributions,
                        "zero",
                        "constant",
                        "alternating",
                        "offset",
                        "tiny",
                        "impulse",
                        "nan",
                        "posinf",
                        "neginf",
                        *(["mixed_inf"] if n > 1 else []),
                    ]
                )
            )
        for distribution in distributions:
            tensors = allocate(n)
            x = tensors[0]
            prepare(x, distribution)
            cpu = x.cpu().double()
            reference, l1 = cpu.sum().reshape(1), cpu.abs().sum().reshape(1)
            native = x.sum().cpu().double().reshape(1)
            block, size, _ = geometry(n)
            padded = torch.nn.functional.pad(cpu, (0, size * block - n))
            mid_reference = padded.reshape(size, block).sum(1)
            mid_l1 = padded.reshape(size, block).abs().sum(1)
            saved = {}
            for name in args.names:
                tensors[3].fill_(CANARY)
                tensors[4].fill_(CANARY)
                tensors[1].fill_(float("nan"))
                tensors[2].fill_(float("nan"))
                records = data["routes"][name][str(n)]
                launch(kernels, records, tensors)
                torch.cuda.synchronize()
                output, middle = tensors[2].cpu().double(), tensors[1].cpu().double()
                errors = check_values(output, reference, l1)
                native_errors = check_values(output, native, l1)
                if records[0]["kind"] == "single":
                    assert torch.isnan(middle).all()
                    intermediate = "untouched"
                elif records[0]["kind"] == "wide_mid":
                    intermediate = check_values(middle[:1], reference, l1)
                    assert torch.isnan(middle[1:]).all()
                else:
                    intermediate = check_values(middle, mid_reference, mid_l1)
                assert (tensors[3][size:] == CANARY).all() and (
                    tensors[4][1:] == CANARY
                ).all()
                assert torch.allclose(
                    x.cpu().double(), cpu, rtol=0, atol=0, equal_nan=True
                )
                saved[name] = (output, middle)
                rows.append(
                    dict(
                        n=n,
                        pattern=distribution,
                        name=name,
                        output=errors,
                        native=native_errors,
                        intermediate=intermediate,
                        canary=True,
                        input_unchanged=True,
                    )
                )
            for name in PUBLIC:
                if name in saved and "py_" + name in saved:
                    same = [
                        torch.allclose(a, b, rtol=0, atol=0, equal_nan=True)
                        for a, b in zip(saved[name], saved["py_" + name])
                    ]
                    # Different legal reduction ordering is allowed, but never hidden as bitwise equality.
                    equivalent.append(
                        dict(
                            n=n,
                            pattern=distribution,
                            name=name,
                            output_exact=same[0],
                            intermediate_exact=same[1],
                        )
                    )
            print(f"PASS {n} {distribution}", flush=True)
    return dict(
        **stamp(),
        passed=True,
        artifacts=evidence,
        records=rows,
        public_vs_direct=equivalent,
    )


def benchmark(args, data, kernels, evidence):
    bind(args.pipeline, evidence)
    bind(args.validation, evidence)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    result = dict(
        **stamp(),
        artifacts=evidence,
        records=[],
        controls=[],
        edges=EDGES,
        rounds=args.rounds,
        rep_ms=20,
        seed=args.seed,
        pipeline_sha256=sha256(args.pipeline),
        validation_sha256=sha256(args.validation),
        metric="complete operator graph replay, preallocated buffers, no cache clearing or clock locking",
    )
    for n in PERF_SIZES:
        tensors = allocate(n)
        calls = {
            name: partial(launch, kernels, data["routes"][name][str(n)], tensors)
            for name in TIMED
        }
        for call in calls.values():
            for _ in range(20):
                call()
            triton.testing.do_bench_cudagraph(call, rep=100)
        torch.cuda.synchronize()
        labels = [(left, right, left + "->" + right) for left, right in EDGES]
        labels.append(("baseline", "baseline", "A/A"))
        pairs = {label: [] for _, _, label in labels}
        for turn in range(args.rounds):
            rng.shuffle(labels)
            for left, right, label in labels:
                order = ["reference", "candidate"]
                rng.shuffle(order)
                raw, times = {}, {}
                for role in order:
                    call = calls[left if role == "reference" else right]
                    replay = triton.testing.do_bench_cudagraph(
                        call, rep=20, return_mode="all"
                    )
                    assert replay and all(math.isfinite(v) and v > 0 for v in replay)
                    raw[role] = [value * 1000 for value in replay]
                    times[role] = statistics.mean(raw[role])
                pairs[label].append(
                    dict(
                        round=turn,
                        order=order,
                        baseline_us=times["reference"],
                        candidate_us=times["candidate"],
                        speedup=times["reference"] / times["candidate"],
                        replay_samples_us=raw,
                    )
                )
        control = statistics.median(p["speedup"] for p in pairs["A/A"])
        result["controls"].append(
            dict(
                n=n,
                passed=0.97 <= control <= 1.03,
                median_speedup=control,
                pairs=pairs["A/A"],
            )
        )
        for left, right in EDGES:
            label = left + "->" + right
            samples = pairs[label]
            row = dict(
                n=n,
                name=label,
                reference=left,
                candidate=right,
                pairs=samples,
                median_speedup=statistics.median(p["speedup"] for p in samples),
                baseline_us=statistics.median(p["baseline_us"] for p in samples),
                candidate_us=statistics.median(p["candidate_us"] for p in samples),
                bootstrap_95=interval([p["speedup"] for p in samples], rng),
            )
            result["records"].append(row)
            print(f"n={n} {label}: {row['median_speedup']:.6f}x", flush=True)
        # Checkpoint is complete per shape, preserving partial data if a later operation fails.
        save_json(
            args.output.with_name(f"{args.output.stem}_n{n}.json"),
            dict(
                n=n,
                records=[r for r in result["records"] if r["n"] == n],
                control=result["controls"][-1],
            ),
        )
    result["summaries"] = {}
    for left, right in EDGES:
        label = left + "->" + right
        rows = [r for r in result["records"] if r["name"] == label]
        result["summaries"][label] = dict(
            full=summarize(rows),
            small=summarize([r for r in rows if r["n"] <= 16384]),
            large=summarize([r for r in rows if r["n"] > 16384]),
        )
    result["aa_passed"] = all(row["passed"] for row in result["controls"])
    print(
        json.dumps(
            {name: value["full"] for name, value in result["summaries"].items()},
            indent=2,
        ),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["pipeline", "validate", "bench"])
    parser.add_argument("--names", nargs="+", default=VALIDATED)
    parser.add_argument("--pipeline", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sanitizer", action="store_true")
    args = parser.parse_args()
    assert not args.output.exists() and args.rounds > 0
    names = TIMED if args.mode == "bench" else args.names
    assert all(name in VALIDATED for name in names)
    with isolated() as telemetry:
        data = mapping()
        kernels = load(names, data)
        evidence = identities(names, data, kernels)
        if args.mode == "pipeline":
            result = pipeline(data, evidence)
        elif args.mode == "validate":
            result = validate(args, data, kernels, evidence)
        else:
            result = benchmark(args, data, kernels, evidence)
        result["telemetry"] = list(telemetry)
        result["uncontended"] = bool(telemetry) and all(
            not s.get("other_pids") and "error" not in s for s in telemetry
        )
        save_json(args.output, result)


if __name__ == "__main__":
    main()
