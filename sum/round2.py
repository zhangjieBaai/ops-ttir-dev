"""Round2 launch-contract experiment with native TTIR compilation and bound evidence."""

import argparse
import fcntl
import json
import math
import os
import random
import statistics
import subprocess
import threading
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import torch
import triton
import triton.testing
from benchmark import interval, summarize
from common import (
    OPTIONS,
    PERF_SIZES,
    ROOT,
    TUNING,
    allocate,
    frozen,
    gate,
    geometry,
    kernel_info,
    receipt,
    save_json,
    sha256,
)
from diagnose import normalize_uuid, other_processes
from triton.backends.nvidia.compiler import CUDABackend
from validate import CANARY, check_values, patterns, prepare


@contextmanager
def isolated():
    gate()
    uuid = "GPU-" + normalize_uuid(str(torch.cuda.get_device_properties(0).uuid))
    lock = open(f"/root/tmp/ttir-device-{uuid}.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not other_processes(uuid), "Another process occupies selected GPU"
    stop, samples = threading.Event(), []

    def sample():
        while not stop.is_set():
            try:
                samples.append(
                    dict(
                        other_pids=other_processes(uuid),
                        gpu_state=subprocess.check_output(
                            [
                                "nvidia-smi",
                                "-i",
                                os.environ["CUDA_VISIBLE_DEVICES"],
                                "--query-gpu=timestamp,uuid,clocks.sm,clocks.mem,power.draw,temperature.gpu,"
                                "utilization.gpu,clocks_event_reasons.active",
                                "--format=csv,noheader",
                            ],
                            text=True,
                        ),
                    )
                )
            except Exception as error:
                samples.append(dict(error=repr(error)))
            stop.wait(1)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        yield samples
    finally:
        stop.set()
        thread.join()
        lock.close()


def manifest():
    data = json.loads((ROOT / "round2_manifest.json").read_text())
    assert data["source_manifest_sha256"] == sha256(ROOT / "baseline.json")
    assert data["protocol_sha256"] == sha256(ROOT / "optimization_plan_round2.md")
    assert list(map(int, data["routes"])) == PERF_SIZES
    return data


def route(n, name, original, mapping):
    assert str(n) in mapping["routes"], "Unmapped input"
    candidate = mapping["routes"][str(n)]
    if name == "r2_single16384" and candidate["kind"] == "single":
        return [
            dict(
                path=candidate["path"],
                sha256=candidate["sha256"],
                grid=candidate["grid"],
                runtime_indices=candidate["runtime_indices"],
                kind="single",
            )
        ]
    paths = []
    for stage, group in enumerate(original["shapes"][str(n)]):
        config = original["groups"][group]
        path = ROOT / config["path"]
        if name == "r2_tail":
            path = path.with_name("r2_tail.ttir")
        paths.append(
            dict(
                path=str(path.relative_to(ROOT)),
                sha256=sha256(path),
                runtime_indices=config["runtime_indices"],
                grid=[geometry(n)[1], 1, 1] if stage == 0 else [1, 1, 1],
                kind=f"stage{stage + 1}",
            )
        )
    return paths


def compile_path(record):
    path = ROOT / record["path"]
    assert sha256(path) == record["sha256"]
    ck = triton.compile(str(path), options=OPTIONS)
    assert ck.src.ext == "ttir" and ck.src.src == path.read_text()
    options = vars(CUDABackend(gate()).parse_options(OPTIONS))
    actual = {key: ck.metadata._asdict()[key] for key in options}
    assert json.dumps(actual, default=str) == json.dumps(options, default=str)
    assert ck.metadata.target == gate()
    ck._init_handles()
    directory = ROOT / "artifacts/round2" / ck.hash
    directory.mkdir(parents=True, exist_ok=True)
    for extension, value in ck.asm.items():
        dest = directory / f"kernel.{extension}"
        raw = value if isinstance(value, bytes) else value.encode()
        if dest.exists():
            assert dest.read_bytes() == raw
        else:
            dest.write_bytes(raw)
    return ck


def load(names, original, mapping):
    kernels, routes = {}, {}
    for name in names:
        routes[name] = {}
        for n in PERF_SIZES:
            stages = route(n, name, original, mapping)
            routes[name][str(n)] = stages
            for record in stages:
                key = record["path"]
                if key not in kernels:
                    kernels[key] = compile_path(record)
    return kernels, routes


def identities(kernels, routes):
    return {
        name: {
            n: [kernel_info(kernels[r["path"]], ROOT / r["path"]) for r in records]
            for n, records in shapes.items()
        }
        for name, shapes in routes.items()
    }


def launch(kernels, records, tensors):
    x, mid, out = tensors[:3]
    n = x.numel()
    assert x.ndim == 1 and x.is_contiguous()
    assert all(
        t.dtype == torch.float32 and t.is_cuda and t.device == x.device
        for t in (x, mid, out)
    )
    assert all(t.data_ptr() % 16 == 0 for t in (x, mid, out))
    assert len({t.untyped_storage().data_ptr() for t in (x, mid, out)}) == 3
    for record in records:
        kind = record["kind"]
        arguments = (
            (x, out, n)
            if kind == "single"
            else (x, mid, n) if kind == "stage1" else (mid, out, mid.numel())
        )
        kernels[record["path"]][tuple(record["grid"])](
            *[arguments[i] for i in record["runtime_indices"]]
        )


def stamp():
    return dict(
        **receipt(),
        round2_protocol_sha256=sha256(ROOT / "optimization_plan_round2.md"),
        round2_mapping_sha256=sha256(ROOT / "round2_manifest.json"),
        driver_sha256=sha256(__file__),
    )


def pipeline_mode(args, kernels, routes, evidence):
    old = json.loads(args.pipeline.read_text())
    assert old["passed"] and old["manifest_sha256"] == sha256(ROOT / "baseline.json")
    for n, stages in routes["baseline"].items():
        for stage, record in enumerate(stages):
            ck = kernels[record["path"]]
            group = frozen()["shapes"][n][stage]
            expected = old["artifacts"]["baseline"][group]
            assert kernel_info(ck)["stages"] == expected["stages"]
    return dict(
        **stamp(),
        passed=True,
        baseline_pipeline_sha256=sha256(args.pipeline),
        artifacts=evidence,
        routes=routes,
    )


def validate_mode(args, kernels, routes, evidence):
    torch.manual_seed(args.seed)
    rows = []
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
            native = torch.sum(x).cpu().double().reshape(1)
            block, size, _ = geometry(n)
            padded = torch.nn.functional.pad(cpu, (0, size * block - n))
            ref_mid = padded.reshape(size, block).sum(1)
            mid_l1 = padded.reshape(size, block).abs().sum(1)
            for name in args.names:
                tensors[3].fill_(CANARY)
                tensors[4].fill_(CANARY)
                tensors[1].fill_(float("nan"))
                tensors[2].fill_(float("nan"))
                stages = routes[name][str(n)]
                launch(kernels, stages, tensors)
                torch.cuda.synchronize()
                errors = check_values(tensors[2].cpu().double(), reference, l1)
                check_values(tensors[2].cpu().double(), native, l1)
                if stages[0]["kind"] == "single":
                    assert torch.isnan(
                        tensors[1]
                    ).all(), "Single kernel must leave mid untouched"
                    intermediate = "NOT_APPLICABLE; untouched mid verified"
                else:
                    intermediate = check_values(
                        tensors[1].cpu().double(), ref_mid, mid_l1
                    )
                assert (tensors[3][size:] == CANARY).all() and (
                    tensors[4][1:] == CANARY
                ).all()
                assert torch.allclose(
                    x.cpu().double(), cpu, rtol=0, atol=0, equal_nan=True
                )
                rows.append(
                    dict(
                        n=n,
                        pattern=distribution,
                        name=name,
                        errors=errors,
                        intermediate=intermediate,
                        canaries=True,
                        input_unchanged=True,
                    )
                )
            print(f"PASS {n} {distribution}", flush=True)
    return dict(**stamp(), passed=True, artifacts=evidence, routes=routes, records=rows)


def require_evidence(path, expected):
    saved = json.loads(path.read_text())
    assert saved["passed"]
    for name, shapes in json.loads(json.dumps(expected, default=str)).items():
        assert saved["artifacts"][name] == shapes, (path, name)
    return saved


def bench_mode(args, kernels, routes, evidence):
    require_evidence(args.pipeline, evidence)
    require_evidence(args.validation, evidence)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    sizes = TUNING if args.suite == "tuning" else PERF_SIZES
    result = dict(
        **stamp(),
        artifacts=evidence,
        routes=routes,
        records=[],
        controls=[],
        seed=args.seed,
        metric="complete operator CUDA graph replay device time, preallocated buffers",
        rounds=args.rounds,
        rep_ms=20,
        suite=args.suite,
        pipeline_sha256=sha256(args.pipeline),
        validation_sha256=sha256(args.validation),
    )
    candidates = [name for name in args.names if name != "baseline"]
    for n in sizes:
        tensors = allocate(n)
        calls = {
            name: partial(launch, kernels, routes[name][str(n)], tensors)
            for name in args.names
        }
        for call in calls.values():
            for _ in range(20):
                call()
            triton.testing.do_bench_cudagraph(call, rep=100)
        torch.cuda.synchronize()
        pairs = {name: [] for name in [*candidates, "A/A"]}
        for turn in range(args.rounds):
            order_candidates = list(pairs)
            rng.shuffle(order_candidates)
            for name in order_candidates:
                order = ["reference", "candidate"]
                rng.shuffle(order)
                raw, times = {}, {}
                for role in order:
                    call = calls[
                        "baseline" if name == "A/A" or role == "reference" else name
                    ]
                    replay = triton.testing.do_bench_cudagraph(
                        call, rep=20, return_mode="all"
                    )
                    assert replay and all(math.isfinite(t) and t > 0 for t in replay)
                    raw[role] = [v * 1000 for v in replay]
                    times[role] = statistics.mean(raw[role])
                pairs[name].append(
                    dict(
                        round=turn,
                        order=order,
                        replay_samples_us=raw,
                        baseline_us=times["reference"],
                        candidate_us=times["candidate"],
                        speedup=times["reference"] / times["candidate"],
                    )
                )
        control = statistics.median(p["speedup"] for p in pairs["A/A"])
        result["controls"].append(
            dict(
                n=n,
                median_speedup=control,
                passed=0.97 <= control <= 1.03,
                pairs=pairs["A/A"],
            )
        )
        for name in candidates:
            samples = pairs[name]
            row = dict(
                n=n,
                name=name,
                pairs=samples,
                median_speedup=statistics.median(p["speedup"] for p in samples),
                baseline_us=statistics.median(p["baseline_us"] for p in samples),
                candidate_us=statistics.median(p["candidate_us"] for p in samples),
                bootstrap_95=interval([p["speedup"] for p in samples], rng),
            )
            result["records"].append(row)
            print(f"n={n} {name}: {row['median_speedup']:.5f}x", flush=True)
    result["summaries"] = {
        name: summarize([r for r in result["records"] if r["name"] == name])
        for name in candidates
    }
    result["aa_passed"] = all(c["passed"] for c in result["controls"])
    print(json.dumps(result["summaries"], indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["pipeline", "validate", "bench"])
    parser.add_argument("--names", nargs="+", default=["baseline", "r2_single16384"])
    parser.add_argument("--pipeline", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=["tuning", "full"], default="full")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sanitizer", action="store_true")
    args = parser.parse_args()
    assert not args.output.exists()
    assert all(n in ["baseline", "r2_tail", "r2_single16384"] for n in args.names)
    with isolated() as telemetry:
        original, mapping = frozen(), manifest()
        kernels, routes = load(args.names, original, mapping)
        evidence = identities(kernels, routes)
        function = {
            "pipeline": pipeline_mode,
            "validate": validate_mode,
            "bench": bench_mode,
        }[args.mode]
        result = function(args, kernels, routes, evidence)
        result["telemetry"] = list(telemetry)
        result["uncontended"] = bool(telemetry) and all(
            not x.get("other_pids") and "error" not in x for x in telemetry
        )
        save_json(args.output, result)


if __name__ == "__main__":
    main()
