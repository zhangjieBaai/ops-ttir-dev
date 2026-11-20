"""Compile, validate and directly pair optimized Triton source against frozen TTIR."""

import argparse
import json
import math
import random
import statistics
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402
import triton  # noqa: E402
import triton.testing  # noqa: E402
from audit_artifacts import text_sections  # noqa: E402
from benchmark import bootstrap, summary  # noqa: E402
from common import (  # noqa: E402
    ALL_SHAPES,
    PERF_SHAPES,
    allocate,
    check_contention,
    compile_ir,
    digest,
    gate,
    group_for,
    ir_path,
    key,
    launch,
    launch_source,
    manifest,
    normalized,
    receipt,
    save_json,
    sha256,
    snapshot,
)
from environment import verify_distribution_files  # noqa: E402
from export_baseline import verify_extraction  # noqa: E402
from optimized_source import (  # noqa: E402
    logsumexp_source_inner,
    logsumexp_source_noninner,
)
from triton.backends.nvidia.compiler import CUDABackend  # noqa: E402
from validate import DISTRIBUTIONS, prepare  # noqa: E402
from verify_pipeline import normalize_ttgir  # noqa: E402

ROOT = Path(__file__).resolve().parent
SELECTED = "r4_focus"


def artifact(kernel, label):
    folder = ROOT.parent / "artifacts/source_comparison" / label / kernel.hash
    folder.mkdir(parents=True, exist_ok=True)
    for stage, value in kernel.asm.items():
        raw = value if isinstance(value, bytes) else value.encode()
        path = folder / ("kernel." + stage)
        if path.exists():
            assert path.read_bytes() == raw
        else:
            path.write_bytes(raw)
    kernel._init_handles()
    return dict(
        hash=kernel.hash,
        stage_hashes={s: digest(v) for s, v in kernel.asm.items()},
        metadata=kernel.metadata._asdict(),
        registers=kernel.n_regs,
        spills=kernel.n_spills,
        text_sections=text_sections(folder / "kernel.cubin"),
        folder=str(folder),
    )


def source_kernel(shape, tensors):
    group, config = group_for(shape)
    x, y = tensors[:2]
    m, n, k = shape
    fn = logsumexp_source_inner if k == 1 else logsumexp_source_noninner
    extra = {"SINGLE_ROW": config["runtime_indices"] == [0, 1, 3]} if k == 1 else {}
    return fn[tuple(manifest()["mapping"][key(shape)]["grid"])](
        y,
        x,
        m,
        n,
        *([] if k == 1 else [k]),
        **config["meta"],
        **extra,
        **config["options"]
    )


def launch_native(ck, shape, tensors):
    if not isinstance(ck.src, triton.compiler.ASTSource):
        return launch(ck, shape, tensors)
    group, config = group_for(shape)
    x, y = tensors[:2]
    m, n, k = shape
    values = dict(
        output_ptr=y,
        input_ptr=x,
        M=m,
        N=n,
        K=k,
        **config["meta"],
        SINGLE_ROW=config["runtime_indices"] == [0, 1, 3]
    )
    # AST-source launchers retain constexpr placeholders; file-source launchers
    # expose only runtime parameters. Both launch the same device argument ABI.
    arguments = [values[name] for name in ck.src.signature]
    ck[tuple(manifest()["mapping"][key(shape)]["grid"])](*arguments)


def build_all():
    target = gate()
    check_contention()
    verify_extraction()
    backend = CUDABackend(target)
    kernels = {}
    info = {}
    points = []
    compile_times = []
    for shape in ALL_SHAPES:
        group, config = group_for(shape)
        ts = allocate(shape)
        start = time.perf_counter()
        src = source_kernel(shape, ts)
        compile_times.append((shape, time.perf_counter() - start))
        indices = [
            i for i, (_, v) in enumerate(src.src.signature.items()) if v != "constexpr"
        ]
        assert indices == config["runtime_indices"], (
            shape,
            indices,
            config["runtime_indices"],
        )
        expected = {
            k: normalized(src.metadata._asdict()[k]) for k in config["parsed_options"]
        }
        assert expected == config["parsed_options"] and src.metadata.target == target
        baseline_src = launch_source(shape, ts)
        baseline = compile_ir(group, "baseline")
        assert baseline_src.asm["ttir"] == ir_path(group, "baseline").read_text()
        assert all(
            baseline_src.asm[s] == baseline.asm[s] for s in ("llir", "ptx", "cubin")
        )
        assert normalize_ttgir(baseline_src.asm["ttgir"], backend) == normalize_ttgir(
            baseline.asm["ttgir"], backend
        )
        expected_output = ts[1].clone()
        ts[1].fill_(float("nan"))
        launch(baseline, shape, ts)
        assert torch.equal(expected_output, ts[1])
        if group not in kernels:
            ttir = compile_ir(group, SELECTED)
            entry = artifact(src, "source")
            ttir_info = artifact(ttir, "ttir")
            baseline_info = artifact(baseline, "baseline")
            path = Path(entry["folder"]) / "input.ttir"
            text = src.asm["ttir"]
            if path.exists():
                assert path.read_text() == text
            else:
                path.write_text(text)
            roundtrip = triton.compile(str(path), options=config["options"])
            roundtrip_info = artifact(roundtrip, "source_roundtrip")
            assert all(src.asm[s] == roundtrip.asm[s] for s in ("llir", "ptx", "cubin"))
            assert normalize_ttgir(src.asm["ttgir"], backend) == normalize_ttgir(
                roundtrip.asm["ttgir"], backend
            )
            assert roundtrip.metadata.target == target
            assert all(
                normalized(roundtrip.metadata._asdict()[k]) == v
                for k, v in expected.items()
            )
            info[group] = dict(
                source=entry,
                ttir=ttir_info,
                baseline=baseline_info,
                roundtrip=roundtrip_info,
                selected_ttir_sha256=sha256(ir_path(group, SELECTED)),
                source_ttir_path=str(path),
                source_file_downstream_equal=True,
                source_ttir_text_code_equal=sorted(entry["text_sections"].values())
                == sorted(ttir_info["text_sections"].values()),
            )
            kernels[group] = {"source": src, "ttir": ttir, "baseline": baseline}
        else:
            assert kernels[group]["source"].asm["ttir"] == src.asm["ttir"], (
                "shape-dependent source IR",
                shape,
            )
        points.append(
            dict(
                shape=shape,
                group=group,
                source_hash=src.hash,
                original_source_file_equal=True,
                source_options_equal=True,
            )
        )
        print("compiled", shape, group, flush=True)
    return kernels, dict(
        groups=info, points=points, compile_or_cache_wall_seconds=compile_times
    )


def validate(kernels, seed, sanitizer=False):
    torch.manual_seed(seed)
    records = []
    distributions = ("normal", "neg_inf") if sanitizer else DISTRIBUTIONS
    for shape in ALL_SHAPES:
        group, _ = group_for(shape)
        for distribution in distributions:
            ts = allocate(shape, guarded=True)
            x, y, storage = ts
            prepare(x, distribution)
            saved = x.clone()
            ref64 = torch.logsumexp(x.double(), 1).to(torch.float16).float()
            ref32 = torch.logsumexp(x.float(), 1)
            outputs = {}
            for name, ck in kernels[group].items():
                y.fill_(float("nan"))
                (
                    print("check", shape, distribution, name, flush=True)
                    if sanitizer
                    else None
                )
                launch_native(ck, shape, ts)
                torch.cuda.synchronize()
                torch.testing.assert_close(y.float(), ref64, rtol=1e-3, atol=2e-3)
                torch.testing.assert_close(y.float(), ref32, rtol=1e-3, atol=2e-3)
                assert torch.equal(torch.isneginf(y), torch.isneginf(ref64))
                assert torch.isfinite(y[torch.isfinite(ref64)]).all()
                assert torch.equal(x, saved)
                assert (storage[:16] == 12345).all() and (storage[-16:] == 12345).all()
                outputs[name] = y.clone()
                finite = torch.isfinite(ref64)
                records.append(
                    dict(
                        shape=shape,
                        group=group,
                        name=name,
                        distribution=distribution,
                        max_abs=(
                            float((y.float()[finite] - ref64[finite]).abs().max())
                            if finite.any()
                            else 0.0
                        ),
                    )
                )
            records[-1]["source_ttir_output_bitwise_equal"] = torch.equal(
                outputs["source"], outputs["ttir"]
            )
        print("accuracy", shape, flush=True)
    return records


def benchmark(kernels, rounds, seed, output):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    pairs = [
        ("baseline", "source"),
        ("baseline", "ttir"),
        ("ttir", "source"),
        ("aa0", "aa1"),
    ]
    records = []
    controls = []
    telemetry_path = output / "telemetry.csv"
    with telemetry_path.open("x") as stream:
        telemetry = subprocess.Popen(
            [
                "nvidia-smi",
                "-i",
                __import__("os").environ["CUDA_VISIBLE_DEVICES"],
                "--query-gpu=timestamp,uuid,clocks.sm,clocks.mem,temperature.gpu,power.draw,"
                "utilization.gpu,clocks_event_reasons.active",
                "--format=csv",
                "-lms",
                "200",
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        try:
            for shape in PERF_SHAPES:
                group, _ = group_for(shape)
                ts = allocate(shape)
                for ck in kernels[group].values():
                    for _ in range(20):
                        launch_native(ck, shape, ts)
                    triton.testing.do_bench_cudagraph(
                        partial(launch_native, ck, shape, ts), rep=100
                    )
                torch.cuda.synchronize()
                raw_pairs = {p: [] for p in pairs}
                for turn in range(rounds):
                    order = list(pairs)
                    rng.shuffle(order)
                    for ref, candidate in order:
                        check_contention()
                        within = [ref, candidate]
                        rng.shuffle(within)
                        samples = {}
                        for label in within:
                            ck = kernels[group][
                                "ttir" if label.startswith("aa") else label
                            ]
                            samples[label] = [
                                v * 1000
                                for v in triton.testing.do_bench_cudagraph(
                                    partial(launch_native, ck, shape, ts),
                                    rep=20,
                                    return_mode="all",
                                )
                            ]
                            assert samples[label] and all(
                                math.isfinite(v) and v > 0 for v in samples[label]
                            )
                        check_contention()
                        a = statistics.mean(samples[ref])
                        b = statistics.mean(samples[candidate])
                        raw_pairs[ref, candidate].append(
                            dict(
                                round=turn,
                                order=within,
                                samples_us=samples,
                                reference_us=a,
                                candidate_us=b,
                                speedup=a / b,
                            )
                        )
                point = []
                for (ref, candidate), rows in raw_pairs.items():
                    ratio = statistics.median(p["speedup"] for p in rows)
                    row = dict(
                        shape=shape,
                        group=group,
                        reference=ref,
                        candidate=candidate,
                        pairs=rows,
                        reference_us=statistics.median(p["reference_us"] for p in rows),
                        candidate_us=statistics.median(p["candidate_us"] for p in rows),
                        median_speedup=ratio,
                    )
                    if ref == "aa0":
                        row["passed"] = 0.97 <= ratio <= 1.03
                        controls.append(row)
                    else:
                        records.append(row)
                    point.append(row)
                save_json(output / "points" / (key(shape) + ".json"), point)
                print(
                    "timed",
                    shape,
                    [
                        (r["reference"], r["candidate"], round(r["median_speedup"], 5))
                        for r in point
                    ],
                    flush=True,
                )
        finally:
            telemetry.terminate()
            telemetry.wait()
    summaries = {}
    for ref, candidate in pairs[:-1]:
        rows = [
            r for r in records if r["reference"] == ref and r["candidate"] == candidate
        ]
        summaries[ref + "/" + candidate] = dict(
            **summary(rows), bootstrap_95=bootstrap(rows)
        )
    return dict(
        records=records,
        controls=controls,
        aa_passed=all(r["passed"] for r in controls),
        summaries=summaries,
        telemetry_sha256=sha256(telemetry_path),
        rounds=rounds,
        rep_ms=20,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--sanitizer", action="store_true")
    p.add_argument("--accuracy-only", action="store_true")
    args = p.parse_args()
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    gate()
    environment = verify_distribution_files()
    check_contention()
    inputs = dict(
        comparison_protocol_sha256=sha256(ROOT / "protocol.md"),
        source_sha256=sha256(ROOT / "optimized_source.py"),
        runner_sha256=sha256(__file__),
        baseline_manifest_sha256=sha256(ROOT.parent / "baseline.json"),
        selected=SELECTED,
        environment=environment,
    )
    kernels, pipeline = build_all()
    save_json(
        args.output / "pipeline.json",
        dict(passed=True, **receipt(), **inputs, **pipeline),
    )
    accuracy = validate(kernels, args.seed, args.sanitizer)
    save_json(
        args.output / "accuracy.json",
        dict(passed=True, seed=args.seed, records=accuracy, **inputs),
    )
    if not args.sanitizer and not args.accuracy_only:
        before = snapshot()
        timing = benchmark(kernels, args.rounds, args.seed, args.output)
        save_json(
            args.output / "benchmark.json",
            dict(
                **receipt(),
                **inputs,
                **timing,
                gpu_before=before,
                gpu_after=snapshot(),
                contention_after=check_contention()
            ),
        )
        print(
            json.dumps(timing["summaries"], indent=2),
            "AA",
            timing["aa_passed"],
            flush=True,
        )


if __name__ == "__main__":
    main()
