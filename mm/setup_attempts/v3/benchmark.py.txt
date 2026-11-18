"""Baseline-only CUDA graph kernel timing; correctness and hash gated."""

import argparse
import json
import statistics
from pathlib import Path

import torch
import triton.testing
from common import (
    PERF_SHAPES,
    ROOT,
    allocate,
    compile_ir,
    dump_kernel,
    frozen,
    launch,
    save_json,
    sha256,
)
from verify_pipeline import snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    assert not args.output.exists()
    config = frozen()
    validation = json.loads(args.validation.read_text())
    pipeline = json.loads(args.pipeline.read_text())
    for record in [validation, pipeline]:
        assert record["passed"] and record["baseline_sha256"] == config["ttir_sha256"]
        assert record["config_sha256"] == sha256(ROOT / "baseline.json")
    assert validation["suite"] == "full"
    assert validation["validator_sha256"] == sha256(ROOT / "validate.py")
    assert validation["pipeline_sha256"] == sha256(args.pipeline)
    device_uuid = str(torch.cuda.get_device_properties(0).uuid)
    assert validation["device_uuid"] == pipeline["torch_device_uuid"] == device_uuid
    torch.manual_seed(args.seed)
    ck = compile_ir()
    info = dump_kernel(ck, "baseline")
    assert info["stages"]["cubin"] == pipeline["baseline"]["stages"]["cubin"]
    result = dict(
        passed=True,
        timer="triton.testing.do_bench_cudagraph; kernel only; preallocated output",
        seed=args.seed,
        rounds=5,
        rep_ms=20,
        records=[],
        baseline_sha256=config["ttir_sha256"],
        config_sha256=sha256(ROOT / "baseline.json"),
        validation_sha256=sha256(args.validation),
        pipeline_sha256=sha256(args.pipeline),
        benchmark_sha256=sha256(__file__),
        resources=info,
        device_uuid=device_uuid,
        gpu_before=snapshot(),
        clock_lock="not set or verified",
        cache="no explicit flush; graph repeated-input cache state",
    )
    for m, n, k in PERF_SHAPES:
        ts = allocate(m, n, k)
        for _ in range(10):
            launch(ck, ts, config)
        torch.cuda.synchronize()
        samples = [
            triton.testing.do_bench_cudagraph(lambda: launch(ck, ts, config), rep=20)
            * 1000
            for _ in range(5)
        ]
        median = statistics.median(samples)
        row = dict(
            shape=[m, n, k],
            samples_us=samples,
            median_us=median,
            min_us=min(samples),
            max_us=max(samples),
            tflops=2 * m * n * k / median / 1e6,
        )
        result["records"].append(row)
        print(
            f"{m:4}x{n:4}x{k:4}: {median:9.3f} us, {row['tflops']:8.2f} TFLOPS",
            flush=True,
        )
    result["gpu_after"] = snapshot()
    save_json(args.output, result)


if __name__ == "__main__":
    main()
