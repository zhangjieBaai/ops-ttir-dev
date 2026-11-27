"""Require byte-identical source and frozen-file downstream compilation."""

import argparse
import inspect

import torch
from common import (
    ROOT,
    allocate,
    compile_ir,
    environment,
    launch,
    manifest,
    record,
    save,
    sha,
    source_launch,
)
from triton._C import libtriton
from triton.backends.nvidia.compiler import CUDABackend, get_ptxas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate")
    args = parser.parse_args()
    env = environment()
    results = []
    for case in manifest()["cases"]:
        ts = allocate(case["shape"])
        source = source_launch(ts, case["config"])
        original = ts[4].clone()
        baseline = compile_ir(case, "baseline")
        assert (
            source.asm["ttir"]
            == (ROOT / "ir" / case["id"] / "baseline.ttir").read_text()
        )
        assert all(source.asm[s] == baseline.asm[s] for s in ("llir", "ptx", "cubin"))
        source_options = CUDABackend(source.metadata.target).parse_options({}).__dict__
        keys = list(source_options)
        assert {k: getattr(source.metadata, k) for k in keys} == {
            k: getattr(baseline.metadata, k) for k in keys
        }
        launch(baseline, ts, case["config"])
        torch.testing.assert_close(ts[4], original, rtol=0, atol=0)
        candidate_record = None
        if args.candidate:
            candidate = compile_ir(case, args.candidate)
            assert {k: getattr(source.metadata, k) for k in keys} == {
                k: getattr(candidate.metadata, k) for k in keys
            }
            launch(candidate, ts, case["config"])
            torch.testing.assert_close(ts[4], original, rtol=0, atol=0)
            candidate_record = record(
                candidate, ROOT / "ir" / case["id"] / (args.candidate + ".ttir")
            )
        results.append(
            dict(
                id=case["id"],
                passed=True,
                llir_ptx_cubin_byte_identical=True,
                options_identical=True,
                outputs_bitwise_equal=True,
                baseline=record(baseline, ROOT / "ir" / case["id"] / "baseline.ttir"),
                candidate=candidate_record,
            )
        )
    save(
        args.output,
        dict(
            passed=True,
            environment=env,
            cases=results,
            toolchain=dict(
                backend_sha256=sha(inspect.getfile(CUDABackend)),
                libtriton_sha256=sha(libtriton.__file__),
                ptxas_sha256=sha(get_ptxas(90).path),
            ),
        ),
    )
    print("PASS: all cases source/file LLIR, PTX and cubin byte-identical", flush=True)


if __name__ == "__main__":
    main()
