"""Freeze original row TTIR and check the source/native-file pipeline at all 30 shapes."""

import argparse
import inspect
import tempfile

import torch
import triton
from common import (
    CONTRACT,
    OPTIONS,
    ROOT,
    SHAPES,
    allocate,
    compile_ir,
    digest,
    environment,
    identity,
    launch,
    provenance,
    save,
)
from triton._C import libtriton
from triton._C.libtriton import ir
from triton.backends.nvidia.compiler import CUDABackend, get_ptxas
from tutorial import var_mean_welford_kernel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--candidate")
    args = p.parse_args()
    env = environment.verify_distribution_files()
    backend = CUDABackend(triton.runtime.driver.active.get_current_target())
    parsed = backend.parse_options(OPTIONS).__dict__
    records = []
    file_ck = None
    for m, n in SHAPES:
        ts = allocate(m, n)
        x, var, mean = ts[:3]
        ck = var_mean_welford_kernel[(m,)](
            x, var, mean, m, n, 1.0, BLOCK_N=1024, **OPTIONS
        )
        path = ROOT / "ir/baseline.ttir"
        if not path.exists():
            path.write_text(ck.asm["ttir"])
        assert path.read_text() == ck.asm["ttir"], (m, n, "TTIR mapping mismatch")
        if file_ck is None:
            file_ck = compile_ir("baseline")
        for stage in ["llir", "ptx", "cubin"]:
            assert ck.asm[stage] == file_ck.asm[stage], (m, n, stage)
        assert all(
            ck.metadata._asdict()[k] == file_ck.metadata._asdict()[k] for k in parsed
        )
        context = ir.context()
        ir.load_dialects(context)
        backend.load_dialects(context)
        normalized = []
        for text in [ck.asm["ttgir"], file_ck.asm["ttgir"]]:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".ttgir") as f:
                f.write(text)
                f.flush()
                normalized.append(str(ir.parse_mlir_module(f.name, context)))
        assert normalized[0] == normalized[1]
        expected = [var.clone(), mean.clone()]
        var.fill_(float("nan"))
        mean.fill_(float("nan"))
        launch(file_ck, ts)
        for actual, ref in zip([var, mean], expected):
            torch.testing.assert_close(actual, ref, rtol=0, atol=0)
        records.append(
            dict(
                shape=[m, n],
                group="row_f32_b1024_w4",
                source_ttir_sha256=digest(path),
                passed=True,
            )
        )
    candidate = None
    if args.candidate:
        opt = compile_ir(args.candidate)
        assert all(
            opt.metadata._asdict()[k] == file_ck.metadata._asdict()[k] for k in parsed
        )
        candidate = identity(args.candidate, opt)
    save(
        args.output,
        dict(
            passed=True,
            provenance=provenance(),
            environment=env,
            source_sha256=digest(ROOT / "tutorial.py"),
            source_identity=digest(ROOT / "upstream/var_mean.py.snapshot"),
            toolchain={
                "backend": digest(inspect.getfile(CUDABackend)),
                "libtriton": digest(libtriton.__file__),
                "ptxas": digest(get_ptxas(90).path),
            },
            baseline=identity("baseline", file_ck),
            candidate=candidate,
            records=records,
        ),
    )
    manifest = ROOT / "baseline.json"
    if not manifest.exists():
        save(
            manifest,
            dict(
                contract_sha256=digest(ROOT / "contract.json"),
                source_sha256=digest(ROOT / "tutorial.py"),
                options=parsed,
                mapping=records,
                baseline=identity("baseline", file_ck),
                scope=CONTRACT["version"],
            ),
        )
    print(
        "PASS: 30 source/file mappings, LLVM/PTX/cubin bytes, normalized TTGIR and both outputs"
    )


if __name__ == "__main__":
    main()
