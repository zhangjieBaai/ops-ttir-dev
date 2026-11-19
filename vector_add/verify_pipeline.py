"""Audit upstream source -> frozen TTIR -> native Triton downstream stages."""

import argparse
import inspect
import json
import os
import subprocess
import tempfile
from pathlib import Path

import torch
from common import (
    OPTIONS,
    PERF_SIZES,
    ROOT,
    allocate,
    compile_ir,
    gate,
    kernel_info,
    launch,
    launch_source,
    manifest,
    receipt,
    save_json,
    sha256,
)
from environment import verify_distribution_files
from export_baseline import verify_extraction
from triton._C import libtriton
from triton._C.libtriton import ir
from triton.backends.nvidia.compiler import CUDABackend, get_ptxas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--names", nargs="*", default=[])
    args = parser.parse_args()
    assert not args.output.exists()
    target = gate()
    env = verify_distribution_files()
    verify_extraction()
    manifest()
    backend = CUDABackend(target)
    options = backend.parse_options(OPTIONS)
    baseline = compile_ir()
    rows = []
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    for n in PERF_SIZES + [16, 1008, 1040, 98432]:
        x, y, out, _ = allocate(n)
        source = launch_source(x, y, out)
        source._init_handles()
        assert source.asm["ttir"] == (ROOT / "ir/baseline.ttir").read_text()
        expected = out.clone()
        out.fill_(float("nan"))
        launch(baseline, x, y, out)
        assert torch.equal(expected.view(torch.int32), out.view(torch.int32))
        identical = {
            s: source.asm[s] == baseline.asm[s] for s in ("llir", "ptx", "cubin")
        }
        assert all(identical.values()), identical
        normalized = []
        for ck in (source, baseline):
            assert ck.metadata.target == target
            assert all(
                json.dumps(ck.metadata._asdict()[k]) == json.dumps(v)
                for k, v in vars(options).items()
            )
            with tempfile.NamedTemporaryFile(mode="w", suffix=".ttgir") as tmp:
                tmp.write(ck.asm["ttgir"])
                tmp.flush()
                normalized.append(str(ir.parse_mlir_module(tmp.name, context)))
        assert normalized[0] == normalized[1]
        rows.append(
            dict(
                n=n,
                exact_stages=identical,
                normalized_ttgir_identical=True,
                output_bitwise=True,
                source=kernel_info(source, "tutorial"),
            )
        )
        print(f"PASS pipeline N={n}", flush=True)
        del x, y, out, expected
    candidates = {name: kernel_info(compile_ir(name), name) for name in args.names}
    torch.cuda.synchronize()
    libraries = sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if ".so" in line
        }
    )
    components = [
        p for p in libraries if any(s in p for s in ("libtorch", "libc10", "libcudart"))
    ]
    assert all(
        p.startswith(str(Path(torch.__file__).parents[1])) for p in components
    ), components
    assert not any(
        "flagtree" in p.lower() or "flagprism" in p.lower() for p in libraries
    )
    ptxas = get_ptxas(90).path
    files = [
        inspect.getfile(CUDABackend),
        libtriton.__file__,
        ptxas,
        ROOT / "environment.py",
        ROOT.parent / "layer_norm/environment.py",
    ]
    files += [p for _, p in options.extern_libs]
    save_json(
        args.output,
        dict(
            passed=True,
            **receipt(),
            environment=env,
            torch=torch.__version__,
            target=str(target),
            actual_options=vars(options),
            shapes=rows,
            baseline=kernel_info(baseline, "baseline"),
            candidates=candidates,
            input_ttir_sha256=sha256(ROOT / "ir/baseline.ttir"),
            input_stage="after make_ttir; file entry resumes at make_ttgir",
            toolchain={str(p): sha256(p) for p in files},
            ptxas_version=subprocess.check_output([ptxas, "--version"], text=True),
            env_vars={
                k: v
                for k, v in os.environ.items()
                if any(
                    s in k
                    for s in ("TRITON", "CUDA", "PYTHON", "LD_LIBRARY", "PATH", "OMP")
                )
            },
            loaded_libraries=libraries,
            torch_components=components,
            clocks="not locked by this experiment; recorded snapshots only",
            cache="no explicit clearing; repeated graph workload",
        ),
    )


if __name__ == "__main__":
    main()
