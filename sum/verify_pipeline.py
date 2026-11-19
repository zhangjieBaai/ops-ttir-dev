"""Verify every shape's source/file identity and capture actual toolchain provenance."""

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
    frozen,
    gate,
    kernel_info,
    launch,
    launch_source,
    load_kernels,
    receipt,
    save_json,
    sha256,
)
from environment import verify_distribution_files
from export_baseline import verify_extraction
from triton._C import libtriton
from triton._C.libtriton import ir
from triton.backends.nvidia.compiler import CUDABackend, get_ptxas


def normalized(text, backend):
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ttgir") as stream:
        stream.write(text)
        stream.flush()
        return str(ir.parse_mlir_module(stream.name, context))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    target = gate()
    env = verify_distribution_files()
    verify_extraction()
    manifest = frozen()
    backend = CUDABackend(target)
    kernels = {name: load_kernels(name) for name in ["baseline", *args.names]}
    records = []
    for n in PERF_SIZES:
        tensors = allocate(n)
        source = launch_source(tensors)
        expected_mid, expected_out = tensors[1].clone(), tensors[2].clone()
        groups = manifest["shapes"][str(n)]
        for stage, group in enumerate(groups):
            ck = kernels["baseline"][group]
            assert (
                source[stage].asm["ttir"]
                == (ROOT / manifest["groups"][group]["path"]).read_text()
            )
            same = {
                ext: source[stage].asm[ext] == ck.asm[ext]
                for ext in ("llir", "ptx", "cubin")
            }
            assert all(same.values()), (n, stage, same)
            assert normalized(source[stage].asm["ttgir"], backend) == normalized(
                ck.asm["ttgir"], backend
            )
            options = vars(backend.parse_options(OPTIONS))
            for other in [source[stage], *[kernels[name][group] for name in kernels]]:
                actual = {key: other.metadata._asdict()[key] for key in options}
                assert json.dumps(actual, default=str) == json.dumps(
                    options, default=str
                )
                assert other.metadata.target == target
            records.append(
                dict(
                    n=n,
                    stage=stage + 1,
                    group=group,
                    identical=same,
                    normalized_ttgir_identical=True,
                    entry=ck.asm["ttir"].split("tt.func public @")[1].split(" {\n")[0],
                )
            )
        tensors[1].fill_(float("nan"))
        tensors[2].fill_(float("nan"))
        launch(kernels["baseline"], tensors, manifest)
        assert torch.equal(tensors[1], expected_mid) and torch.equal(
            tensors[2], expected_out
        )
        print(f"PASS pipeline {n}", flush=True)
    torch.cuda.synchronize()
    libraries = sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if ".so" in line
        }
    )
    assert not any(
        "flagtree" in p.lower() or "flagprism" in p.lower() for p in libraries
    )
    ptxas = get_ptxas(90).path
    tools = [
        inspect.getfile(CUDABackend),
        libtriton.__file__,
        ptxas,
        ROOT / "environment.py",
        ROOT.parent / "layer_norm/environment.py",
    ]
    tools.extend(p for _, p in backend.parse_options(OPTIONS).extern_libs)
    save_json(
        args.output,
        dict(
            **receipt(),
            passed=True,
            environment=env,
            records=records,
            artifacts={
                name: {
                    g: kernel_info(
                        ck,
                        (ROOT / manifest["groups"][g]["path"]).with_name(
                            f"{name}.ttir"
                        ),
                    )
                    for g, ck in items.items()
                }
                for name, items in kernels.items()
            },
            loaded_shared_libraries=libraries,
            toolchain={str(p): sha256(p) for p in tools},
            ptxas_version=subprocess.check_output([ptxas, "--version"], text=True),
            environment_variables={
                key: value
                for key, value in os.environ.items()
                if key.startswith(("TRITON", "CUDA", "PYTHON", "LD_", "NVIDIA"))
                or key == "PATH"
            },
            bitwise_mid_and_out=True,
            input_stage="post-make_ttir; file entry begins at make_ttgir",
        ),
    )


if __name__ == "__main__":
    main()
