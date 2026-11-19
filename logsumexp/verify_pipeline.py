"""Per-point source/TTIR pipeline equivalence and environment attestation."""

import argparse
import inspect
import os
import tempfile
from pathlib import Path

import torch
from common import (
    ALL_SHAPES,
    allocate,
    check_contention,
    compile_ir,
    digest,
    gate,
    group_for,
    ir_path,
    kernel_info,
    key,
    launch,
    launch_source,
    manifest,
    normalized,
    receipt,
    save_json,
    sha256,
)
from environment import verify_distribution_files
from export_baseline import verify_extraction
from triton._C import libtriton
from triton._C.libtriton import ir
from triton.backends.nvidia.compiler import CUDABackend, get_ptxas


def normalize_ttgir(text, backend):
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ttgir") as stream:
        stream.write(text)
        stream.flush()
        return str(ir.parse_mlir_module(stream.name, context))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = gate()
    check_contention()
    environment = verify_distribution_files()
    verify_extraction()
    backend = CUDABackend(target)
    names = list(dict.fromkeys(["baseline", *args.names]))
    result = dict(
        **receipt(), environment=environment, passed=False, points=[], kernels={}
    )
    for shape in ALL_SHAPES:
        group, config = group_for(shape)
        tensors = allocate(shape)
        source = launch_source(shape, tensors)
        source._init_handles()
        expected = tensors[1].clone()
        assert source.asm["ttir"] == ir_path(group, "baseline").read_text()
        baseline = compile_ir(group)
        options = config["parsed_options"]
        assert normalized({k: source.metadata._asdict()[k] for k in options}) == options
        assert baseline.metadata.target == source.metadata.target
        assert all(
            source.asm[stage] == baseline.asm[stage]
            for stage in ("llir", "ptx", "cubin")
        )
        assert normalize_ttgir(source.asm["ttgir"], backend) == normalize_ttgir(
            baseline.asm["ttgir"], backend
        )
        tensors[1].fill_(float("nan"))
        launch(baseline, shape, tensors)
        assert torch.equal(expected, tensors[1])
        torch.testing.assert_close(
            tensors[1].float(),
            torch.logsumexp(tensors[0].float(), dim=1),
            rtol=1e-3,
            atol=2e-3,
        )
        result["points"].append(
            dict(
                shape=shape,
                group=group,
                source_file_ttir_equal=True,
                llir_ptx_cubin_byte_equal=True,
                ttgir_parser_equal=True,
                output_bitwise_equal=True,
                source_cubin_sha256=digest(source.asm["cubin"]),
            )
        )
        if group not in result["kernels"]:
            result["kernels"][group] = {}
            for name in names:
                ck = compile_ir(group, name)
                result["kernels"][group][name] = kernel_info(ck, group, name)
        print("PASS pipeline", key(shape), group, flush=True)
    environment["torch_version"] = torch.__version__
    environment["torch_path"] = torch.__file__
    environment["venv_config"] = Path("/root/venvs/ir-ops-dev/pyvenv.cfg").read_text()
    environment["relevant_environment"] = {
        k: v
        for k, v in os.environ.items()
        if any(s in k for s in ("TRITON", "CUDA", "PYTHON", "LD_", "FLAG", "NVIDIA"))
    }
    environment["loaded_libraries"] = sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "/" in line
            and any(s in line for s in ("triton", "cuda", "nvidia", "torch"))
        }
    )
    result["toolchain"] = dict(
        backend_path=inspect.getfile(CUDABackend),
        backend_sha256=sha256(inspect.getfile(CUDABackend)),
        libtriton_sha256=sha256(libtriton.__file__),
        ptxas_path=get_ptxas(90).path,
        ptxas_sha256=sha256(get_ptxas(90).path),
        libdevice_sha256=sha256(
            dict(
                next(iter(manifest()["groups"].values()))["parsed_options"][
                    "extern_libs"
                ]
            )["libdevice"]
        ),
    )
    result["file_entry"] = (
        "post make_ttir, resumes at make_ttgir; native runtime JIT; cache hits allowed"
    )
    result["contention_after"] = check_contention()
    result["passed"] = True
    save_json(args.output, result)
    print("PASS pipeline all points")


if __name__ == "__main__":
    main()
