"""Separate accuracy group for runtime guard boundaries omitted from perf matrix."""

import argparse
from pathlib import Path

import torch
import triton
from ablation_common import NAMES
from common import (
    ROOT,
    allocate,
    check_contention,
    compile_ir,
    digest,
    gate,
    ir_path,
    kernel_info,
    launch_source,
    manifest,
    normalized,
    original_config,
    receipt,
    save_json,
    sha256,
)
from triton.backends.nvidia.compiler import CUDABackend
from triton_syntax_ablation import public_mean
from validate import DISTRIBUTIONS, prepare
from verify_triton_syntax import config as syntax_config

PROBES = [(3, n, k) for n in (2, 7, 8, 9) for k in (1024, 1025)]


def probe_launch(ck, shape, tensors, config):
    m, n, k = shape
    x, y = tensors[:2]
    assert (
        x.is_contiguous() and y.is_contiguous() and x.dtype == y.dtype == torch.float16
    )
    assert x.data_ptr() % 16 == y.data_ptr() % 16 == 0
    assert x.untyped_storage().data_ptr() != y.untyped_storage().data_ptr()
    args = (y, x, m, n, k)
    ck[(m, triton.cdiv(k, 1024), 1)](*[args[i] for i in config["runtime_indices"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = gate()
    check_contention()
    torch.manual_seed(1239)
    mappings = []
    records = []
    identities = {}
    for shape in PROBES:
        m, n, k = shape
        tensors = allocate(shape, guarded=True)
        source = launch_source(shape, tensors)
        reference_output = tensors[1].clone()
        _, options = original_config(shape)
        parsed = normalized(vars(CUDABackend(target).parse_options(options)))
        matches = [
            (group, c)
            for group, c in manifest()["groups"].items()
            if ir_path(group, "baseline").read_text() == source.asm["ttir"]
            and c["parsed_options"] == parsed
        ]
        assert len(matches) == 1, (shape, len(matches))
        group, config = matches[0]
        kernels = {name: compile_ir(group, name) for name in NAMES}
        baseline = kernels["baseline"]
        assert all(
            source.asm[stage] == baseline.asm[stage]
            for stage in ("llir", "ptx", "cubin")
        )
        tensors[1].fill_(float("nan"))
        probe_launch(baseline, shape, tensors, config)
        assert torch.equal(tensors[1], reference_output)
        mappings.append(
            dict(
                shape=shape,
                group=group,
                options=parsed,
                grid=[m, triton.cdiv(k, 1024), 1],
                source_ttir_sha256=digest(source.asm["ttir"]),
                source_cubin_sha256=digest(source.asm["cubin"]),
                source_file_byte_equal=True,
                source_file_output_bitwise_equal=True,
            )
        )
        identities[group] = {
            name: kernel_info(ck, group, name) for name, ck in kernels.items()
        }
        for distribution in DISTRIBUTIONS:
            tensors = allocate(shape, guarded=True)
            x, y, storage = tensors
            prepare(x, distribution)
            saved = x.clone()
            reference = torch.mean(x.cpu().double(), 1).half().float().cuda()
            native = x.mean(1).float()
            for name, ck in kernels.items():
                for route in ("ttir", "public_triton"):
                    y.fill_(float("nan"))
                    if route == "ttir":
                        probe_launch(ck, shape, tensors, config)
                    else:
                        public_mean[(m, triton.cdiv(k, 1024), 1)](
                            y, x, m, n, k, **syntax_config(name, n), **options
                        )
                    torch.cuda.synchronize()
                    torch.testing.assert_close(
                        y.float(), reference, rtol=2e-3, atol=2e-3, equal_nan=True
                    )
                    torch.testing.assert_close(
                        y.float(), native, rtol=2e-3, atol=2e-3, equal_nan=True
                    )
                    assert torch.isfinite(y[torch.isfinite(reference)]).all()
                    assert (storage[:16] == 12345).all() and (
                        storage[-16:] == 12345
                    ).all()
                    torch.testing.assert_close(x, saved, rtol=0, atol=0, equal_nan=True)
                    records.append(
                        dict(
                            shape=shape,
                            group=group,
                            name=name,
                            route=route,
                            distribution=distribution,
                            passed=True,
                        )
                    )
        print("PASS runtime branch probes", shape, group, flush=True)
    save_json(
        args.output,
        dict(
            **receipt(),
            passed=True,
            supplement_sha256=sha256(ROOT / "ablation_branch_probe_plan.md"),
            plan_sha256=sha256(ROOT / "ablation_plan.md"),
            mappings=mappings,
            kernels=identities,
            records=records
        ),
    )


if __name__ == "__main__":
    main()
