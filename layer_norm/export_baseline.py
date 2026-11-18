"""Freeze tutorial-produced TTIR, and prove that TTIR compilation executes correctly."""

import argparse
import platform

import torch
import triton
from common import (
    BLOCK,
    EPS,
    OPTIONS,
    ROOT,
    allocate,
    compile_ir,
    launch,
    save_json,
    sha256,
)
from tutorial import _layer_norm_fwd_fused


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check existing frozen baseline without overwriting",
    )
    args = parser.parse_args()
    torch.manual_seed(42)
    tensors = allocate(1151, 8192)
    x, y, w, b, mean, rstd = tensors[:6]
    source = _layer_norm_fwd_fused[(1151,)](
        x, y, w, b, mean, rstd, x.stride(0), 8192, EPS, BLOCK_SIZE=BLOCK, **OPTIONS
    )
    source_y, source_mean, source_rstd = y.clone(), mean.clone(), rstd.clone()
    baseline = ROOT / "ir/baseline.ttir"
    if args.verify or baseline.exists():
        assert (
            baseline.read_text() == source.asm["ttir"]
        ), "Frozen baseline differs; never overwrite it implicitly"
    else:
        baseline.write_text(source.asm["ttir"])
    ir_kernel = compile_ir("baseline")
    launch(ir_kernel, tensors)
    torch.testing.assert_close(y, source_y, rtol=0, atol=0)
    torch.testing.assert_close(mean, source_mean, rtol=0, atol=0)
    torch.testing.assert_close(rstd, source_rstd, rtol=0, atol=0)
    ref = torch.nn.functional.layer_norm(x, (8192,), w, b, EPS)
    torch.testing.assert_close(y, ref, rtol=0, atol=1e-2)
    result = dict(
        passed=True,
        source_vs_ttir_bitwise=True,
        python=platform.python_version(),
        torch=torch.__version__,
        triton=triton.__version__,
        triton_path=triton.__file__,
        gpu=torch.cuda.get_device_name(),
        target=str(triton.runtime.driver.active.get_current_target()),
        options=OPTIONS,
        block=BLOCK,
        eps=EPS,
        dtype="float16",
        baseline_sha256=sha256(baseline),
        source_sha256=sha256(ROOT / "tutorial.py"),
        cubin_identical=source.asm["cubin"] == ir_kernel.asm["cubin"],
        y_max_error=(y - ref).abs().max().item(),
    )
    save_json(ROOT / "results/baseline_pipeline.json", result)
    print(result)


if __name__ == "__main__":
    main()
