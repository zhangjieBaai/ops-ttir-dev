"""Freeze five tutorial-sized dX baselines and verify the complete native backend."""

import argparse
import hashlib
import tempfile

import torch
import triton
from common import (
    BLOCKS,
    ROOT,
    allocate,
    compile_ir,
    digest,
    launch,
    options,
    save,
    verify_environment,
)
from triton._C.libtriton import ir
from triton.backends.nvidia.compiler import CUDABackend
from tutorial import layer_norm_bwd_dx


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    environment = verify_environment()
    torch.manual_seed(42)
    groups = {}
    for block in BLOCKS:
        ts = allocate(17, block)
        dx, dy, x, w, mean, rstd = ts[:6]
        ck = layer_norm_bwd_dx[(17,)](
            dx,
            dy,
            x,
            w,
            mean,
            rstd,
            x.stride(0),
            block,
            BLOCK_SIZE_N=block,
            **options(block),
        )
        path = ROOT / f"ir/b{block}/baseline.ttir"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            assert path.read_text() == ck.asm["ttir"]
        else:
            path.write_text(ck.asm["ttir"])
        ref = dx.clone()
        file_ck = compile_ir("baseline", block)
        for stage in ["llir", "ptx", "cubin"]:
            assert ck.asm[stage] == file_ck.asm[stage], (block, stage)
        target = triton.runtime.driver.active.get_current_target()
        backend = CUDABackend(target)
        parsed = backend.parse_options(options(block))
        assert all(
            ck.metadata._asdict()[k] == file_ck.metadata._asdict()[k]
            for k in parsed.__dict__
        )
        context = ir.context()
        ir.load_dialects(context)
        backend.load_dialects(context)
        texts = []
        for text in [ck.asm["ttgir"], file_ck.asm["ttgir"]]:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".ttgir") as f:
                f.write(text)
                f.flush()
                texts.append(str(ir.parse_mlir_module(f.name, context)))
        assert texts[0] == texts[1]
        launch(file_ck, ts)
        torch.testing.assert_close(dx, ref, rtol=0, atol=0)
        groups[str(block)] = dict(
            passed=True,
            options=ck.metadata._asdict(),
            ttir_sha256=digest(path),
            source_file_cubin_identical=True,
            source_file_output_bitwise=True,
            stage_hashes={
                s: hashlib.sha256(v if isinstance(v, bytes) else v.encode()).hexdigest()
                for s, v in ck.asm.items()
                if s in ["llir", "ptx", "cubin"]
            },
        )
    save(
        args.output,
        dict(
            passed=True,
            environment=environment,
            source_sha256=digest(ROOT / "tutorial.py"),
            groups=groups,
        ),
    )
    print(
        "PASS: all five baseline families match tutorial dX source at LLVM/PTX/cubin/output"
    )


if __name__ == "__main__":
    main()
