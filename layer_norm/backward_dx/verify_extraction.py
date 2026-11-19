"""Check dX extraction against the unmodified official fused dX+dW+dB kernel."""

import importlib.util

import torch
from common import BLOCKS, ROOT, allocate, compile_ir, launch, options, save


def main():
    path = ROOT / "artifacts/original_fused.py"
    path.write_text(
        "import triton\nimport triton.language as tl\n"
        + (ROOT / "upstream/fused_backward.py.txt").read_text()
    )
    spec = importlib.util.spec_from_file_location("original_fused_dx", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    records = []
    torch.manual_seed(42)
    for block in BLOCKS:
        n = block - 16
        ts = allocate(17, n, 16)
        dx, dy, x, w, mean, rstd = ts[:6]
        dw = torch.empty(32, n, device="cuda", dtype=x.dtype)
        db = torch.empty_like(dw)
        locks = torch.zeros(64, device="cuda", dtype=torch.int32)
        mod._layer_norm_bwd_dx_fused[(17,)](
            dx,
            dy,
            dw,
            db,
            x,
            w,
            mean,
            rstd,
            locks,
            x.stride(0),
            n,
            GROUP_SIZE_M=32,
            BLOCK_SIZE_N=block,
            **options(block)
        )
        full = dx.clone()
        launch(compile_ir("baseline", block), ts)
        torch.testing.assert_close(dx, full, rtol=0, atol=0)
        records.append(dict(block=block, N=n, M=17, padding=16, dx_bitwise_equal=True))
    save(
        ROOT / "results/extraction_check.json",
        dict(
            passed=True,
            records=records,
            scope="Only dX equivalence; not dW/dB accuracy or performance",
        ),
    )
    print("PASS: extracted dX equals original fused tutorial dX for all five families")


if __name__ == "__main__":
    main()
