"""Numerical stress for approximate log on original fixed shape contract."""

import torch
from common import (
    ALL_SHAPES,
    ROOT,
    allocate,
    check_contention,
    compile_ir,
    group_for,
    ir_path,
    launch,
    receipt,
    save_json,
    sha256,
)


def main():
    check_contention()
    torch.manual_seed(823)
    records = []
    for shape in ALL_SHAPES:
        group, _ = group_for(shape)
        ck = compile_ir(group, "r5_fastlog")
        for mode in ("near_one", "uniform_wide", "near_zero"):
            x, y, storage = allocate(shape, guarded=True)
            if mode == "near_one":
                x.fill_(-20)
                x[:, 0, :] = 0
            elif mode == "uniform_wide":
                x.uniform_(-100, 100)
            else:
                x.normal_(0, 0.01)
                x.sub_(torch.logsumexp(x.double(), 1, keepdim=True).to(x.dtype))
            before = x.clone()
            ref = torch.logsumexp(x.double(), 1).to(torch.float16)
            y.fill_(float("nan"))
            launch(ck, shape, (x, y, storage))
            torch.cuda.synchronize()
            torch.testing.assert_close(y.float(), ref.float(), rtol=1e-3, atol=2e-3)
            assert torch.equal(x, before)
            assert (storage[:16] == 12345).all() and (storage[-16:] == 12345).all()
            records.append(
                dict(
                    shape=shape,
                    mode=mode,
                    max_abs=float((y.float() - ref.float()).abs().max()),
                )
            )
    save_json(
        ROOT / "results/runs/r5_setup/stress.json",
        dict(
            **receipt(),
            passed=True,
            records=records,
            hashes={
                group_for(s)[0]: sha256(ir_path(group_for(s)[0], "r5_fastlog"))
                for s in ALL_SHAPES
            }
        ),
    )
    print("PASS", len(records), "additional checks")


if __name__ == "__main__":
    main()
