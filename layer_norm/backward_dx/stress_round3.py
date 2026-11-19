"""Additional cancellation-sensitive checks without relaxing existing dX tolerances."""

import argparse
import os

import torch
from common import (
    ROOT,
    allocate,
    block_for,
    compile_ir,
    digest,
    launch,
    reference,
    save,
)
from validate import check_output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("names", nargs="+")
    p.add_argument("--seed", type=int, default=47)
    p.add_argument("--output", required=True)
    p.add_argument("--saved-stats", action="store_true")
    args = p.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    assert not os.path.exists(args.output)
    torch.manual_seed(args.seed)
    kernels = {}
    out = dict(
        passed=True,
        seed=args.seed,
        saved_stats=args.saved_stats,
        records=[],
        kernels={},
    )
    for n in [1024, 1536, 3072, 8192, 8704, 15872]:
        for offset in [32.0, 256.0, 4096.0]:
            for dist in ["noise", "outlier", "correlated", "alternating"]:
                ts = allocate(17, n, 16)
                dx, dy, x, w, mean, rstd, storage = ts
                x.fill_(offset)
                if dist == "noise":
                    x.add_((torch.randn_like(x.float()) * 0.02).to(x.dtype))
                else:
                    x[:, 0] = offset + max(offset / 1024, 0.125)
                if dist == "correlated":
                    dy.copy_(x - offset)
                    w.fill_(1.0)
                elif dist == "alternating":
                    dy.fill_(1.0)
                    w.fill_(1.0)
                    w[1::2] = -1.0
                xf = x.double()
                mean.copy_(xf.mean(-1).float())
                rstd.copy_(torch.rsqrt(xf.var(-1, unbiased=False) + 1e-5).float())
                ref, native = reference(ts)
                original_native = native
                if args.saved_stats:
                    native = torch.ops.aten.native_layer_norm_backward(
                        dy, x, [n], mean, rstd, w, None, [True, False, False]
                    )[0]
                b = block_for(n)
                for name in args.names:
                    key = (name, b)
                    if key not in kernels:
                        kernels[key] = compile_ir(name, b)
                        out["kernels"][f"{name}/b{b}"] = digest(
                            ROOT / f"ir/b{b}/{name}.ttir"
                        )
                    storage.fill_(float("nan"))
                    launch(kernels[key], ts)
                    rec = check_output(ts, ref, native)
                    rec["recomputed_stats_native_max_abs"] = (
                        (dx.float() - original_native.float()).abs().max().item()
                    )
                    rec.update(kernel=name, N=n, offset=offset, distribution=dist)
                    rec["passed"] &= bool(storage[:, n:].isnan().all())
                    out["passed"] &= rec["passed"]
                    out["records"].append(rec)
                    if not rec["passed"]:
                        print("FAIL", rec, flush=True)
        print("checked", n, flush=True)
    save(args.output, out)
    print("passed", out["passed"], "records", len(out["records"]))
    if not out["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
