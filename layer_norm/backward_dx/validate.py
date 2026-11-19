"""dX checks against native ATen autograd, actual saved-stats FP64 formula, and small CPU FP64."""

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


def cases(suite):
    if suite == "smoke":
        return [(17, n, 0, "normal") for n in [1024, 1536, 3072, 6144, 8704, 15872]]
    edges = [
        16,
        1008,
        1024,
        1040,
        2032,
        2048,
        2064,
        4080,
        4096,
        4112,
        8176,
        8192,
        8208,
        16368,
        16384,
    ]
    result = [(17, n, p, "normal") for n in edges for p in [0, 16]]
    if suite == "sanitizer":
        return result
    result += [(4096, n, 0, "normal") for n in range(1024, 15873, 512)]
    result += [
        (m, n, 16, d)
        for m in [1, 33, 257]
        for n in [1024, 3072, 8192, 8704, 15872]
        for d in ["normal", "offset", "constant", "lowvar", "zero_dy"]
    ]
    return result


def check_output(tensors, ref, native):
    dx = tensors[0]
    actual = dx.double()
    rms = (
        (actual - ref).square().mean().sqrt() / ref.square().mean().sqrt().clamp_min(1)
    ).item()
    ok = bool(torch.isfinite(dx).all())
    ok &= torch.allclose(actual, ref, rtol=0.01, atol=0.01)
    ok &= torch.allclose(dx.float(), native.float(), rtol=0.01, atol=0.01)
    ok &= rms <= 0.003
    return dict(
        passed=ok,
        fp64_max_abs=(actual - ref).abs().max().item(),
        native_max_abs=(dx.float() - native.float()).abs().max().item(),
        normalized_rms=rms,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("names", nargs="+")
    p.add_argument("--suite", choices=["full", "smoke", "sanitizer"], default="full")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    assert not os.path.exists(args.output), "Choose a new output path"
    torch.manual_seed(args.seed)
    kernels = {}
    result = dict(
        passed=True,
        suite=args.suite,
        seed=args.seed,
        records=[],
        kernels={},
        tolerance=dict(rtol=0.01, atol=0.01, normalized_rms_max=0.003),
    )
    for i, (m, n, pad, d) in enumerate(cases(args.suite)):
        block = block_for(n)
        ts = allocate(m, n, pad, d)
        dx, dy, x, w, mean, rstd, storage = ts
        ref, native = reference(ts)
        before = [t.clone() for t in ts[1:6]]
        cpu = None
        if m * n <= 4096:
            cpu_x = x.detach().cpu().double().requires_grad_(True)
            cpu_y = torch.nn.functional.layer_norm(
                cpu_x, (n,), w.cpu().double(), None, 1e-5
            )
            cpu = torch.autograd.grad(cpu_y, cpu_x, dy.cpu().double())[0]
        for name in args.names:
            key = (name, block)
            if key not in kernels:
                kernels[key] = compile_ir(name, block)
                result["kernels"][f"{name}/b{block}"] = digest(
                    ROOT / f"ir/b{block}/{name}.ttir"
                )
            storage.fill_(float("nan"))
            launch(kernels[key], ts)
            rec = check_output(ts, ref, native)
            rec.update(kernel=name, M=m, N=n, pad=pad, distribution=d, block=block)
            rec["passed"] &= bool(storage[:, n:].isnan().all())
            rec["passed"] &= all(torch.equal(t, b) for t, b in zip(ts[1:6], before))
            if cpu is not None:
                rec["cpu_fp64_max_abs"] = (dx.cpu().double() - cpu).abs().max().item()
                rec["passed"] &= torch.allclose(
                    dx.cpu().double(), cpu, rtol=0.01, atol=0.01
                )
            result["passed"] &= rec["passed"]
            result["records"].append(rec)
            if not rec["passed"]:
                print("FAIL", rec, flush=True)
        if i % 10 == 0:
            print(i + 1, "/", len(cases(args.suite)), flush=True)
    save(args.output, result)
    print("passed", result["passed"], "checks", len(result["records"]), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
