"""Native ATen output, FP64 statistics, stress/edge cases and padding canaries."""

import argparse
import os

import torch
from common import EPS, ROOT, allocate, compile_ir, launch, save_json, sha256


def cases(suite):
    if suite == "round2-boundaries":
        return [
            (257, n, pad, d, EPS)
            for edge in [1024, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 14336, 16384]
            for n in [edge - 16, edge, edge + 16]
            for pad in [0, 16]
            for d in ["normal", "offset"]
        ]
    if suite == "smoke":
        return [
            (17, n, 0, "normal", EPS)
            for n in [1024, 2048, 4096, 8192, 8704, 15872, 16400, 32768]
        ]
    if suite == "sanitizer":
        return [
            (17, n, 16, "normal", EPS)
            for n in [
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
                16400,
                32768,
            ]
        ]
    result = [(4096, n, 0, "normal", EPS) for n in range(1024, 15873, 512)]
    result += [(1151, 8192, 0, "normal", EPS)]
    result += [
        (m, n, 0, "normal", EPS)
        for m in [1, 128, 256, 512]
        for n in [1024, 8192, 8704, 15872]
    ]
    result += [
        (257, n, 0, d, EPS)
        for n in [1024, 4096, 8192, 8704, 15872]
        for d in ["constant", "offset", "zero_mean"]
    ]
    sizes = [
        16,
        128,
        512,
        992,
        1008,
        1024,
        1040,
        1520,
        1536,
        1552,
        2032,
        2048,
        2064,
        4080,
        4096,
        4112,
        8176,
        8192,
        8208,
        10224,
        10240,
        10256,
        12272,
        12288,
        12304,
        14320,
        14336,
        14352,
        16368,
        16384,
        16400,
        32768,
    ]
    result += [(257, n, pad, "normal", EPS) for n in sizes for pad in [0, 16]]
    result += [
        (1, n, 16, d, eps)
        for n in [16, 8192, 8208, 16400]
        for d in ["constant", "offset"]
        for eps in [1e-6, 1e-3]
    ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+")
    parser.add_argument(
        "--suite",
        choices=["smoke", "full", "sanitizer", "round2-boundaries"],
        default="full",
    )
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    assert os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    ), "Isolate a GPU explicitly before running"
    torch.manual_seed(42)
    kernels = {name: compile_ir(name) for name in args.names}
    result = dict(
        suite=args.suite,
        passed=True,
        kernels={name: sha256(ROOT / f"ir/{name}.ttir") for name in kernels},
        tolerances=dict(
            y_max_abs_lt=1e-2,
            mean_rtol=2e-5,
            mean_atol=1e-5,
            rstd_rtol=2e-4,
            rstd_atol=1e-5,
        ),
        records=[],
    )
    for index, (m, n, pad, distribution, eps) in enumerate(cases(args.suite)):
        tensors = allocate(m, n, pad)
        x, y, w, b, mean, rstd, storage = tensors
        if distribution == "constant":
            x.fill_(2.0)
        elif distribution == "offset":
            x.add_(1000.0)
        elif distribution == "normal":
            x.mul_(0.5).add_(-2.3)
        ref = torch.nn.functional.layer_norm(x, (n,), w, b, eps).float()
        xf = x.double()
        mean_ref = xf.mean(-1).float()
        rstd_ref = torch.rsqrt(xf.var(-1, unbiased=False) + eps).float()
        del xf
        for name, ck in kernels.items():
            # Reset sentinels so an unwritten output cannot inherit a previous kernel's result.
            storage.fill_(float("nan"))
            mean.fill_(float("nan"))
            rstd.fill_(float("nan"))
            launch(ck, tensors, eps)
            error = (y.float() - ref).abs().max().item()
            mean_error = (mean - mean_ref).abs().max().item()
            rstd_error = (rstd - rstd_ref).abs().max().item()
            passed = bool(
                torch.isfinite(y).all()
                and torch.isfinite(mean).all()
                and torch.isfinite(rstd).all()
            )
            passed &= error < 1e-2
            passed &= torch.allclose(mean, mean_ref, rtol=2e-5, atol=1e-5)
            passed &= torch.allclose(rstd, rstd_ref, rtol=2e-4, atol=1e-5)
            passed &= bool(storage[:, n:].isnan().all())
            row = dict(
                kernel=name,
                M=m,
                N=n,
                pad=pad,
                distribution=distribution,
                eps=eps,
                y_error=error,
                mean_error=mean_error,
                rstd_error=rstd_error,
                passed=passed,
            )
            result["records"].append(row)
            result["passed"] &= passed
            if not passed:
                print("FAIL", row, flush=True)
        if index % 10 == 0:
            print(f"{index+1}/{len(cases(args.suite))} cases checked", flush=True)
    save_json(args.output, result)
    print(
        f"{args.suite}: {len(result['records'])} kernel cases; passed={result['passed']}",
        flush=True,
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
