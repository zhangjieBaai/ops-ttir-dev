"""FP64/ATen oracle, stress inputs, output guards and input preservation."""

import argparse
from pathlib import Path

import torch
from common import (
    ALL_SHAPES,
    allocate,
    check_contention,
    compile_ir,
    gate,
    group_for,
    kernel_info,
    launch,
    receipt,
    save_json,
)

DISTRIBUTIONS = (
    "normal",
    "zero",
    "constant",
    "offset",
    "alternating",
    "outlier",
    "neg_inf",
    "nan",
)


def prepare(x, distribution):
    if distribution == "zero":
        x.zero_()
    elif distribution == "constant":
        x.fill_(3.5)
    elif distribution == "offset":
        x.add_(1000)
    elif distribution == "alternating":
        x[:, ::2, :] = 10000
        x[:, 1::2, :] = -10000
    elif distribution == "outlier":
        x.fill_(1)
        x[:, 0, :] = 10000
    elif distribution == "neg_inf":
        x.fill_(-float("inf"))
    elif distribution == "nan":
        x[:, 0, :] = float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sanitizer",
        action="store_true",
        help="All shapes, normal and all -inf stress",
    )
    args = parser.parse_args()
    gate()
    check_contention()
    torch.manual_seed(args.seed)
    names = list(dict.fromkeys(["baseline", *args.names]))
    kernels = {
        (group_for(s)[0], name): compile_ir(group_for(s)[0], name)
        for s in ALL_SHAPES
        for name in names
    }
    result = dict(
        **receipt(), seed=args.seed, passed=False, kernels={}, records=[], failures=[]
    )
    for (group, name), ck in kernels.items():
        result["kernels"].setdefault(group, {})[name] = kernel_info(ck, group, name)
    distributions = ("normal", "neg_inf") if args.sanitizer else DISTRIBUTIONS
    for shape in ALL_SHAPES:
        group, _ = group_for(shape)
        for distribution in distributions:
            tensors = allocate(shape, guarded=True)
            x, y, storage = tensors
            prepare(x, distribution)
            saved_x = x.clone()
            reference64 = (
                torch.mean(x.cpu().double(), dim=1).to(torch.float16).float().cuda()
            )
            reference32 = torch.mean(x, dim=1).float()
            for name in names:
                y.fill_(float("nan"))
                launch(kernels[group, name], shape, tensors)
                torch.cuda.synchronize()
                record = dict(
                    shape=shape, group=group, distribution=distribution, name=name
                )
                try:
                    torch.testing.assert_close(
                        y.float(), reference64, rtol=2e-3, atol=2e-3, equal_nan=True
                    )
                    torch.testing.assert_close(
                        y.float(), reference32, rtol=2e-3, atol=2e-3, equal_nan=True
                    )
                    assert torch.equal(torch.isneginf(y), torch.isneginf(reference64))
                    assert torch.isfinite(y[torch.isfinite(reference64)]).all()
                    assert (storage[:16] == 12345.0).all() and (
                        storage[-16:] == 12345.0
                    ).all()
                    torch.testing.assert_close(
                        x, saved_x, rtol=0, atol=0, equal_nan=True
                    )
                    finite = torch.isfinite(reference64)
                    record.update(
                        passed=True,
                        max_abs_error_fp64_rounded=(
                            float((y.float()[finite] - reference64[finite]).abs().max())
                            if finite.any()
                            else 0.0
                        ),
                    )
                except (AssertionError, RuntimeError) as error:
                    record.update(passed=False, error=str(error))
                    result["failures"].append(record)
                result["records"].append(record)
        print("validated", shape, flush=True)
    result["contention_after"] = check_contention()
    result["passed"] = not result["failures"]
    save_json(args.output, result)
    print(
        "PASS" if result["passed"] else "FAIL",
        len(result["records"]),
        "checks",
        flush=True,
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
