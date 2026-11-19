"""CPU FP64, native ATen, intermediate coverage, canaries and stress verification."""

import argparse
from pathlib import Path

import torch
from common import (
    ATOL,
    L1_FACTOR,
    PERF_SIZES,
    allocate,
    frozen,
    geometry,
    hashes,
    launch,
    load_kernels,
    receipt,
    save_json,
)

CANARY = 1234567.0


def check_values(actual, reference, l1):
    special = ~torch.isfinite(reference)
    if special.any():
        assert torch.equal(torch.isnan(actual), torch.isnan(reference))
        assert torch.equal(torch.isposinf(actual), torch.isposinf(reference))
        assert torch.equal(torch.isneginf(actual), torch.isneginf(reference))
    finite = ~special
    error = (actual[finite].double() - reference[finite]).abs()
    bound = ATOL + L1_FACTOR * l1[finite]
    assert torch.isfinite(actual[finite]).all()
    assert (error <= bound).all(), (error.max().item(), bound.max().item())
    return dict(
        max_abs_error=error.max().item() if error.numel() else 0,
        max_bound_fraction=(error / bound).max().item() if error.numel() else 0,
    )


def patterns(n):
    names = ["random"]
    if n in [1, 127, 65535, 65536, 65537, 1048577, 67108864]:
        names += ["zero", "constant", "alternating", "offset", "tiny", "impulse"]
    if n in [1, 127, 65537]:
        names += ["nan", "posinf", "neginf"]
        if n > 1:
            names += ["mixed_inf"]
    return names


def prepare(x, pattern):
    if pattern == "zero":
        x.zero_()
    elif pattern == "constant":
        x.fill_(0.125)
    elif pattern == "alternating":
        x.fill_(1.0)
        x[1::2] = -1.0
    elif pattern == "offset":
        x.add_(1000)
    elif pattern == "tiny":
        x.mul_(1e-8)
    elif pattern == "impulse":
        x.zero_()
        x[-1] = 7.0
    elif pattern in ["nan", "posinf", "neginf", "mixed_inf"]:
        x[0] = float(
            "nan" if pattern == "nan" else "-inf" if pattern == "neginf" else "inf"
        )
        if pattern == "mixed_inf":
            x[-1] = -float("inf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sanitizer",
        action="store_true",
        help="All 30 shapes, random and selected boundary stress",
    )
    args = parser.parse_args()
    assert not args.output.exists()
    torch.manual_seed(args.seed)
    manifest = frozen()
    kernels = {name: load_kernels(name) for name in args.names}
    result = dict(
        **receipt(),
        passed=False,
        seed=args.seed,
        kernels={name: hashes(name) for name in args.names},
        records=[],
    )
    for n in PERF_SIZES:
        for pattern in ["random"] if args.sanitizer else patterns(n):
            tensors = allocate(n)
            x = tensors[0]
            prepare(x, pattern)
            cpu = x.cpu().double()
            reference, l1 = cpu.sum().reshape(1), cpu.abs().sum().reshape(1)
            native = torch.sum(x).cpu().double().reshape(1)
            block, size, _ = geometry(n)
            padded = torch.nn.functional.pad(cpu, (0, size * block - n))
            ref_mid = padded.reshape(size, block).sum(1)
            mid_l1 = padded.reshape(size, block).abs().sum(1)
            for name, items in kernels.items():
                tensors[3].fill_(CANARY)
                tensors[4].fill_(CANARY)
                tensors[1].fill_(float("nan"))
                tensors[2].fill_(float("nan"))
                launch(items, tensors, manifest)
                torch.cuda.synchronize()
                out, mid = tensors[2].cpu().double(), tensors[1].cpu().double()
                errors = check_values(out, reference, l1)
                native_errors = check_values(out, native, l1)
                mid_errors = check_values(mid, ref_mid, mid_l1)
                assert (tensors[3][size:] == CANARY).all() and (
                    tensors[4][1:] == CANARY
                ).all()
                assert torch.allclose(
                    x.cpu().double(), cpu, rtol=0, atol=0, equal_nan=True
                )
                result["records"].append(
                    dict(
                        n=n,
                        pattern=pattern,
                        name=name,
                        output=errors,
                        native=native_errors,
                        intermediate=mid_errors,
                        canary=True,
                        input_unchanged=True,
                    )
                )
            print(f"PASS accuracy {n} {pattern}", flush=True)
    result["passed"] = True
    save_json(args.output, result)


if __name__ == "__main__":
    main()
