"""Compare actual TTIR output against CPU FP64 and native GPU ATen."""

import argparse

import torch
from common import (
    CONTRACT,
    SHAPES,
    allocate,
    compile_ir,
    identity,
    launch,
    provenance,
    save,
)

BOUNDARIES = [
    16,
    112,
    128,
    144,
    496,
    512,
    528,
    1008,
    1024,
    1040,
    4080,
    4096,
    4112,
    8176,
    8192,
    8208,
    16368,
    16384,
]
DISTRIBUTIONS = [
    "normal",
    "constant",
    "offset",
    "lowvar",
    "zero",
    "alternating",
    "outlier",
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--names", nargs="+", default=["baseline"])
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sanitizer", action="store_true")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    kernels = {name: compile_ir(name) for name in args.names}
    cases = [(m, n, "normal") for m, n in SHAPES]
    cases += [(m, n, d) for m in [1, 17] for n in BOUNDARIES for d in DISTRIBUTIONS]
    if args.sanitizer:
        cases = [
            (17, n, d)
            for n in [16, 128, 1040, 4096, 8208, 16384]
            for d in ["normal", "offset"]
        ]
    records = []
    for m, n, d in cases:
        ts = allocate(m, n, d)
        x = ts[0]
        for correction in [0.0, 1.0]:
            cpu = torch.var_mean(x.cpu().double(), dim=-1, correction=correction)
            native = torch.var_mean(x, dim=-1, correction=correction)
            for name, ck in kernels.items():
                ts[3].fill_(float("nan"))
                ts[4].fill_(float("nan"))
                launch(ck, ts, correction)
                errs = []
                for idx, label in [(1, "variance"), (2, "mean")]:
                    actual = ts[idx].cpu().double()
                    tol = CONTRACT["tolerance"][label]
                    torch.testing.assert_close(actual, cpu[idx - 1], **tol)
                    native_error = float(
                        (actual - native[idx - 1].cpu().double()).abs().max()
                    )
                    errs.append(
                        (float((actual - cpu[idx - 1]).abs().max()), native_error)
                    )
                for storage in ts[3:]:
                    assert (
                        torch.isnan(storage[:16]).all()
                        and torch.isnan(storage[-16:]).all()
                    )
                records.append(
                    dict(
                        name=name,
                        shape=[m, n],
                        distribution=d,
                        correction=correction,
                        max_abs_var=errs[0][0],
                        max_abs_mean=errs[1][0],
                        native_max_abs_var=errs[0][1],
                        native_max_abs_mean=errs[1][1],
                        passed=True,
                    )
                )
    torch.cuda.synchronize()
    save(
        args.output,
        dict(
            passed=True,
            provenance=provenance(),
            seed=args.seed,
            identities={n: identity(n, k) for n, k in kernels.items()},
            records=records,
        ),
    )
    print(f"PASS: {len(records)} candidate/case/correction checks")


if __name__ == "__main__":
    main()
