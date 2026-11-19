"""Exact FP32 add checks, independent source and CPU oracles, guards/refusals."""

import argparse
from pathlib import Path

import torch
from common import (
    BOUNDARIES,
    PERF_SIZES,
    ROOT,
    allocate,
    compile_ir,
    kernel_info,
    launch,
    launch_source,
    receipt,
    runtime_args,
    save_json,
    sha256,
)


def check_equal(actual, expected):
    nan = torch.isnan(expected)
    assert torch.equal(torch.isnan(actual), nan)
    assert torch.equal(actual[~nan].view(torch.int32), expected[~nan].view(torch.int32))


def check_case(kernels, n, pattern):
    x, y, out, storage = allocate(n, guarded=True)
    if pattern == "cancellation":
        x.mul_(1e20)
        y.copy_(-x)
    elif pattern == "constant":
        x.fill_(1.25)
        y.fill_(-0.125)
    elif pattern == "special":
        values = torch.tensor(
            [
                0.0,
                -0.0,
                float("inf"),
                -float("inf"),
                float("nan"),
                1e-40,
                -1e-40,
                3e38,
                -3e38,
                1.0,
                -1.0,
                1e20,
                -1e20,
                1e-30,
                -1e-30,
                0.5,
            ],
            device="cuda",
        )
        x.copy_(values.repeat((n + 15) // 16)[:n])
        y.copy_(torch.roll(values, 1).repeat((n + 15) // 16)[:n])
    x_before, y_before = x.clone(), y.clone()
    reference = x + y
    cpu_checked = n <= 65536
    if cpu_checked:
        check_equal(reference.cpu(), x.cpu() + y.cpu())
    storage.fill_(123456.0)
    out.fill_(float("nan"))
    launch_source(x, y, out)
    check_equal(out, reference)
    source = out.clone()
    errors = {}
    for name, ck in kernels.items():
        storage.fill_(123456.0)
        out.fill_(float("nan"))
        launch(ck, x, y, out)
        check_equal(out, reference)
        check_equal(out, source)
        assert torch.all(storage[:16] == 123456.0) and torch.all(
            storage[-16:] == 123456.0
        )
        check_equal(x, x_before)
        check_equal(y, y_before)
        finite = torch.isfinite(reference)
        errors[name] = (
            float((out[finite] - reference[finite]).abs().max())
            if finite.any()
            else 0.0
        )
    return dict(
        n=n,
        pattern=pattern,
        passed=True,
        cpu_checked=cpu_checked,
        exact_non_nan_bits=True,
        canaries=True,
        inputs_unchanged=True,
        max_abs_error=errors,
    )


def refusals():
    x, y, out, _ = allocate(32)
    tests = [
        (x[:17], y[:17], out[:17]),
        (x[:0], y[:0], out[:0]),
        (x, y, x),
        (x[::2], y[::2], out[::2]),
        (x.half(), y.half(), out.half()),
        (x[1:17], y[1:17], out[1:17]),
    ]
    for tensors in tests:
        try:
            runtime_args(*tensors)
        except AssertionError:
            continue
        raise AssertionError("Unsupported input accepted")
    return len(tests)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+", help="Include baseline explicitly")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sanitizer", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.manual_seed(args.seed)
    kernels = {name: compile_ir(name) for name in args.names}
    sizes = sorted(
        set(BOUNDARIES + ([4096, 2**22, 2**27] if args.sanitizer else PERF_SIZES))
    )
    rows = []
    for n in sizes:
        rows.append(check_case(kernels, n, "random"))
        print(f"PASS random N={n}", flush=True)
    for n in (16, 1040, 98432):
        for pattern in ("constant", "cancellation", "special"):
            rows.append(check_case(kernels, n, pattern))
    torch.cuda.synchronize()
    save_json(
        args.output,
        dict(
            passed=True,
            **receipt(),
            seed=args.seed,
            sanitizer=args.sanitizer,
            kernels={name: sha256(ROOT / f"ir/{name}.ttir") for name in kernels},
            artifacts={name: kernel_info(ck, name) for name, ck in kernels.items()},
            cases=rows,
            refusals_passed=refusals(),
        ),
    )


if __name__ == "__main__":
    main()
