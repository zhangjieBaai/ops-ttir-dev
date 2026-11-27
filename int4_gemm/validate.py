"""Accuracy, canary, input immutability and pressure distributions."""

import argparse

import torch
from common import (
    ROOT,
    allocate,
    assert_output,
    compile_ir,
    environment,
    launch,
    manifest,
    record,
    reference,
    save,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--names", nargs="+", default=["baseline"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    env = environment()
    torch.backends.cuda.matmul.allow_tf32 = False
    results, kernels = [], {}
    for case in manifest()["cases"]:
        if args.quick and case["shape"][1] > 1024:
            continue
        for name in args.names:
            kernels[case["id"], name] = compile_ir(case, name)
        modes = ["random", "extreme_nibble", "zero_scale", "zero", "constant"]
        for mode in modes:
            ts = allocate(case["shape"], seed=args.seed, mode=mode)
            inputs = [t.clone() for t in ts[:4]]
            ref = reference(ts)
            baseline = kernels.get((case["id"], "baseline")) or compile_ir(
                case, "baseline"
            )
            launch(baseline, ts, case["config"])
            base_out = ts[4].clone()
            if case["shape"][1] <= 256:
                assert_output(ts, reference(ts, cpu=True))
            for name in args.names:
                ts[4].fill_(float("nan"))
                ck = kernels[case["id"], name]
                launch(ck, ts, case["config"])
                error = assert_output(ts, ref)
                torch.testing.assert_close(ts[4], base_out, rtol=0, atol=0)
                for actual, original in zip(ts[:4], inputs):
                    assert torch.equal(actual, original)
                results.append(
                    dict(
                        case=case["id"],
                        name=name,
                        mode=mode,
                        max_abs_error=error,
                        baseline_bitwise=True,
                        inputs_unchanged=True,
                        canary=True,
                        kernel=record(ck, ROOT / "ir" / case["id"] / (name + ".ttir")),
                    )
                )
            print(case["id"], mode, "PASS", flush=True)
    save(
        args.output, dict(passed=True, environment=env, seed=args.seed, checks=results)
    )


if __name__ == "__main__":
    main()
