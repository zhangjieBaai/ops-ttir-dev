"""Exercise the actual allocating public-Triton example and its frozen-scope refusals."""

import argparse
from pathlib import Path

import torch
from ablation_common import isolated
from ablation_example import VARIANTS, sum_fp32
from common import PERF_SIZES, ROOT, allocate, receipt, save_json, sha256
from validate import check_values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    rows = []
    with isolated() as telemetry:
        torch.manual_seed(20261120)
        for n in PERF_SIZES:
            x = allocate(n)[0]
            cpu = x.cpu().double()
            reference, l1 = cpu.sum().reshape(1), cpu.abs().sum().reshape(1)
            for variant in sorted(VARIANTS):
                output = sum_fp32(x, variant)
                error = check_values(output.cpu().double().reshape(1), reference, l1)
                rows.append(dict(n=n, variant=variant, error=error))
        x = torch.empty(17, dtype=torch.float32, device="cuda")
        rejected = []
        for variant, size in [("cs", 16), ("single", 17)]:
            try:
                sum_fp32(x[:size], variant)
            except ValueError as error:
                rejected.append(dict(variant=variant, n=size, error=str(error)))
            else:
                raise AssertionError(
                    "Invalid input or unsupported cache modifier accepted"
                )
        save_json(
            args.output,
            dict(
                **receipt(),
                passed=True,
                example_sha256=sha256(ROOT / "ablation_example.py"),
                source_sha256=sha256(ROOT / "ablation_triton.py"),
                records=rows,
                refusals=rejected,
                telemetry=list(telemetry),
                uncontended=bool(telemetry)
                and all(
                    not s.get("other_pids") and "error" not in s for s in telemetry
                ),
            ),
        )
    print(f"PASS {len(rows)} public wrapper outputs and {len(rejected)} refusals")


if __name__ == "__main__":
    main()
