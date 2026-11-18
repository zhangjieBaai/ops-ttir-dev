"""Explain infinite setup-autotune timings without re-tuning or executing kernels."""

import argparse
import json
import math
from pathlib import Path

from common import (
    SEED_SHAPE,
    allocate,
    dump_kernel,
    frozen,
    gate,
    runtime_args,
    save_json,
)
from tutorial import get_cuda_autotune_config, matmul_kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gate()
    frozen()
    config = json.loads(
        (Path(__file__).parent / "setup_attempts/v3/baseline.json").read_text()
    )
    tensors = allocate(*SEED_SHAPE)
    records = []
    for candidate in get_cuda_autotune_config():
        timings = config["autotune_timings"][str(candidate)]
        if not any(math.isinf(value) for value in timings):
            continue
        row = dict(
            config=str(candidate),
            original_timing_status="infinite; not eligible for selection",
        )
        try:
            ck = matmul_kernel.warmup(
                *runtime_args(tensors),
                ACTIVATION="",
                **candidate.all_kwargs(),
                grid=(1, 1, 1),
            )
            # Native loader checks resource limits; no GPU kernel is launched.
            ck._init_handles()
            row.update(
                status="compiled_and_loadable_now; original failure reason not reproduced",
                metadata=ck.metadata._asdict(),
            )
            dump_kernel(ck, "autotune_diagnostic")
        except Exception as exc:
            row.update(status="failed", exception=type(exc).__name__, message=str(exc))
        records.append(row)
    save_json(args.output, dict(baseline_sha256=config["ttir_sha256"], records=records))
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
