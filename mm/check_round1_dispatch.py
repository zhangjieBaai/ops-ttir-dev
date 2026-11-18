"""Exercise delivered optimized dispatch and prove selected TTIR/cubin correspondence."""

import argparse
from pathlib import Path

import torch
from common import (
    ATOL,
    PERF_SHAPES,
    ROOT,
    RTOL,
    allocate,
    compile_ir,
    digest,
    save_json,
    sha256,
)
from experiment_round1 import compatible, idle
from optimized import load_kernels, matmul


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    idle()
    state = load_kernels()
    manifest, selection, kernels = state
    artifact_checks = {}
    for family, config in manifest["families"].items():
        source = compile_ir(family, selection["families"][family]["name"])
        ck = kernels[family]
        compatible(source, ck, config)
        assert ck.asm["cubin"] == source.asm["cubin"]
        artifact_checks[family] = dict(
            ttir_path=selection["families"][family]["optimized_path"],
            ttir_sha256=selection["families"][family]["optimized_sha256"],
            cubin_sha256=digest(ck.asm["cubin"]),
        )
    records = []
    torch.manual_seed(107)
    for shape in PERF_SHAPES:
        a, b, _, _ = allocate(*shape)
        actual = matmul(a, b, state)
        torch.testing.assert_close(actual, torch.mm(a, b), atol=ATOL, rtol=RTOL)
        family = manifest["shape_to_family"][str(shape[0])]
        records.append(
            dict(shape=shape, family=family, passed=True, **artifact_checks[family])
        )
    try:
        a, b, _, _ = allocate(128, 128, 128)
        matmul(a, b, state)
    except KeyError:
        pass
    else:
        raise AssertionError("Unmapped shape must be rejected")
    idle()
    save_json(
        args.output,
        dict(
            passed=True,
            selection_sha256=sha256(ROOT / "results/round1_selection.json"),
            records=records,
            unmapped_shape_rejected=True,
            artifact_checks=artifact_checks,
        ),
    )
    print(
        "PASS: all31 delivered matmul calls use selected native TTIR; "
        "six cubins match selected sources; unknown shape rejected"
    )


if __name__ == "__main__":
    main()
