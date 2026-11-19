"""Validate the adapter boundary and actual matched-envelope binding without timing."""

# flake8: noqa: E402 -- standalone MM helper bootstrap.

import argparse
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import triton.compiler.compiler as compiler
from bridge import compile_file, descriptor_source_reader
from common import ROOT as MM
from common import allocate, dump_kernel, save_json, sha256
from environment import verify_distribution_files
from experiment_round1 import idle
from experiment_round2 import extended_cases
from probe import CONFIG, ROOT, launch, parser_checks
from triton._C import libtriton
from triton.backends.nvidia import driver
from triton.backends.nvidia.compiler import CUDABackend, get_ptxas


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists()
    idle()
    environment = verify_distribution_files()
    original = compiler.IRSource.__init__
    baseline, a0 = compile_file(ROOT / "ir/envelope_baseline.ttir", CONFIG["options"])
    candidate, a1 = compile_file(ROOT / "ir/envelope_host_v1.ttir", CONFIG["options"])
    assert baseline.src.signature == candidate.src.signature
    assert baseline.metadata.target == candidate.metadata.target
    backend = CUDABackend(baseline.metadata.target)
    options = backend.parse_options(CONFIG["options"])
    for k in vars(options):
        assert getattr(baseline.metadata, k) == getattr(candidate.metadata, k), k
    for ck in [baseline, candidate]:
        assert ck.metadata.global_scratch_size == 0
    assert (
        len(baseline.metadata.tensordesc_meta)
        == len(candidate.metadata.tensordesc_meta)
        == 3
    )
    assert a0["native_expanded_signature"] == a1["native_expanded_signature"]
    # Identity with the original pointer baseline after accounting for the
    # declared descriptor envelope is checked on every boundary/stress input.
    from common import compile_ir
    from common import launch as original_launch

    native = compile_ir("bm64_bn128_bk64_g8_w4_s4")
    records = []
    torch.manual_seed(123)
    for m, n, k, pad, dist in extended_cases(CONFIG, "full"):
        ts = allocate(m, n, k, pad)
        a, b, c, storage = ts
        if dist == "zero":
            a.zero_()
        elif dist == "constant":
            a.fill_(0.25)
            b.fill_(0.25)
        elif dist == "large":
            a.mul_(4)
            b.mul_(4)
        elif dist == "small":
            a.mul_(1e-3)
            b.mul_(1e-3)
        elif dist == "alternating":
            b[::2].copy_(b[1::2])
            a[:, 1::2].copy_(-a[:, ::2])
        storage.fill_(float("nan"))
        original_launch(native, ts, CONFIG)
        reference = c.clone()
        for name, ck in [
            ("envelope_baseline", baseline),
            ("envelope_host_v1", candidate),
        ]:
            storage.fill_(float("nan"))
            launch(ck, ts, name)
            torch.testing.assert_close(c, reference, rtol=0, atol=0)
            assert storage[:, n:].isnan().all()
        records.append(
            dict(
                shape=[m, n, k],
                padding=pad,
                distribution=dist,
                all_three_outputs_bitwise_identical=True,
            )
        )
    # The adapter constructor restores even if native source parsing fails.
    try:
        with descriptor_source_reader(ROOT / "ir/nonexistent.ttir"):
            import triton

            triton.compile(str(ROOT / "ir/nonexistent.ttir"), options=CONFIG["options"])
    except Exception as e:
        constructor_failure = dict(
            error_type=type(e).__name__,
            reader_restored=compiler.IRSource.__init__ is original,
        )
    else:
        raise AssertionError("Invalid source must not compile")
    assert constructor_failure["reader_restored"]
    rejected = []
    for label, path in [("unknown_path", ROOT / "results/absent.ttir")]:
        try:
            compile_file(path, CONFIG["options"])
        except AssertionError:
            rejected.append(label)
        else:
            raise AssertionError("Undeclared prototype must be rejected")
    idle()
    save_json(
        args.output,
        dict(
            passed=True,
            stage_a_status="INTEGRATION_POSSIBLE_RACECHECK_BLOCKS_OPTIMIZATION",
            environment=environment,
            torch=torch.__version__,
            torch_path=torch.__file__,
            gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
            parser=parser_checks(),
            constructor_failure=constructor_failure,
            undeclared_rejected=rejected,
            matched_logical_and_expanded_signatures=True,
            full_parsed_options_equal=True,
            workspace_zero=True,
            records=records,
            artifacts={
                name: dump_kernel(ck, "host_descriptor/" + name)
                for name, ck in [
                    ("envelope_baseline", baseline),
                    ("envelope_host_v1", candidate),
                ]
            },
            audit=dict(baseline=a0, candidate=a1),
            source_hashes={
                str(p): sha256(p)
                for p in [
                    ROOT / "bridge.py",
                    ROOT / "probe.py",
                    ROOT / "final_checks.py",
                    ROOT / "source_controls.py",
                ]
            },
            toolchain={
                str(p): sha256(p)
                for p in [
                    inspect.getfile(compiler),
                    inspect.getfile(driver),
                    inspect.getfile(CUDABackend),
                    libtriton.__file__,
                    get_ptxas(90).path,
                ]
            },
            old_baseline_manifest_sha256=sha256(MM / "baseline.json"),
            old_selection_sha256=sha256(MM / "results/round1_selection.json"),
            native_control="native_source_control.json and native_source_racecheck.log; no reader adapter; warnings reproduced",
        ),
    )
    print(
        "PASS: 37 triple-bitwise cases, matched logical/physical ABI, exact parsed options, zero workspace and restoration/refusals; race qualification still blocked"
    )


if __name__ == "__main__":
    main()
