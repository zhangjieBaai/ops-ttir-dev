"""Check hint preservation and vector-load evidence, rather than assuming hints work."""

import argparse
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = args.results_dir
    trials = {}
    for version in [1, 2, 3]:
        suffix = "" if version == 1 else f"_v{version}"
        result_path = results / f"source_feasibility{suffix}.json"
        result = json.loads(result_path.read_text())
        variants = {}
        for bits, item in result["variants"].items():
            ttir_path = Path(item["actual_ttir_path"])
            ptx_path = ttir_path.with_suffix(".ptx")
            assert sha(ttir_path) == item["products"]["ttir"]
            assert sha(ptx_path) == item["products"]["ptx"]
            ttir = ttir_path.read_text()
            ptx = ptx_path.read_text()
            n_arg_hint = bool(
                re.search(r"%N: i32 \{tt.divisibility = 16 : i32\}", ttir)
            )
            ext_hint = any(
                "extsi %N" in line and "divisibility" in line
                for line in ttir.splitlines()
            )
            vector4 = ptx.count("ld.global.v4.b32")
            vector2 = ptx.count("ld.global.v2.b32")
            if version == 1:
                assert not n_arg_hint and not ext_hint
            elif version == 2:
                assert not n_arg_hint and ext_hint == (bits[-1] == "1")
                assert vector4 == 0 and vector2 == 0
            else:
                assert n_arg_hint == (bits[-1] == "1")
                if bits[-1] == "1":
                    assert vector4 > 0
            if bits in ["000", "001", "110", "111"]:
                destination = HERE / f"source_ir/probe_v{version}/{bits}.ptx"
                destination.write_bytes(ptx_path.read_bytes())
            variants[bits] = dict(
                n_argument_divisibility16=n_arg_hint,
                cast_divisibility16=ext_hint,
                ptx_vector4_load_instructions=vector4,
                ptx_vector2_load_instructions=vector2,
                ttir_sha256=sha(ttir_path),
                ptx_sha256=sha(ptx_path),
            )
        if version == 1:
            for bits in ["000", "010", "100", "110"]:
                other = bits[:-1] + "1"
                assert all(
                    result["variants"][bits]["products"][stage]
                    == result["variants"][other]["products"][stage]
                    for stage in ["ttir", "ptx", "cubin"]
                )
        trials[str(version)] = dict(receipt_sha256=sha(result_path), variants=variants)
    output = args.output or results / "source_audit_final.json"
    value = dict(
        passed=True,
        trials=trials,
        source_sha256=sha(Path(__file__)),
        source_dependencies={
            name: sha(HERE / name)
            for name in [
                "source_probes.py",
                "source_probes_v2.py",
                "source_probes_v3.py",
                "verify_source.py",
            ]
        },
        conclusions={
            "v1": "Direct argument hint discarded in this pinned example.",
            "v2": "Cast hint survives but does not reproduce target vector loads.",
            "v3": "JIT alignment specialization reproduces exact N argument attribute and vector loads.",
        },
    )
    if output.exists():
        assert json.loads(output.read_text()) == value
    else:
        output.write_text(json.dumps(value, indent=2) + "\n")
    print("PASS: three hint placements audited against actual TTIR/PTX hashes")


if __name__ == "__main__":
    main()
