"""Audit immutable candidate generation, ELF code, and real compilation resources."""

import argparse
import json
import struct
from pathlib import Path

from common import ROOT, digest, ir_path, manifest, save_json, sha256
from make_candidates import narrow_indices, trim_padding, two_pass, without_locations


def text_sections(path):
    raw = path.read_bytes()
    assert raw[:6] == b"\x7fELF\x02\x01"
    section_offset = struct.unpack_from("<Q", raw, 40)[0]
    size, count, strings_index = struct.unpack_from("<HHH", raw, 58)
    sections = [
        struct.unpack_from("<IIQQQQIIQQ", raw, section_offset + i * size)
        for i in range(count)
    ]
    strings = sections[strings_index]
    names = raw[strings[4] : strings[4] + strings[5]]
    result = {}
    for section in sections:
        end = names.index(b"\x00", section[0])
        name = names[section[0] : end].decode()
        if name.startswith(".text."):
            result[name] = digest(raw[section[4] : section[4] + section[5]])
    assert result
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    benchmark = json.loads(args.benchmark.read_text())
    result = dict(passed=True, artifacts={}, candidate_reproduction={})
    transforms = {
        "r1_i32": lambda t, c: narrow_indices(t),
        "r1_trim": trim_padding,
        "r1_two_pass": lambda t, c: two_pass(t),
        "r1_combined": lambda t, c: narrow_indices(two_pass(trim_padding(t, c))),
    }
    for group, config in manifest()["groups"].items():
        clean = without_locations(ir_path(group, "baseline").read_text())
        for name, transform in transforms.items():
            assert transform(clean, config) == ir_path(group, name).read_text()
        mixed = (
            two_pass(trim_padding(clean, config))
            if "TILE_K" in config["meta"]
            else ir_path(group, "baseline").read_text()
        )
        assert mixed == ir_path(group, "r1_mixed").read_text()
        result["candidate_reproduction"][group] = {
            n: sha256(ir_path(group, n)) for n in [*transforms, "r1_mixed"]
        }
    for group, artifacts in benchmark["artifacts"].items():
        codes = {}
        for name, artifact in artifacts.items():
            cubin = (
                ROOT / "artifacts" / group / name / artifact["hash"] / "kernel.cubin"
            )
            assert sha256(cubin) == artifact["stages"]["cubin"]
            codes[name] = text_sections(cubin)
        result["artifacts"][group] = dict(
            text_sections=codes,
            text_equal=codes["baseline"] == codes["r1_trim"],
            resources={
                n: {k: v[k] for k in ("n_regs", "n_spills")}
                | {"shared": v["metadata"]["shared"]}
                for n, v in artifacts.items()
            },
        )
    save_json(args.output, result)
    print("PASS candidate reproduction / ELF code / resources")


if __name__ == "__main__":
    main()
