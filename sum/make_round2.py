"""Direct, immutable TTIR rewrites for fixed-launch tail and separate single-CTA paths."""

import argparse
import re

import triton
from common import ROOT, frozen, geometry, save_json, sha256, write_once


def tail_load(text):
    # Specialized n=1 has no runtime length/mask. Keep its frozen IR unchanged.
    if "%mask = arith.extsi %M" not in text:
        return text
    pattern = r"    (%\w+) = tt.load (%\w+), (%\w+), (%\w+) : (tensor<\d+x!tt.ptr<f32>>)([^\n]*)"

    def replace(match):
        value, pointer, mask, zero, pointers, location = match.groups()
        tensor = pointers.replace("!tt.ptr<f32>", "f32")
        return (
            "    %r2_end = arith.addi %offset_1, %offset : i64\n"
            "    %r2_full = arith.cmpi sle, %r2_end, %mask : i64\n"
            f"    {value} = scf.if %r2_full -> ({tensor}) {{\n"
            f"      %r2_vector = tt.load {pointer} : {pointers}\n"
            f"      scf.yield %r2_vector : {tensor}\n"
            "    } else {\n"
            f"      %r2_tail = tt.load {pointer}, {mask}, {zero} : {pointers}\n"
            f"      scf.yield %r2_tail : {tensor}\n"
            f"    }}{location}"
        )

    result, count = re.subn(pattern, replace, text)
    assert count == 1
    return result


def single_cta(text, n):
    old, _, _ = geometry(n)
    width = triton.next_power_of_2(n)
    if n == 1:
        return text
    text = text.replace(f"tensor<{old}x", f"tensor<{width}x")
    text = text.replace(f"end = {old} : i32", f"end = {width} : i32")
    return text.replace(f"arith.constant {old} : i64", f"arith.constant {width} : i64")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", action="store_true")
    args = parser.parse_args()
    manifest = frozen()
    record = {}
    for group, config in manifest["groups"].items():
        path = ROOT / config["path"]
        candidate = path.with_name("r2_tail.ttir")
        original = path.read_text()
        write_once(
            candidate, tail_load(original) if group.startswith("s1_") else original
        )
        record[group] = dict(
            path=str(candidate.relative_to(ROOT)), sha256=sha256(candidate)
        )
    output = ROOT / "results/round2_tail_candidates.json"
    if not output.exists():
        save_json(output, record)
    if args.single:
        routes = {}
        for n, groups in manifest["shapes"].items():
            size = int(n)
            if size > 16384:
                routes[n] = dict(kind="two_stage", groups=groups)
                continue
            path = ROOT / "ir/launch_v2" / f"n{n}" / "r2_single16384.ttir"
            source = ROOT / manifest["groups"][groups[0]]["path"]
            write_once(path, single_cta(source.read_text(), size))
            routes[n] = dict(
                kind="single",
                path=str(path.relative_to(ROOT)),
                sha256=sha256(path),
                source_group=groups[0],
                grid=[1, 1, 1],
                runtime_indices=manifest["groups"][groups[0]]["runtime_indices"],
            )
        save_json(
            ROOT / "round2_manifest.json",
            dict(
                experiment="sum_launch_v2",
                source_manifest_sha256=sha256(ROOT / "baseline.json"),
                protocol_sha256=sha256(ROOT / "optimization_plan_round2.md"),
                threshold=16384,
                routes=routes,
            ),
        )


if __name__ == "__main__":
    main()
