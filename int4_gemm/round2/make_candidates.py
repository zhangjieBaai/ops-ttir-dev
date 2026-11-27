"""Deterministic direct TTIR rewrites from the accepted INT4 parent."""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import common as c  # noqa: E402


def narrow(text, bk, bn):
    begin = text.index("      %group_low =")
    end = text.index("      %low =", begin)
    segment = text[begin:end]
    scale = re.search(r"(%scale_low_\d+) = arith.extf", segment)[1]
    zero = re.search(r"(%zero_low_\d+) = arith.extf", segment)[1]
    nmask = re.search(
        r"(%packed_\d+) = arith.cmpi slt, %packed, %packed_\d+ : tensor<1x", text
    )[1]
    rowf = f"tensor<1x{bn}xf16>"
    row32 = f"tensor<1x{bn}xf32>"
    rowi = f"tensor<1x{bn}xi32>"
    rowp = f"tensor<1x{bn}x!tt.ptr<f16>>"
    wide = f"tensor<{bk//2}x{bn}xf32>"
    repl = f"""      %r2_group = arith.divui %k0, %qGroupSize : i32
      %r2_sidx = arith.muli %r2_group, %stride_sg : i32
      %r2_zidx = arith.muli %r2_group, %stride_zg : i32
      %r2_sp = tt.addptr %qScale_ptr, %r2_sidx : !tt.ptr<f16>, i32
      %r2_zp = tt.addptr %qZeros_ptr, %r2_zidx : !tt.ptr<f16>, i32
      %r2_spv = tt.splat %r2_sp : !tt.ptr<f16> -> {rowp}
      %r2_zpv = tt.splat %r2_zp : !tt.ptr<f16> -> {rowp}
      %r2_saddr = tt.addptr %r2_spv, %packed : {rowp}, {rowi}
      %r2_zaddr = tt.addptr %r2_zpv, %packed : {rowp}, {rowi}
      %r2_one = arith.constant dense<1.0> : {rowf}
      %r2_zero = arith.constant dense<0.0> : {rowf}
      %r2_sload = tt.load %r2_saddr, {nmask}, %r2_one : {rowp}
      %r2_zload = tt.load %r2_zaddr, {nmask}, %r2_zero : {rowp}
      %r2_sf = arith.extf %r2_sload : {rowf} to {row32}
      %r2_zf = arith.extf %r2_zload : {rowf} to {row32}
      {scale} = tt.broadcast %r2_sf : {row32} -> {wide}
      {zero} = tt.broadcast %r2_zf : {row32} -> {wide}
"""
    return text[:begin] + repl + text[end:]


def full_k(text):
    for prefix, param in (("a", "K"), ("packed", "K2")):
        splat = re.search(
            rf"(%{prefix}_\d+) = tt.splat %{param} : i32 -> (tensor<[^>]+xi32>)", text
        )
        assert splat, (prefix, param)
        pattern = rf"(%{prefix}_\d+) = arith.cmpi slt, %{prefix}_\d+, {splat[1]} : {re.escape(splat[2])}"
        text, count = re.subn(
            pattern,
            lambda m: f"{m[1]} = arith.constant dense<true> : {splat[2].replace('xi32', 'xi1')}",
            text,
        )
        assert count == 1
    return text


def a_late(text):
    begin = text.index("      %offs_k")
    end = text.index("      %byte_idx =", begin)
    part = text[begin:end]
    text = text[:begin] + text[end:]
    before_dot = re.search(r"      %acc_\d+ = tt.dot", text).start()
    return text[:before_dot] + part + text[before_dot:]


def group_pair(text, bk):
    if bk == 128:
        return text
    begin = text.index("    %acc_")
    start = text.index("\n", begin) + 1
    end = text.index("      scf.yield", start)
    finish = text.index("\n", end) + 1
    header = text[begin:start]
    body = text[start:end]
    yielded = re.search(r"scf.yield (%\w+)", text[end:finish])[1]
    incoming = re.search(r"iter_args\((%\w+) = %acc\)", header)[1]
    begin_scale = body.index("      %r2_group =")
    end_scale = body.index("      %low =", begin_scale)
    scale = re.search(r"(%scale_low_\d+) = tt.broadcast", body)[1]
    zero = re.search(r"(%zero_low_\d+) = tt.broadcast", body)[1]
    second = body[:begin_scale] + body[end_scale:]
    definitions = set(re.findall(r"^\s*(%[\w]+)\s*=", second, re.M))
    replacements = {x: x + "_pair" for x in definitions}
    replacements.update({"%k0": "%r2_second_k", incoming: yielded})
    # Shared wide scale/zero definitions remain those of the first half.
    assert scale not in definitions and zero not in definitions
    second = re.sub(r"%[A-Za-z_]\w*", lambda m: replacements.get(m[0], m[0]), second)
    second = "      %r2_second_k = arith.addi %k0, %c64_i32 : i32\n" + second
    new_header = header.replace("step %c64_i32", "step %r2_step128")
    new_yield = text[end:finish].replace(yielded, replacements[yielded])
    return (
        text[:begin]
        + "    %r2_step128 = arith.constant 128 : i32\n"
        + new_header
        + body
        + second
        + new_yield
        + text[finish:]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded", action="store_true")
    args = parser.parse_args()
    base = ROOT / "expanded" if args.expanded else ROOT
    cases = (
        json.loads((base / "manifest.json").read_text())["cases"]
        if args.expanded
        else c.manifest()["cases"]
    )
    records = []
    for case in cases:
        old = c.ROOT / "ir" / case["id"]
        folder = base / "ir" / case["id"]
        if args.expanded:
            parent = (folder / "parent.ttir").read_text()
        else:
            parent = (old / "scalar_unsigned_v1.ttir").read_text()
            c.write_once(folder / "baseline.ttir", (old / "baseline.ttir").read_text())
            c.write_once(folder / "parent.ttir", parent)
        bk, bn = case["config"][2], case["config"][1]
        n = narrow(parent, bk, bn)
        combined = full_k(n)
        variants = {
            "r2_narrow": n,
            "r2_full_k": full_k(parent),
            "r2_narrow_full_k": combined,
            "r2_a_late": a_late(combined),
            "r2_scale_cg": re.sub(
                r"(tt.load %r2_[sz]addr[^\n]+?) :", r"\1 cacheModifier = cg :", combined
            ),
            "r2_group_pair": group_pair(combined, bk),
        }
        for name, value in variants.items():
            path = folder / f"{name}.ttir"
            c.write_once(path, value)
            records.append(
                {
                    "case": case["id"],
                    "name": name,
                    "parent_sha256": c.digest(parent),
                    "sha256": c.sha(path),
                }
            )
    c.write_once(base / "candidates.json", json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
