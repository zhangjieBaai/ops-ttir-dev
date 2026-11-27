"""Round 6: integer interleave and exact nibble conversion, directly in TTIR."""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OP = ROOT.parent
OUT = ROOT / "r6"
sys.path.insert(0, str(OP))
import common as c  # noqa: E402


def interleave(text, case):
    bk, bn = case["config"][2], case["config"][1]
    half = bk // 2
    for suffix in ["", "_pair"] if bk == 64 else [""]:
        begin = text.index(f"      %low{suffix} =")
        end = re.search(r"      %acc_\w+ = tt.dot", text[begin:]).start() + begin
        segment = text[begin:end]
        low = re.search(r"^      (%low\w*) = arith.andi [^\n]+\n", segment, re.M)
        high_shift = re.search(
            r"^      (%high\w*) = arith.shrui [^\n]+\n", segment, re.M
        )
        high = re.search(r"^      (%high\w*) = arith.andi [^\n]+\n", segment, re.M)
        result = re.search(r"(%w_\w+) = tt.reshape", segment)[1]
        name = "%r6_i" + suffix
        h8 = f"tensor<{half}x{bn}xi8>"
        j8 = f"tensor<{half}x{bn}x2xi8>"
        t8 = f"tensor<{half}x2x{bn}xi8>"
        w8 = f"tensor<{bk}x{bn}xi8>"
        wf = f"tensor<{bk}x{bn}xf32>"
        repl = low[0] + high_shift[0] + high[0]
        repl += f"""      {name}_join = tt.join {low[1]}, {high[1]} : {h8} -> {j8}
      {name}_trans = tt.trans {name}_join {{order = array<i32: 0, 2, 1>}} : {j8} -> {t8}
      {name}_reshape = tt.reshape {name}_trans : {t8} -> {w8}
      {name}_fp = arith.uitofp {name}_reshape : {w8} to {wf}
      {name}_scale = tt.broadcast %r2_sf : tensor<1x{bn}xf32> -> {wf}
      {name}_zero = tt.broadcast %r2_zf : tensor<1x{bn}xf32> -> {wf}
      {name}_sub = arith.subf {name}_fp, {name}_zero : {wf}
      {name}_mul = arith.mulf {name}_sub, {name}_scale : {wf}
      {result} = arith.truncf {name}_mul : {wf} to tensor<{bk}x{bn}xf16>
"""
        text = text[:begin] + repl + text[end:]
    return text


def magic(text, _case):
    pattern = r"^      (%\w+) = arith.uitofp (%\w+) : tensor<(\d+x\d+)xi8> to tensor<\3xf32>[^\n]*\n"

    def replace(m):
        dest, src, shape = m[1], m[2], m[3]
        i8, i32, fp = (f"tensor<{shape}x{x}>" for x in ("i8", "i32", "f32"))
        return f"""      {dest}_u32 = arith.extui {src} : {i8} to {i32}
      {dest}_biasbits = arith.constant dense<1258291200> : {i32}
      {dest}_bits = arith.ori {dest}_u32, {dest}_biasbits : {i32}
      {dest}_biased = tt.bitcast {dest}_bits : {i32} -> {fp}
      {dest}_bias = arith.constant dense<8388608.0> : {fp}
      {dest} = arith.subf {dest}_biased, {dest}_bias : {fp}
"""

    # 0x4b000000 is FP32 2**23. Its first 16 mantissas are exact 2**23+n.
    assert 1258291200 == 0x4B000000
    result, count = re.subn(pattern, replace, text, flags=re.M)
    assert count in (1, 2, 4), count
    return result


def main():
    manifest_text = (ROOT / "r5/manifest.json").read_text()
    manifest = json.loads(manifest_text)
    records = []
    names = ["interleave_u8_v1", "magic_fp32_v1", "interleave_magic_v1"]
    c.write_once(OUT / "manifest.json", manifest_text)
    for case in manifest["cases"]:
        source = ROOT / "r5/ir" / case["id"]
        dest = OUT / "ir" / case["id"]
        parent = (source / "optimized.ttir").read_text()
        for name, filename in (
            ("baseline", "baseline"),
            ("anchor", "anchor"),
            ("parent", "optimized"),
        ):
            c.write_once(
                dest / f"{name}.ttir", (source / f"{filename}.ttir").read_text()
            )
        joined = interleave(parent, case)
        values = [joined, magic(parent, case), magic(joined, case)]
        for name, value in zip(names, values):
            path = dest / f"{name}.ttir"
            c.write_once(path, value)
            records.append(
                {
                    "case": case["id"],
                    "name": name,
                    "parent_sha256": c.digest(parent),
                    "sha256": c.sha(path),
                }
            )
    c.write_once(
        OUT / "candidates.json",
        json.dumps({"names": names, "records": records}, indent=2) + "\n",
    )


if __name__ == "__main__":
    main()
