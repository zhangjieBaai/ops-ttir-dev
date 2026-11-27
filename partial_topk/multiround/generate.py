"""Three distinct direct-TTIR selection experiments with an accepted parent chain."""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from complex_suite.support import digest, freeze, save  # noqa: E402

REDUCE = re.compile(
    r'(?P<indent> +)(?P<result>%\w+) = "tt.reduce"\((?P<input>%\w+)\) '
    r"<\{axis = 0 : i32\}> \(\{.*?\n\s+\}\) : "
    r"\(tensor<(?P<b>\d+)xi32>\) -> i32[^\n]*\n",
    re.S,
)


def hierarchical(text, groups):
    maximums = [m for m in REDUCE.finditer(text) if "arith.maxui" in m[0]]
    for iteration, m in reversed(list(enumerate(maximums))):
        start, end = m.span()
        maxname, keys, b = m["result"], m["input"], int(m["b"])
        minimum = REDUCE.search(text, end)
        assert minimum and "arith.minsi" in minimum[0]
        bestidx = minimum["result"]
        idselect = re.search(
            re.escape(minimum["input"]) + r" = arith.select %\w+, (%\w+), %\w+",
            text[end : minimum.start()],
        )
        ids = idselect[1]
        width = b // groups
        p = f"%h{iteration}_"
        body = f"""      {p}keys = tt.reshape {keys} : tensor<{b}xi32> -> tensor<{groups}x{width}xi32>
      {p}ids = tt.reshape {ids} : tensor<{b}xi32> -> tensor<{groups}x{width}xi32>
      {p}limit = arith.constant dense<2147483647> : tensor<{groups}x{width}xi32>
      {p}limitg = arith.constant dense<2147483647> : tensor<{groups}xi32>
      {p}inf = arith.constant dense<8388607> : tensor<{groups}xi32>
      {p}localmax = "tt.reduce"({p}keys) <{{axis = 1 : i32}}> ({{
      ^bb0({p}kl: i32, {p}kr: i32):
        {p}km = arith.maxui {p}kl, {p}kr : i32
        tt.reduce.return {p}km : i32
      }}) : (tensor<{groups}x{width}xi32>) -> tensor<{groups}xi32>
      {p}expand = tt.expand_dims {p}localmax {{axis = 1 : i32}} : tensor<{groups}xi32> -> tensor<{groups}x1xi32>
      {p}maxvec = tt.broadcast {p}expand : tensor<{groups}x1xi32> -> tensor<{groups}x{width}xi32>
      {p}match = arith.cmpi eq, {p}keys, {p}maxvec : tensor<{groups}x{width}xi32>
      {p}idvec = arith.select {p}match, {p}ids, {p}limit : tensor<{groups}x{width}xi1>, tensor<{groups}x{width}xi32>
      {p}localid = "tt.reduce"({p}idvec) <{{axis = 1 : i32}}> ({{
      ^bb0({p}il: i32, {p}ir: i32):
        {p}im = arith.minsi {p}il, {p}ir : i32
        tt.reduce.return {p}im : i32
      }}) : (tensor<{groups}x{width}xi32>) -> tensor<{groups}xi32>
      {p}finite = arith.cmpi ne, {p}localmax, {p}inf : tensor<{groups}xi32>
      {p}safeid = arith.select {p}finite, {p}localid, {p}limitg : tensor<{groups}xi1>, tensor<{groups}xi32>
      {p}reverse = arith.subi {p}limitg, {p}safeid : tensor<{groups}xi32>
      {p}high64 = arith.extui {p}localmax : tensor<{groups}xi32> to tensor<{groups}xi64>
      {p}low64 = arith.extui {p}reverse : tensor<{groups}xi32> to tensor<{groups}xi64>
      {p}shiftg = arith.constant dense<32> : tensor<{groups}xi64>
      {p}shifted = arith.shli {p}high64, {p}shiftg : tensor<{groups}xi64>
      {p}packed = arith.ori {p}shifted, {p}low64 : tensor<{groups}xi64>
      {p}winner = "tt.reduce"({p}packed) <{{axis = 0 : i32}}> ({{
      ^bb0({p}wl: i64, {p}wr: i64):
        {p}wm = arith.maxui {p}wl, {p}wr : i64
        tt.reduce.return {p}wm : i64
      }}) : (tensor<{groups}xi64>) -> i64
      {p}shift = arith.constant 32 : i64
      {p}limit1 = arith.constant 2147483647 : i32
      {p}high = arith.shrui {p}winner, {p}shift : i64
      {maxname} = arith.trunci {p}high : i64 to i32
      {p}low = arith.trunci {p}winner : i64 to i32
      {bestidx} = arith.subi {p}limit1, {p}low : i32
"""
        text = text[:start] + body + text[end : minimum.start()] + text[minimum.end() :]
    return text


def batch_stores(text, batch):
    pattern = re.compile(
        r"(?m)^\s*tt.store (%\w+), (%\w+) : !tt.ptr<(f32|i32)>[^\n]*\n"
    )
    stores = list(pattern.finditer(text))
    assert len(stores) % (batch * 2) == 0
    pieces = []
    cursor = 0
    for start in range(0, len(stores), 2 * batch):
        group = stores[start : start + 2 * batch]
        assert [m[3] for m in group] == ["f32", "i32"] * batch
        for m in group:
            pieces.append(text[cursor : m.start()])
            cursor = m.end()
        prefix = f"%out{start}_"
        lines = []
        for ty, subset in (("f32", group[::2]), ("i32", group[1::2])):
            values = []
            for i, m in enumerate(subset):
                name = f"{prefix}{ty}_{i}"
                lines.append(
                    f"      {name} = tt.splat {m[2]} : {ty} -> tensor<1x{ty}>\n"
                )
                values.append(name)
            width = 1
            while len(values) > 1:
                next_values = []
                for i in range(0, len(values), 2):
                    name = f"{prefix}{ty}_w{width}_p{i}"
                    lines.append(
                        f"      {name}_join = tt.join {values[i]}, {values[i+1]} : "
                        f"tensor<{width}x{ty}> -> tensor<{width}x2x{ty}>\n"
                    )
                    lines.append(
                        f"      {name}_trans = tt.trans {name}_join {{order = array<i32: 1, 0>}} : "
                        f"tensor<{width}x2x{ty}> -> tensor<2x{width}x{ty}>\n"
                    )
                    lines.append(
                        f"      {name} = tt.reshape {name}_trans : tensor<2x{width}x{ty}> -> tensor<{2*width}x{ty}>\n"
                    )
                    next_values.append(name)
                values = next_values
                width *= 2
            ptr = f"{prefix}{ty}_ptr"
            offsets = f"{prefix}{ty}_range"
            out = f"{prefix}{ty}_out"
            lines.append(
                f"      {offsets} = tt.make_range {{end = {batch} : i32, start = 0 : i32}} : tensor<{batch}xi32>\n"
            )
            lines.append(
                f"      {ptr} = tt.splat {subset[0][1]} : !tt.ptr<{ty}> -> tensor<{batch}x!tt.ptr<{ty}>>\n"
            )
            lines.append(
                f"      {out} = tt.addptr {ptr}, {offsets} : tensor<{batch}x!tt.ptr<{ty}>>, tensor<{batch}xi32>\n"
            )
            lines.append(
                f"      tt.store {out}, {values[0]} : tensor<{batch}x!tt.ptr<{ty}>>\n"
            )
        pieces.extend(lines)
    pieces.append(text[cursor:])
    return "".join(pieces)


def normalize_late(text):
    b = re.search(r"%bits = tt.bitcast.*tensor<(\d+)xf32>", text)[1]
    text = text.replace("tt.bitcast %normalized_values", "tt.bitcast %values_8")
    text = text.replace(
        "%init_keys = arith.xori %bits, %flip", "%m5_rawkeys = arith.xori %bits, %flip"
    )
    line = re.search(r"[^\n]*%m5_rawkeys =[^\n]*\n", text)
    extra = f"""    %m5_absmask = arith.constant dense<2147483647> : tensor<{b}xi32>
    %m5_zero = arith.constant dense<0> : tensor<{b}xi32>
    %m5_positivezero = arith.constant dense<-2147483648> : tensor<{b}xi32>
    %m5_absbits = arith.andi %bits, %m5_absmask : tensor<{b}xi32>
    %m5_iszero = arith.cmpi eq, %m5_absbits, %m5_zero : tensor<{b}xi32>
    %init_keys = arith.select %m5_iszero, %m5_positivezero, %m5_rawkeys : tensor<{b}xi1>, tensor<{b}xi32>
"""
    return text[: line.end()] + extra + text[line.end() :]


def contract(text, shape, stage):
    full = shape[1] % 1024 == 0
    # Full local chunks contain1024 finite elements; merge always has at leastK.
    if full or stage == "merge":
        pattern = re.compile(
            r"(?m)^(\s*)(%r2u\d+_r2_bid) = arith.select %\w+, (%\w+), %\w+ : i32[^\n]*\n"
        )
        for m in reversed(list(pattern.finditer(text))):
            text = text[: m.start()] + text[m.end() :]
            text = re.sub(re.escape(m[2]) + r"\b", m[3], text)
    if full:
        if stage == "merge":
            text = re.sub(
                r"%chunk = tt.get_program_id y : i32[^\n]*",
                "%chunk = arith.constant 0 : i32",
                text,
            )
        # Every input lane is valid for the power-of-two full stages in this matrix.
        text = re.sub(
            r"(tt.load %\w+), %mask_\d+, %\w+( : tensor<[^\n]+)", r"\1\2", text
        )
        pattern = re.search(
            r"(\s*)(%ids(?:_\d+)?) = arith.select %mask_\d+, (%\w+), %cst :[^\n]*\n",
            text,
        )
        if pattern:
            text = text[: pattern.start()] + text[pattern.end() :]
            text = re.sub(re.escape(pattern[2]) + r"\b", pattern[3], text)
    return text


def main(round_id):
    manifest = json.loads((ROOT.parent / "round2/expanded/manifest.json").read_text())
    rr = ROOT / f"r{round_id}"
    records = []
    for item in manifest["records"]:
        for stage in ("local", "merge"):
            group = item["group"]
            anchor = (
                ROOT.parent / "round2/expanded/ir" / group / stage / "optimized.ttir"
            )
            parent = (
                anchor
                if round_id == 3
                else ROOT / f"r{round_id-1}/ir" / group / stage / "optimized.ttir"
            )
            text = parent.read_text()
            folder = rr / "ir" / group / stage
            freeze(folder / "parent.ttir", text)
            if round_id == 3:
                variants = {
                    "hier4": hierarchical(text, 4),
                    "hier8": hierarchical(text, 8),
                }
            elif round_id == 4:
                variants = {f"store{b}": batch_stores(text, b) for b in (2, 4, 8)}
            else:
                simplified = contract(text, item["shape"], stage)
                variants = {
                    "late_zero": normalize_late(text),
                    "contract": simplified,
                    "contract_late_zero": normalize_late(simplified),
                }
            for name, value in variants.items():
                p = folder / f"{name}.ttir"
                freeze(p, value)
                records.append(
                    {
                        "group": group,
                        "stage": stage,
                        "name": name,
                        "sha256": digest(p),
                        "parent_sha256": digest(parent),
                        "anchor_sha256": digest(anchor),
                    }
                )
    save(rr / "candidates.json", records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, choices=[3, 4, 5], required=True)
    main(parser.parse_args().round)
