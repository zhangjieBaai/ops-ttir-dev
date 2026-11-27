"""Pack stable score/index keys or unroll the frozen TTIR selection loop."""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from complex_suite.support import digest, freeze, save  # noqa: E402


def loop_parts(text):
    start = text.index(" = scf.for ")
    begin = text.rfind("\n", 0, start) + 1
    yield_begin = text.index("      scf.yield ", start)
    yield_end = text.index("\n", yield_begin)
    close_end = text.index("\n", yield_end + 1)
    return begin, yield_begin, yield_end, close_end


def packed(text):
    begin, ybegin, _, close_end = loop_parts(text)
    header = text[begin : text.index("\n", begin)]
    b = int(re.search(r"tensor<(\d+)xi32>", header)[1])
    stop = re.search(r" to (%\w+) step ", header)[1]
    initial_ids = re.search(r"iter_args\(%\w+ = %\w+, %\w+ = (%\w+)\)", header)[1]
    loopbody = text[begin:ybegin]
    bestid = re.search(r'(%best_id_\d+) = "tt.reduce"', loopbody)[1]
    store_begin = loopbody.index("      %0 =")
    store_end = loopbody.index("      %selected =")
    stores = loopbody[store_begin:store_end]
    stores = re.sub(re.escape(bestid) + r"\b", "%r2_bestid", stores)
    stores = re.sub(r"%best\b", "%r2_best", stores)
    sentinel = 8388607 << 32
    replacement = f"""    %r2_shiftv = arith.constant dense<32> : tensor<{b}xi64>
    %r2_sentinel = arith.constant dense<{sentinel}> : tensor<{b}xi64>
    %r2_shift = arith.constant 32 : i64
    %r2_intmax = arith.constant 2147483647 : i32
    %r2_zero = arith.constant 0 : i32
    %r2_all = arith.constant -1 : i32
    %r2_signbit = arith.constant -2147483648 : i32
    %r2_reverse = arith.subi %cst, {initial_ids} : tensor<{b}xi32>
    %r2_low = arith.extui %r2_reverse : tensor<{b}xi32> to tensor<{b}xi64>
    %r2_keys64 = arith.extui %init_keys : tensor<{b}xi32> to tensor<{b}xi64>
    %r2_high = arith.shli %r2_keys64, %r2_shiftv : tensor<{b}xi64>
    %r2_initial = arith.ori %r2_high, %r2_low : tensor<{b}xi64>
    %r2_loop = scf.for %step = %c0_i32 to {stop} step %c1_i32 iter_args(%r2_state = %r2_initial)
      -> (tensor<{b}xi64>) : i32 {{
      %r2_max = "tt.reduce"(%r2_state) <{{axis = 0 : i32}}> ({{
      ^bb0(%r2_left: i64, %r2_right: i64):
        %r2_value = arith.maxui %r2_left, %r2_right : i64
        tt.reduce.return %r2_value : i64
      }}) : (tensor<{b}xi64>) -> i64
      %r2_idxbits = arith.trunci %r2_max : i64 to i32
      %r2_bestid = arith.subi %r2_intmax, %r2_idxbits : i32
      %r2_highbits = arith.shrui %r2_max, %r2_shift : i64
      %r2_bestkey = arith.trunci %r2_highbits : i64 to i32
      %r2_positive = arith.cmpi slt, %r2_bestkey, %r2_zero : i32
      %r2_undo = arith.select %r2_positive, %r2_signbit, %r2_all : i32
      %r2_bestbits = arith.xori %r2_bestkey, %r2_undo : i32
      %r2_best = tt.bitcast %r2_bestbits : i32 -> f32
{stores}      %r2_broadcast = tt.splat %r2_max : i64 -> tensor<{b}xi64>
      %r2_selected = arith.cmpi eq, %r2_state, %r2_broadcast : tensor<{b}xi64>
      %r2_next = arith.select %r2_selected, %r2_sentinel, %r2_state : tensor<{b}xi1>, tensor<{b}xi64>
      scf.yield %r2_next : tensor<{b}xi64>
    }}
"""
    return text[:begin] + replacement + text[close_end + 1 :]


def unroll(text, valid_names=False):
    begin, ybegin, yend, close_end = loop_parts(text)
    header_end = text.index("{", begin) + 1
    header = text[begin:header_end]
    stop = re.search(r" to (%\w+) step ", header)[1]
    count = int(re.search(re.escape(stop) + r" = arith.constant (\d+) : i32", text)[1])
    pairs = re.findall(r"(%\w+) = (%\w+)", header.split("iter_args(")[1].split(")")[0])
    current = dict(pairs)
    body = text[header_end + 1 : ybegin]
    yielded = re.findall(r"%\w+", text[ybegin:yend].split(" : ")[0])
    defs = set(re.findall(r"(%\w+)(?::\d+)? = ", body))
    for args in re.findall(r"\^bb\w+\(([^)]+)\)", body):
        defs.update(re.findall(r"(%\w+):", args))
    iv = re.search(r"scf.for (%\w+)", header)[1]
    expanded = []
    for i in range(count):
        mappings = {
            name: (f"%r2u{i}_{name[1:]}" if valid_names else f"{name}_r2u{i}")
            for name in defs
        }
        mappings.update(current)
        mappings[iv] = f"%r2_step{i}"
        expanded.append(f"    %r2_step{i} = arith.constant {i} : i32\n")
        expanded.append(re.sub(r"%\w+", lambda m: mappings.get(m[0], m[0]), body))
        current = {
            name: mappings.get(value, value) for (name, _), value in zip(pairs, yielded)
        }
    return text[:begin] + "".join(expanded) + text[close_end + 1 :]


def static_ids(text):
    begin, ybegin, _, _ = loop_parts(text)
    header_end = text.index("\n", begin)
    header = text[begin:header_end]
    current, initial = re.search(
        r"iter_args\(%\w+ = %\w+, (%\w+) = (%\w+)\)", header
    ).groups()
    bid = re.search(r'(%best_id_\d+) = "tt.reduce"', text[header_end:ybegin])[1]
    b = int(re.search(r"tensor<(\d+)xi32>", header)[1])
    target = re.search(
        r"(%ids_\d+) = arith.select %selected_\d+, %cst, "
        + re.escape(current)
        + r"[^\n]*\n",
        text,
    )
    assert target
    text = text.replace(target[0], "")
    text = text.replace(target[1] + " :", initial + " :")
    text = text.replace(target[1] + ",", initial + ",")
    # Preserve INTMAX padding after the final finite element of a short chunk.
    line = re.search(
        re.escape(bid) + r' = "tt.reduce".*?\n      \}\)[^\n]*\n', text, re.S
    )
    assert line
    end = line.end()
    extra = f"""      %r2_infkey = arith.constant 8388607 : i32
      %r2_intmax = arith.constant 2147483647 : i32
      %r2_finite = arith.cmpi ne, %maxkey, %r2_infkey : i32
      %r2_bid = arith.select %r2_finite, {bid}, %r2_intmax : i32
"""
    tail = re.sub(re.escape(bid) + r"\b", "%r2_bid", text[end:])
    text = text[:end] + extra + tail
    # The iterated ids are constant, enabling loop-state DCE.
    assert f"tensor<{b}xi32>" in text
    return text


def main(expanded=False):
    records = []
    output_root = ROOT / "expanded" if expanded else ROOT
    manifest_path = (
        output_root / "manifest.json" if expanded else ROOT.parent / "manifest.json"
    )
    for row in json.loads(manifest_path.read_text())["records"]:
        for stage in ("local", "merge"):
            folder = output_root / "ir" / row["group"] / stage
            path = (
                folder / "parent.ttir"
                if expanded
                else ROOT.parent / "ir" / row["group"] / stage / "ordered_carry_v3.ttir"
            )
            text = path.read_text()
            freeze(folder / "parent.ttir", text)
            p64 = packed(text)
            for name, value in {
                "packed64": p64,
                "unroll_v2": unroll(text, True),
                "static_ids": static_ids(text),
                "packed_unroll_v2": unroll(p64, True),
                "static_unroll_v2": unroll(static_ids(text), True),
            }.items():
                p = folder / f"{name}.ttir"
                freeze(p, value)
                records.append(
                    {
                        "group": row["group"],
                        "stage": stage,
                        "name": name,
                        "parent_sha256": digest(path),
                        "sha256": digest(p),
                    }
                )
    save(output_root / "candidates_v2.json", records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded", action="store_true")
    main(parser.parse_args().expanded)
