"""Three sequential immutable rounds, direct TTIR rewrites only."""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO))
from complex_suite.support import digest, freeze, save  # noqa: E402


def unroll(text, scalar_indices=False, ordered_concat=False):
    begin = text.index("    %topk_idx:2 =")
    header_end = text.index("\n", begin)
    end = text.index("    %r2_raw_weights =", header_end)
    loop = text[header_end + 1 : end]
    body = loop[: loop.index("      scf.yield")]
    definitions = set(re.findall(r"(%\w+)(?::\d+)? =", body))
    definitions.update(re.findall(r"(%\w+): (?:i\d+|f\d+)", body))
    states = ["%r2_keys", "%cst_1"]
    parts, indices = [], []
    for i in range(8):
        mapping = {name: f"%mr3_u{i}_{name[1:]}" for name in definitions}
        mapping.update(dict(zip(["%selection_scores_52", "%topk_idx_54"], states)))
        mapping["%i"] = f"%mr3_step{i}"
        parts.append(f"    %mr3_step{i} = arith.constant {i} : i32\n")
        item = re.sub(r"%\w+", lambda m: mapping.get(m[0], m[0]), body)
        if scalar_indices:
            item = re.sub(
                r"^.*%mr3_u\d+_(?:topk_vals_63|topk_vals_64|topk_idx_67|topk_idx_68) =[^\n]*\n",
                "",
                item,
                flags=re.M,
            )
        parts.append(item)
        states = [mapping["%selection_scores_71"], mapping["%topk_idx_68"]]
        indices.append(mapping["%selected_idx"])
    final = states[1]
    if scalar_indices:
        values = []
        for i, idx in enumerate(indices):
            parts.append(f"    %mr3_index{i} = tt.splat {idx} : i32 -> tensor<1xi32>\n")
            values.append(f"%mr3_index{i}")
        width = 1
        while len(values) > 1:
            new = []
            for j in range(0, len(values), 2):
                name = f"%mr3_cat{width}_{j}"
                if ordered_concat:
                    parts.append(
                        f"    {name}_join = tt.join {values[j]}, {values[j+1]} "
                        f": tensor<{width}xi32> -> tensor<{width}x2xi32>\n"
                        f"    {name}_trans = tt.trans {name}_join {{order = array<i32: 1, 0>}} "
                        f": tensor<{width}x2xi32> -> tensor<2x{width}xi32>\n"
                        f"    {name} = tt.reshape {name}_trans : tensor<2x{width}xi32> -> tensor<{2*width}xi32>\n"
                    )
                else:
                    parts.append(
                        f"    {name} = tt.cat {values[j]}, {values[j+1]} "
                        f": tensor<{width}xi32> -> tensor<{2*width}xi32>\n"
                    )
                new.append(name)
            values, width = new, width * 2
        final = values[0]
    tail = text[end:].replace("%topk_idx#1", final)
    return text[:begin] + "".join(parts) + tail


def ordered_top2(text, size):
    fty, ity = f"tensor<{size}xf32>", f"tensor<{size}xi32>"
    start = text.index('    %max1 = "tt.reduce"')
    end = text.index("    %group_scores_offset =", start)
    original = text[start:end]
    init = f"""    %mr4_zero = arith.constant dense<0.0> : {fty}
    %mr4_iszero = arith.cmpf oeq, %x_14, %mr4_zero : {fty}
    %mr4_normal = arith.select %mr4_iszero, %mr4_zero, %x_14 : tensor<{size}xi1>, {fty}
    %mr4_bits = tt.bitcast %mr4_normal : {fty} -> {ity}
    %mr4_shift = arith.constant dense<31> : {ity}
    %mr4_signbit = arith.constant dense<-2147483648> : {ity}
    %mr4_sign = arith.shrsi %mr4_bits, %mr4_shift : {ity}
    %mr4_flip = arith.ori %mr4_sign, %mr4_signbit : {ity}
    %mr4_keys = arith.xori %mr4_bits, %mr4_flip : {ity}
    %mr4_removed = arith.constant dense<8388607> : {ity}
    %mr4_scalarzero = arith.constant 0 : i32
    %mr4_scalarall = arith.constant -1 : i32
    %mr4_scalarsign = arith.constant -2147483648 : i32
"""
    first_end = original.index("    %is_max1 =")
    first = f"""    %mr4_max1 = "tt.reduce"(%mr4_keys) <{{axis = 0 : i32}}> ({{
    ^bb0(%l: i32, %r: i32):
      %v = arith.maxui %l, %r : i32
      tt.reduce.return %v : i32
    }}) : ({ity}) -> i32
"""
    middle = original[first_end : original.index('    %max2 = "tt.reduce"')]
    middle = middle.replace(
        "tt.splat %max1 : f32 -> " + fty, "tt.splat %mr4_max1 : i32 -> " + ity
    )
    middle = middle.replace(
        "arith.cmpf oeq, %x_14, %is_max1 : " + fty,
        "arith.cmpi eq, %mr4_keys, %is_max1 : " + ity,
    )
    middle = middle.replace(
        f"%cst_0, %x_14 : tensor<{size}xi1>, {fty}",
        f"%mr4_removed, %mr4_keys : tensor<{size}xi1>, {ity}",
    )
    second = f"""    %mr4_max2 = "tt.reduce"(%x2_21) <{{axis = 0 : i32}}> ({{
    ^bb0(%l: i32, %r: i32):
      %v = arith.maxui %l, %r : i32
      tt.reduce.return %v : i32
    }}) : ({ity}) -> i32
"""
    decode = ""
    for i in (1, 2):
        decode += f"""    %mr4_pos{i} = arith.cmpi slt, %mr4_max{i}, %mr4_scalarzero : i32
    %mr4_undo{i} = arith.select %mr4_pos{i}, %mr4_scalarsign, %mr4_scalarall : i32
    %mr4_decoded{i} = arith.xori %mr4_max{i}, %mr4_undo{i} : i32
    %max{i} = tt.bitcast %mr4_decoded{i} : i32 -> f32
"""
    return text[:start] + init + first + middle + second + decode + text[end:]


def pair_top2(text, size):
    start = text.index('    %max1 = "tt.reduce"')
    end = text.index("    %group_scores_offset =", start)
    region = f"""    %mr4_pair:2 = "tt.reduce"(%x_14, %cst_0) <{{axis = 0 : i32}}> ({{
    ^bb0(%a: f32, %b: f32, %c: f32, %d: f32):
      %hi = arith.maxnumf %a, %c : f32
      %cross = arith.minnumf %a, %c : f32
      %lo2 = arith.maxnumf %b, %d : f32
      %lo = arith.maxnumf %cross, %lo2 : f32
      tt.reduce.return %hi, %lo : f32, f32
    }}) : (tensor<{size}xf32>, tensor<{size}xf32>) -> (f32, f32)
"""
    tail = re.sub(r"%max1\b", "%mr4_pair#0", text[end:])
    tail = re.sub(r"%max2\b", "%mr4_pair#1", tail)
    return text[:start] + region + tail


def specialize(text, e, stage, floating=False):
    body = text.index("\n", text.index("  tt.func public")) + 1
    vals = {
        "num_experts_per_group": e // 8,
        "n_group": 8,
        "stride_scores_token": e,
        "stride_group_scores_token": 8,
    }
    if stage == 2:
        vals.update(topk_group=4, topk=8, num_experts=e, stride_out_token=8)
    mapping = {"%" + k: "%mr5_const_" + k for k in vals}
    declarations = "".join(
        f"    {mapping['%' + k]} = arith.constant {v} : i32\n" for k, v in vals.items()
    )
    if floating and stage == 2:
        mapping["%routed_scaling_factor"] = "%mr5_scale"
        declarations += "    %mr5_scale = arith.constant 1.0 : f32\n"
    tail = re.sub(r"%\w+", lambda m: mapping.get(m[0], m[0]), text[body:])
    return text[:body] + declarations + tail


def main(round_name):
    target = ROOT / round_name
    assert round_name in ("r3", "r4", "r5")
    previous = (
        None if round_name == "r3" else ROOT / ("r3" if round_name == "r4" else "r4")
    )
    if previous:
        old_selection = json.loads((previous / "selection.json").read_text())
        assert old_selection["qualified"]
    records = []
    for e in (256, 512):
        original = ROOT.parent / "round2/ir" / f"e{e}_w4"
        folder = target / "ir" / f"e{e}_w4"
        for stage in (1, 2):
            freeze(
                folder / f"baseline_s{stage}.ttir",
                (original / f"baseline_s{stage}.ttir").read_text(),
            )
            anchor_path = original / (
                "baseline_s1.ttir" if stage == 1 else "r2_rank_ordered_gather_s2.ttir"
            )
            freeze(folder / f"anchor_s{stage}.ttir", anchor_path.read_text())
            parent_path = (
                anchor_path
                if previous is None
                else previous / "ir" / f"e{e}_w4" / f"optimized_s{stage}.ttir"
            )
            freeze(folder / f"parent_s{stage}.ttir", parent_path.read_text())
        s1 = (folder / "parent_s1.ttir").read_text()
        s2 = (folder / "parent_s2.ttir").read_text()
        if round_name == "r3":
            efficient = re.sub(
                r'("tt.gather"[^\n]+<\{axis = 0 : i32)(\}>\s*:)',
                r"\1, efficient_layout = true\2",
                s2,
            )
            assert efficient != s2
            candidates = {
                "gather_efficient": (s1, efficient),
                "gather_unit_v2": (
                    s1,
                    efficient.replace("efficient_layout = true", "efficient_layout"),
                ),
                "unroll_experts": (s1, unroll(s2)),
                "scalar_indices": (s1, unroll(s2, True)),
                "scalar_indices_v2": (s1, unroll(s2, True, True)),
            }
        elif round_name == "r4":
            pair_fixed = pair_top2(s1, e // 8)
            start = pair_fixed.index('    %mr4_pair:2 = "tt.reduce"')
            end = pair_fixed.index("    %group_scores_offset =", start)
            region = pair_fixed[start:end]
            for old, new in [
                ("%a", "%mr4_arg_a"),
                ("%b", "%mr4_arg_b"),
                ("%c", "%mr4_arg_c"),
                ("%d", "%mr4_arg_d"),
            ]:
                region = re.sub(re.escape(old) + r"\b", new, region)
            pair_fixed = pair_fixed[:start] + region + pair_fixed[end:]
            candidates = {
                "ordered_top2": (ordered_top2(s1, e // 8), s2),
                "pair_top2": (pair_top2(s1, e // 8), s2),
                "pair_top2_v2": (pair_fixed, s2),
            }
        else:
            if "%mr4_pair:2" not in s1:
                pair_fixed = pair_top2(s1, e // 8)
                start = pair_fixed.index('    %mr4_pair:2 = "tt.reduce"')
                end = pair_fixed.index("    %group_scores_offset =", start)
                region = pair_fixed[start:end]
                for old, new in [
                    ("%a", "%mr4_arg_a"),
                    ("%b", "%mr4_arg_b"),
                    ("%c", "%mr4_arg_c"),
                    ("%d", "%mr4_arg_d"),
                ]:
                    region = re.sub(re.escape(old) + r"\b", new, region)
                pair_fixed = pair_fixed[:start] + region + pair_fixed[end:]
                final_name = "pair_integer_constants"
                final_stages = (specialize(pair_fixed, e, 1), specialize(s2, e, 2))
            else:
                final_name = "all_constants"
                final_stages = (specialize(s1, e, 1), specialize(s2, e, 2, True))
            candidates = {
                "s1_constants": (specialize(s1, e, 1), s2),
                "integer_constants": (specialize(s1, e, 1), specialize(s2, e, 2)),
                final_name: final_stages,
            }
        for name, stages in candidates.items():
            for stage, text in enumerate(stages, 1):
                path = folder / f"{name}_s{stage}.ttir"
                freeze(path, text)
                records.append(
                    {
                        "experts": e,
                        "candidate": name,
                        "stage": stage,
                        "sha256": digest(path),
                        "parent_sha256": digest(folder / f"parent_s{stage}.ttir"),
                    }
                )
    save(target / "candidates.json", records)
    source_manifest = ROOT.parent / "round2/manifest30.json"
    freeze(target / "manifest30.json", source_manifest.read_text())
    save(
        target / "lineage.json",
        {
            "round": round_name,
            "previous_round": str(previous) if previous else "round2",
            "manifest_sha256": digest(source_manifest),
            "candidates": list(candidates),
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("round", choices=["r3", "r4", "r5"])
    main(p.parse_args().round)
