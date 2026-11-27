"""Direct, immutable rewrites of the accepted MoE TTIR parent."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
sys.path.insert(0, str(PARENT.parent))
from complex_suite.support import digest, freeze, save  # noqa: E402


def ordered(text, e):
    fty, ity = f"tensor<{e}xf32>", f"tensor<{e}xi32>"
    init = f"""    %r2_zero = arith.constant dense<0.0> : {fty}
    %r2_iszero = arith.cmpf oeq, %selection_scores, %r2_zero : {fty}
    %r2_scores = arith.select %r2_iszero, %r2_zero, %selection_scores : tensor<{e}xi1>, {fty}
    %r2_bits = tt.bitcast %r2_scores : {fty} -> {ity}
    %r2_31 = arith.constant dense<31> : {ity}
    %r2_signbit = arith.constant dense<-2147483648> : {ity}
    %r2_sign = arith.shrsi %r2_bits, %r2_31 : {ity}
    %r2_flip = arith.ori %r2_sign, %r2_signbit : {ity}
    %r2_keys = arith.xori %r2_bits, %r2_flip : {ity}
    %r2_removed = arith.constant dense<8388607> : {ity}
"""
    begin = text.index("    %topk_idx:3 =")
    end = text.index("    %topk_sum =", begin)
    region = text[begin:end]
    region = region.replace(
        "%selection_scores_52 = %selection_scores", "%selection_scores_52 = %r2_keys"
    )
    region = region.replace(f"-> ({fty},", f"-> ({ity},")
    rbegin = region.index('      %max_val = "tt.reduce"')
    rend = region.index("      %candidate_idx =", rbegin)
    maximum = f"""      %max_val = "tt.reduce"(%selection_scores_52) <{{axis = 0 : i32}}> ({{
      ^bb0(%r2_l: i32, %r2_r: i32):
        %r2_m = arith.maxui %r2_l, %r2_r : i32
        tt.reduce.return %r2_m : i32
      }}) : ({ity}) -> i32
      %is_max = tt.splat %max_val : i32 -> {ity}
      %is_max_55 = arith.cmpi eq, %selection_scores_52, %is_max : {ity}
"""
    region = region[:rbegin] + maximum + region[rend:]
    region = region.replace(
        f"%selection_scores_70, %cst_6, %selection_scores_52 : tensor<{e}xi1>, {fty}",
        f"%selection_scores_70, %r2_removed, %selection_scores_52 : tensor<{e}xi1>, {ity}",
    )
    region = region.replace(
        f"scf.yield %selection_scores_71, %topk_vals_66, %topk_idx_68 : {fty},",
        f"scf.yield %selection_scores_71, %topk_vals_66, %topk_idx_68 : {ity},",
    )
    return text[:begin] + init + region + text[end:]


def gather(text, e):
    begin = text.index("      %selected_raw =")
    end = text.index("      %topk_vals_63 =", begin)
    text = text[:begin] + text[end:]
    text = re.sub(r"^      %topk_vals_6[56] =[^\n]+\n", "", text, flags=re.M)
    text = text.replace("%topk_idx:3 =", "%topk_idx:2 =")
    text = text.replace(", %topk_vals_53 = %cst_2", "")
    text = text.replace(", tensor<8xf32>, tensor<8xi32>)", ", tensor<8xi32>)")
    text = text.replace(
        "scf.yield %selection_scores_71, %topk_vals_66, %topk_idx_68",
        "scf.yield %selection_scores_71, %topk_idx_68",
    )
    text = text.replace(", tensor<8xf32>, tensor<8xi32> loc", ", tensor<8xi32> loc")
    text = text.replace("%topk_idx#1", "%r2_raw_weights").replace(
        "%topk_idx#2", "%topk_idx#1"
    )
    insertion = (
        '    %r2_raw_weights = "tt.gather"(%scored_f32_46, %topk_idx#1) '
        f"<{{axis = 0 : i32}}> : (tensor<{e}xf32>, tensor<8xi32>) -> tensor<8xf32>\n"
    )
    text = text.replace("    %topk_sum =", insertion + "    %topk_sum =", 1)
    return text


def rank(text, e):
    begin = text.index("    %target_num_min =")
    end = text.index("    %expert_offsets =", begin)
    s = """    %r2_row_score = tt.expand_dims %group_scores_f32_18 {axis = 1 : i32} : tensor<8xf32> -> tensor<8x1xf32>
    %r2_col_score = tt.expand_dims %group_scores_f32_18 {axis = 0 : i32} : tensor<8xf32> -> tensor<1x8xf32>
    %r2_row_scores = tt.broadcast %r2_row_score : tensor<8x1xf32> -> tensor<8x8xf32>
    %r2_col_scores = tt.broadcast %r2_col_score : tensor<1x8xf32> -> tensor<8x8xf32>
    %r2_gt = arith.cmpf ogt, %r2_col_scores, %r2_row_scores : tensor<8x8xf32>
    %r2_eq = arith.cmpf oeq, %r2_col_scores, %r2_row_scores : tensor<8x8xf32>
    %r2_row_id = tt.expand_dims %group_offsets {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %r2_col_id = tt.expand_dims %group_offsets {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %r2_row_ids = tt.broadcast %r2_row_id : tensor<8x1xi32> -> tensor<8x8xi32>
    %r2_col_ids = tt.broadcast %r2_col_id : tensor<1x8xi32> -> tensor<8x8xi32>
    %r2_prior = arith.cmpi slt, %r2_col_ids, %r2_row_ids : tensor<8x8xi32>
    %r2_tie = arith.andi %r2_eq, %r2_prior : tensor<8x8xi1>
    %r2_better = arith.ori %r2_gt, %r2_tie : tensor<8x8xi1>
    %r2_counts = arith.extui %r2_better : tensor<8x8xi1> to tensor<8x8xi32>
    %r2_ranks = "tt.reduce"(%r2_counts) <{axis = 1 : i32}> ({
    ^bb0(%r2_l: i32, %r2_r: i32):
      %r2_s = arith.addi %r2_l, %r2_r : i32
      tt.reduce.return %r2_s : i32
    }) : (tensor<8x8xi32>) -> tensor<8xi32>
    %r2_num_groups = tt.splat %topk_group : i32 -> tensor<8xi32>
    %r2_selected = arith.cmpi slt, %r2_ranks, %r2_num_groups : tensor<8xi32>
    %group_selected_25 = arith.andi %r2_selected, %valid_group_11 : tensor<8xi1>
"""
    return text[:begin] + s + text[end:]


def unroll(text, e):
    begin = text.index("    %topk_idx:3 =")
    header_end = text.index("\n", begin)
    end = text.index("    %topk_sum =", header_end)
    loop = text[header_end + 1 : end]
    yield_at = loop.index("      scf.yield")
    body = loop[:yield_at]
    definitions = set(re.findall(r"(%[\w]+)(?::\d+)? =", body))
    definitions.update(re.findall(r"(%[\w]+): (?:i\d+|f\d+)", body))
    states = ["%selection_scores", "%cst_2", "%cst_1"]
    parts = []
    for i in range(8):
        mapping = {name: f"%r2_u{i}_{name[1:]}" for name in definitions}
        mapping.update(
            dict(zip(["%selection_scores_52", "%topk_vals_53", "%topk_idx_54"], states))
        )
        mapping["%i"] = f"%r2_step{i}"
        parts.append(f"    %r2_step{i} = arith.constant {i} : i32\n")
        parts.append(re.sub(r"%[\w]+", lambda m: mapping.get(m[0], m[0]), body))
        states = [
            mapping[x]
            for x in ("%selection_scores_71", "%topk_vals_66", "%topk_idx_68")
        ]
    tail = (
        text[end:].replace("%topk_idx#1", states[1]).replace("%topk_idx#2", states[2])
    )
    return text[:begin] + "".join(parts) + tail


def main():
    records = []
    for e in (256, 512):
        old = PARENT / "ir" / f"e{e}_w4"
        new = ROOT / "ir" / f"e{e}_w4"
        for name in ("baseline_s1", "baseline_s2"):
            freeze(new / f"{name}.ttir", (old / f"{name}.ttir").read_text())
        parent = (old / "membership_s2.ttir").read_text()
        freeze(new / "parent_s2.ttir", parent)
        variants = {
            "r2_ordered": ordered(parent, e),
            "r2_gather": gather(parent, e),
            "r2_rank": rank(parent, e),
            "r2_ordered_gather": gather(ordered(parent, e), e),
            "r2_rank_ordered_gather": rank(gather(ordered(parent, e), e), e),
            "r2_unroll": unroll(parent, e),
        }
        for name, text in variants.items():
            path = new / f"{name}_s2.ttir"
            freeze(path, text)
            records.append(
                {
                    "experts": e,
                    "name": name,
                    "parent_sha256": digest(old / "membership_s2.ttir"),
                    "ttir_sha256": digest(path),
                }
            )
    save(ROOT / "candidates.json", records)


if __name__ == "__main__":
    main()
