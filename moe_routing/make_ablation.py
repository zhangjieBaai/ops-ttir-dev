"""Freeze add-one and leave-one-out TTIR controls for the accepted r6 route."""

import importlib.util
import json
import re
import sys
from pathlib import Path

OP = Path(__file__).resolve().parent
REPO = OP.parent
sys.path.insert(0, str(REPO))
from complex_suite.support import digest, freeze  # noqa: E402


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


r2 = module("moe_ablation_r2", OP / "round2/make_candidates.py")
mr = module("moe_ablation_mr", OP / "multiround/generate.py")
r6 = module("moe_ablation_r6", OP / "multiround/r6_generate.py")
ROOT = OP / "ablation_v1"


def pair(text, size):
    text = mr.pair_top2(text, size)
    begin = text.index('    %mr4_pair:2 = "tt.reduce"')
    end = text.index("    %group_scores_offset =", begin)
    body = text[begin:end]
    for name in "abcd":
        body = re.sub(r"%" + name + r"\b", "%mr4_arg_" + name, body)
    return text[:begin] + body + text[end:]


def sequential_top2(text, experts):
    """Undo the pair reducer inside the final batched 2-D dataflow."""
    size, batch = experts // 8, 4
    ft, it = f"tensor<{batch}x{size}xf32>", f"tensor<{batch}x{size}xi32>"
    rt, ri = f"tensor<{batch}xf32>", f"tensor<{batch}xi32>"
    bt = f"tensor<{batch}x{size}xi1>"
    begin = text.index('      %r6_pair:2 = "tt.reduce"')
    end = text.index("      %r6_out_base =", begin)
    body = f"""      %abl_one = arith.constant dense<1> : {ri}
      %abl_max1 = "tt.reduce"(%r6_scores) <{{axis = 1 : i32}}> ({{
      ^bb0(%abl_a: f32, %abl_b: f32):
        %abl_m = arith.maxnumf %abl_a, %abl_b : f32
        tt.reduce.return %abl_m : f32
      }}) : ({ft}) -> {rt}
      %abl_m1e = tt.expand_dims %abl_max1 {{axis = 1 : i32}} : {rt} -> tensor<{batch}x1xf32>
      %abl_m1b = tt.broadcast %abl_m1e : tensor<{batch}x1xf32> -> {ft}
      %abl_equal = arith.cmpf oeq, %r6_scores, %abl_m1b : {ft}
      %abl_counts = arith.extui %abl_equal : {bt} to {it}
      %abl_count = "tt.reduce"(%abl_counts) <{{axis = 1 : i32}}> ({{
      ^bb0(%abl_i: i32, %abl_j: i32):
        %abl_sum = arith.addi %abl_i, %abl_j : i32
        tt.reduce.return %abl_sum : i32
      }}) : ({it}) -> {ri}
      %abl_unique = arith.cmpi eq, %abl_count, %abl_one : {ri}
      %abl_ue = tt.expand_dims %abl_unique {{axis = 1 : i32}} : tensor<{batch}xi1> -> tensor<{batch}x1xi1>
      %abl_ub = tt.broadcast %abl_ue : tensor<{batch}x1xi1> -> {bt}
      %abl_remove = arith.andi %abl_equal, %abl_ub : {bt}
      %abl_remaining = arith.select %abl_remove, %r6_neginf, %r6_scores : {bt}, {ft}
      %abl_max2 = "tt.reduce"(%abl_remaining) <{{axis = 1 : i32}}> ({{
      ^bb0(%abl_c: f32, %abl_d: f32):
        %abl_n = arith.maxnumf %abl_c, %abl_d : f32
        tt.reduce.return %abl_n : f32
      }}) : ({ft}) -> {rt}
      %r6_result = arith.addf %abl_max1, %abl_max2 : {rt}
"""
    return text[:begin] + body + text[end:]


def runtime_batch(text, experts):
    """Restore runtime integer operands while preserving the fixed batch/tile."""
    text = text.replace(
        "%r6_eight = arith.constant 8 : i32",
        "%r6_eight = arith.addi %n_group, %r6_zero : i32",
    )
    text = text.replace(
        f"%r6_size = arith.constant {experts // 8} : i32",
        "%r6_size = arith.addi %num_experts_per_group, %r6_zero : i32",
    )
    text = text.replace(
        f"%r6_experts = arith.constant {experts} : i32",
        "%r6_experts = arith.addi %stride_scores_token, %r6_zero : i32",
    )
    text = text.replace(
        f"%r6_size_vec = arith.constant dense<{experts // 8}> : tensor<4xi32>",
        "%r6_size_vec = tt.splat %num_experts_per_group : i32 -> tensor<4xi32>",
    )
    text = text.replace(
        "      %r6_out_base = tt.addptr %group_scores_ptr, %r6_pid : !tt.ptr<f32>, i32",
        "      %abl_token_out = arith.muli %r6_token, %stride_group_scores_token : i32\n"
        "      %abl_group_out = arith.addi %abl_token_out, %r6_group : i32\n"
        "      %r6_out_base = tt.addptr %group_scores_ptr, %abl_group_out : !tt.ptr<f32>, i32",
    )
    return text


def main():
    entries = []
    for e in (256, 512):
        source = OP / "ir" / f"e{e}_w4"
        b1 = (source / "baseline_s1.ttir").read_text()
        b2 = (source / "baseline_s2.ttir").read_text()
        member = (source / "membership_s2.ttir").read_text()
        ordered = r2.ordered(member, e)
        gathered = r2.gather(ordered, e)
        ranked = r2.rank(gathered, e)
        paired = pair(b1, e // 8)
        s1, s2 = mr.specialize(paired, e, 1), mr.specialize(ranked, e, 2)
        final1 = r6.batch_groups(s1, e, 4)
        final_dir = OP / "multiround/r6/ir" / f"e{e}_w4"
        assert final1 == (final_dir / "optimized_s1.ttir").read_text()
        assert s2 == (final_dir / "optimized_s2.ttir").read_text()
        variants = {
            "baseline": (b1, b2),
            "add_membership": (b1, member),
            "add_ordered": (b1, ordered),
            "add_gather": (b1, gathered),
            "add_rank": (b1, ranked),
            "add_pair": (paired, ranked),
            "add_constants": (s1, s2),
            "final": (final1, s2),
            "drop_membership": (
                final1,
                mr.specialize(r2.rank(r2.gather(r2.ordered(b2, e), e), e), e, 2),
            ),
            "drop_ordered": (
                final1,
                mr.specialize(r2.rank(r2.gather(member, e), e), e, 2),
            ),
            "drop_gather": (final1, mr.specialize(r2.rank(ordered, e), e, 2)),
            "drop_rank": (final1, mr.specialize(gathered, e, 2)),
            "drop_pair": (sequential_top2(final1, e), s2),
            "drop_constants": (runtime_batch(final1, e), ranked),
            "stage1_only": (final1, b2),
            "stage2_only": (b1, s2),
        }
        for name, stages in variants.items():
            for stage, text in enumerate(stages, 1):
                path = ROOT / "ir" / f"e{e}_w4" / f"{name}_s{stage}.ttir"
                freeze(path, text)
                entries.append(
                    dict(
                        experts=e,
                        variant=name,
                        stage=stage,
                        ir=str(path.relative_to(OP)),
                        sha256=digest(path),
                    )
                )
    freeze(ROOT / "manifest30.json", (OP / "multiround/r6/manifest30.json").read_text())
    freeze(
        ROOT / "manifest.json",
        json.dumps(
            dict(
                variants=list(variants),
                entries=entries,
                final_selection_sha256=digest(OP / "multiround/r6/selection.json"),
            ),
            indent=2,
        )
        + "\n",
    )


if __name__ == "__main__":
    main()
