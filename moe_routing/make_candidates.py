"""Deterministically rewrite frozen TTIR; no Python JIT candidate kernels."""

import json
import re

import common as c


def membership(text, e):
    start = text.index("    %expert_group =")
    end = text.index("    %expert_selected_37 =", start)
    replacement = (
        "    %mr_expand = tt.expand_dims %group_selected_25 {axis = 1 : i32}"
        f""" : tensor<8xi1> -> tensor<8x1xi1>
    %mr_broadcast = tt.broadcast %mr_expand : tensor<8x1xi1> -> tensor<8x{e // 8}xi1>
    %expert_selected_36 = tt.reshape %mr_broadcast : tensor<8x{e // 8}xi1> -> tensor<{e}xi1>
"""
    )
    return text[:start] + replacement + text[end:]


def argmax(text, e):
    loop = text.index("    %topk_idx:3 = scf.for")
    start = text.index('      %max_val = "tt.reduce"', loop)
    end = text.index("      %selected_raw =", start)
    replacement = f"""      %mr_argmax:2 = "tt.reduce"(%selection_scores_52, %expert_offsets) <{{axis = 0 : i32}}> ({{
      ^bb0(%mr_lhs: f32, %mr_li: i32, %mr_rhs: f32, %mr_ri: i32):
        %mr_gt = arith.cmpf ogt, %mr_lhs, %mr_rhs : f32
        %mr_eq = arith.cmpf oeq, %mr_lhs, %mr_rhs : f32
        %mr_lt = arith.cmpi slt, %mr_li, %mr_ri : i32
        %mr_tie = arith.andi %mr_eq, %mr_lt : i1
        %mr_take = arith.ori %mr_gt, %mr_tie : i1
        %mr_val = arith.select %mr_take, %mr_lhs, %mr_rhs : f32
        %mr_idx = arith.select %mr_take, %mr_li, %mr_ri : i32
        tt.reduce.return %mr_val, %mr_idx : f32, i32
      }}) : (tensor<{e}xf32>, tensor<{e}xi32>) -> (f32, i32)
"""
    tail = re.sub(r"%selected_idx(?![\w#])", "%mr_argmax#1", text[end:])
    return text[:start] + replacement + tail


def main():
    manifest = json.loads((c.ROOT / "baseline.json").read_text())
    groups = {(x["experts"], x["warps"]) for x in manifest["mapping"].values()}
    rows = []
    for e, w in sorted(groups):
        folder = c.ROOT / "ir" / c.group(e, w)
        base = (folder / "baseline_s2.ttir").read_text()
        variants = {
            "membership": membership(base, e),
            "argmax": argmax(base, e),
            "combined": argmax(membership(base, e), e),
        }
        for name, text in variants.items():
            path = folder / f"{name}_s2.ttir"
            c.immutable(path, text)
            row = dict(
                group=c.group(e, w),
                variant=name,
                input_sha256=c.sha(base),
                output_sha256=c.sha(text),
            )
            try:
                kernels = c.compile_pair(e, w, name)
                t = c.allocate(16, e)
                c.launch(kernels, t)
                row.update(
                    status="COMPILED", artifacts=c.artifacts(kernels, e, w, name)
                )
            except Exception as exc:
                row.update(status="FAILED", error=str(exc))
            print(row["group"], name, row["status"], row.get("error", ""), flush=True)
            rows.append(row)
    c.save(c.ROOT / "results/candidates.json", rows)


if __name__ == "__main__":
    main()
