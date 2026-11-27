"""Batch independent group top2 rows directly in the accepted stage1 TTIR."""

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parents[1]))
from complex_suite.support import digest, freeze  # noqa: E402


def batch_groups(text, experts, batch):
    size = experts // 8
    begin = text.index("\n", text.index("  tt.func public")) + 1
    end = text.index("    tt.return", begin)
    body = f"""    %r6_zero = arith.constant 0 : i32
    %r6_eight = arith.constant 8 : i32
    %r6_batch = arith.constant {batch} : i32
    %r6_size = arith.constant {size} : i32
    %r6_experts = arith.constant {experts} : i32
    %r6_pid = tt.get_program_id x : i32
    %r6_rest = arith.remui %r6_pid, %r6_batch : i32
    %r6_active = arith.cmpi eq, %r6_rest, %r6_zero : i32
    scf.if %r6_active {{
      %r6_token = arith.divui %r6_pid, %r6_eight : i32
      %r6_group = arith.remui %r6_pid, %r6_eight : i32
      %r6_token_offset = arith.muli %r6_token, %r6_experts : i32
      %r6_group_offset = arith.muli %r6_group, %r6_size : i32
      %r6_score_offset = arith.addi %r6_token_offset, %r6_group_offset : i32
      %r6_rows = tt.make_range {{end = {batch} : i32, start = 0 : i32}} : tensor<{batch}xi32>
      %r6_lanes = tt.make_range {{end = {size} : i32, start = 0 : i32}} : tensor<{size}xi32>
      %r6_size_vec = arith.constant dense<{size}> : tensor<{batch}xi32>
      %r6_row_offsets = arith.muli %r6_rows, %r6_size_vec : tensor<{batch}xi32>
      %r6_row_exp = tt.expand_dims %r6_row_offsets {{axis = 1 : i32}} : tensor<{batch}xi32> -> tensor<{batch}x1xi32>
      %r6_lane_exp = tt.expand_dims %r6_lanes {{axis = 0 : i32}} : tensor<{size}xi32> -> tensor<1x{size}xi32>
      %r6_row_b = tt.broadcast %r6_row_exp : tensor<{batch}x1xi32> -> tensor<{batch}x{size}xi32>
      %r6_lane_b = tt.broadcast %r6_lane_exp : tensor<1x{size}xi32> -> tensor<{batch}x{size}xi32>
      %r6_offsets = arith.addi %r6_row_b, %r6_lane_b : tensor<{batch}x{size}xi32>
      %r6_score_base = tt.addptr %scores_ptr, %r6_score_offset : !tt.ptr<f32>, i32
      %r6_bias_base = tt.addptr %bias_ptr, %r6_group_offset : !tt.ptr<f32>, i32
      %r6_score_splat = tt.splat %r6_score_base : !tt.ptr<f32> -> tensor<{batch}x{size}x!tt.ptr<f32>>
      %r6_bias_splat = tt.splat %r6_bias_base : !tt.ptr<f32> -> tensor<{batch}x{size}x!tt.ptr<f32>>
      %r6_score_ptrs = tt.addptr %r6_score_splat, %r6_offsets
        : tensor<{batch}x{size}x!tt.ptr<f32>>, tensor<{batch}x{size}xi32>
      %r6_bias_ptrs = tt.addptr %r6_bias_splat, %r6_offsets
        : tensor<{batch}x{size}x!tt.ptr<f32>>, tensor<{batch}x{size}xi32>
      %r6_x = tt.load %r6_score_ptrs : tensor<{batch}x{size}x!tt.ptr<f32>>
      %r6_bias = tt.load %r6_bias_ptrs : tensor<{batch}x{size}x!tt.ptr<f32>>
      %r6_half = arith.constant dense<5.000000e-01> : tensor<{batch}x{size}xf32>
      %r6_neginf = arith.constant dense<0xFF800000> : tensor<{batch}x{size}xf32>
      %r6_xhalf = arith.mulf %r6_x, %r6_half : tensor<{batch}x{size}xf32>
      %r6_tanh = tt.extern_elementwise %r6_xhalf
        {{libname = "", libpath = "", pure = true, symbol = "__nv_tanhf"}}
        : (tensor<{batch}x{size}xf32>) -> tensor<{batch}x{size}xf32>
      %r6_half_tanh = arith.mulf %r6_tanh, %r6_half : tensor<{batch}x{size}xf32>
      %r6_sigmoid = arith.addf %r6_half_tanh, %r6_half : tensor<{batch}x{size}xf32>
      %r6_scores = arith.addf %r6_sigmoid, %r6_bias : tensor<{batch}x{size}xf32>
      %r6_pair:2 = "tt.reduce"(%r6_scores, %r6_neginf) <{{axis = 1 : i32}}> ({{
      ^bb0(%r6_a: f32, %r6_b: f32, %r6_c: f32, %r6_d: f32):
        %r6_hi = arith.maxnumf %r6_a, %r6_c : f32
        %r6_cross = arith.minnumf %r6_a, %r6_c : f32
        %r6_lo2 = arith.maxnumf %r6_b, %r6_d : f32
        %r6_lo = arith.maxnumf %r6_cross, %r6_lo2 : f32
        tt.reduce.return %r6_hi, %r6_lo : f32, f32
      }}) : (tensor<{batch}x{size}xf32>, tensor<{batch}x{size}xf32>) -> (tensor<{batch}xf32>, tensor<{batch}xf32>)
      %r6_result = arith.addf %r6_pair#0, %r6_pair#1 : tensor<{batch}xf32>
      %r6_out_base = tt.addptr %group_scores_ptr, %r6_pid : !tt.ptr<f32>, i32
      %r6_out_splat = tt.splat %r6_out_base : !tt.ptr<f32> -> tensor<{batch}x!tt.ptr<f32>>
      %r6_out_ptrs = tt.addptr %r6_out_splat, %r6_rows : tensor<{batch}x!tt.ptr<f32>>, tensor<{batch}xi32>
      tt.store %r6_out_ptrs, %r6_result : tensor<{batch}x!tt.ptr<f32>>
    }}
"""
    return text[:begin] + body + text[end:]


def main():
    target = BASE / "r6"
    previous = BASE / "r5"
    selection = json.loads((previous / "selection.json").read_text())
    assert selection["qualified"] and selection["selected"] == "pair_integer_constants"
    names = ["batch2", "batch4", "batch8"]
    records = []
    for e in (256, 512):
        folder = target / "ir" / f"e{e}_w4"
        for stage in (1, 2):
            for name in ("baseline", "anchor"):
                freeze(
                    folder / f"{name}_s{stage}.ttir",
                    (
                        previous / "ir" / f"e{e}_w4" / f"{name}_s{stage}.ttir"
                    ).read_text(),
                )
            freeze(
                folder / f"parent_s{stage}.ttir",
                (previous / "ir" / f"e{e}_w4" / f"optimized_s{stage}.ttir").read_text(),
            )
        s1 = (folder / "parent_s1.ttir").read_text()
        s2 = (folder / "parent_s2.ttir").read_text()
        for batch, name in zip((2, 4, 8), names):
            for stage, text in enumerate((batch_groups(s1, e, batch), s2), 1):
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
    freeze(target / "candidates.json", json.dumps(records, indent=2) + "\n")
    manifest = BASE.parent / "round2/manifest30.json"
    freeze(target / "manifest30.json", manifest.read_text())
    lineage = {
        "round": "r6",
        "previous_round": str(previous),
        "parent_selection_sha256": digest(previous / "selection.json"),
        "manifest_sha256": digest(manifest),
        "candidates": names,
        "generator_sha256": digest(Path(__file__)),
    }
    freeze(target / "lineage.json", json.dumps(lineage, indent=2) + "\n")


if __name__ == "__main__":
    main()
