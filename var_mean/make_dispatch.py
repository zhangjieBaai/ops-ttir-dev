"""Write bounded runtime dispatch directly as TTIR, retaining centered variance."""

import re

from common import ROOT, digest
from make_candidates import record_generation, strip_locations, write


def tile(width):
    return f"""%zero = arith.constant dense<0.0> : tensor<{width}xf32>
%cols = tt.make_range {{end = {width} : i32, start = 0 : i32}} : tensor<{width}xi32>
%limit = tt.splat %N : i32 -> tensor<{width}xi32>
%mask = arith.cmpi slt, %cols, %limit : tensor<{width}xi32>
%ptr = tt.splat %row_ptr : !tt.ptr<f32> -> tensor<{width}x!tt.ptr<f32>>
%addrs = tt.addptr %ptr, %cols : tensor<{width}x!tt.ptr<f32>>, tensor<{width}xi32>
%values = tt.load %addrs, %mask, %zero : tensor<{width}x!tt.ptr<f32>>
%sum = "tt.reduce"(%values) <{{axis = 0 : i32}}> ({{
^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.reduce.return %s : f32
}}) : (tensor<{width}xf32>) -> f32
%nf = arith.sitofp %N : i32 to f32
%avg = arith.divf %sum, %nf : f32
%avgvec = tt.splat %avg : f32 -> tensor<{width}xf32>
%diff = arith.subf %values, %avgvec : tensor<{width}xf32>
%centered = arith.select %mask, %diff, %zero : tensor<{width}xi1>, tensor<{width}xf32>
%squares = arith.mulf %centered, %centered : tensor<{width}xf32>
%ss = "tt.reduce"(%squares) <{{axis = 0 : i32}}> ({{
^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.reduce.return %s : f32
}}) : (tensor<{width}xf32>) -> f32
%denom = arith.subf %nf, %correction : f32
%variance = arith.divf %ss, %denom : f32
 tt.store %mean_ptr, %avg : !tt.ptr<f32>
 tt.store %var_ptr, %variance : !tt.ptr<f32>
"""


def dispatch(widths):
    width = widths[0]
    if len(widths) == 1:
        return tile(width)
    return f"""%bound{width} = arith.constant {width} : i32
%fits{width} = arith.cmpi sle, %N, %bound{width} : i32
scf.if %fits{width} {{
{tile(width)}
}} else {{
{dispatch(widths[1:])}
}}
"""


def main():
    original = (ROOT / "ir/baseline.ttir").read_text()
    base = strip_locations(original)
    for num, name in [(4, "row_ptr"), (5, "var_ptr"), (6, "mean_ptr")]:
        base = re.sub(rf"%{num}(?![0-9])", f"%{name}", base)
    prefix = base.split("    %_sum = scf.for")[0]
    names = []
    for name, widths in [
        ("r2_single_dispatch_v1", [256, 1024, 4096, 8192, 16384]),
        ("r2_single_dispatch_v2", [1024, 4096, 16384]),
    ]:
        write(name, prefix + dispatch(widths) + "tt.return\n}\n}\n")
        names.append(name)
    record_generation(
        ROOT / "results/runs/r2/candidates_dispatch.json",
        dict(
            names=names,
            baseline_sha256=digest(ROOT / "ir/baseline.ttir"),
            rewrite=(
                "Keep signature and pointer setup; replace loops with single tile centered "
                "reductions under runtime N guards; no Python kernel re-export."
            ),
        ),
    )


if __name__ == "__main__":
    main()
