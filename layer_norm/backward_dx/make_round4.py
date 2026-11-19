"""Direct TTIR cache and reduction experiments with fixed compiler options."""

import re

from common import BLOCKS, ROOT, digest, save
from grouped_ir import IR


def fold(text, block, width):
    width = min(block, width)
    parts = block // width
    begin = text.index("    %pair:2 =")
    end = text.index("    %c1_23 =", begin)
    ty = f"tensor<{parts}x{width}xf32>"
    out = f"tensor<{width}xf32>"
    body = f"""    %fa = tt.reshape %c1 : tensor<{block}xf32> -> {ty}
    %fb = tt.reshape %wdy_21 : tensor<{block}xf32> -> {ty}
    %fp:2 = "tt.reduce"(%fa, %fb) <{{axis = 0 : i32}}> ({{
    ^bb0(%a: f32, %b: f32, %c: f32, %d: f32):
      %foldsum0 = arith.addf %a, %c : f32
      %foldsum1 = arith.addf %b, %d : f32
      tt.reduce.return %foldsum0, %foldsum1 : f32, f32
    }}) : ({ty}, {ty}) -> ({out}, {out})
"""
    final = (
        text[begin:end]
        .replace("%c1, %wdy_21", "%fp#0, %fp#1")
        .replace(f"tensor<{block}xf32>", out)
    )
    return text[:begin] + body + final + text[end:]


def keep(base, block, width):
    b = IR()
    size = min(block, width)
    ty, half, idx, ptr = (
        f"tensor<{size}x{t}>" for t in ["f32", "f16", "i32", "!tt.ptr<f16>"]
    )
    zero = b.op(f"arith.constant dense<0.0> : {ty}")
    hz = b.op(f"arith.constant dense<0.0> : {half}")
    row = b.op("tt.get_program_id x : i32")
    off = b.op(f"arith.muli {row}, %stride : i32")
    pointers = [
        b.op(f"tt.addptr %{p}, {off} : !tt.ptr<f16>, i32") for p in ["X", "DY", "DX"]
    ]
    xp, dp, op = pointers
    stats = []
    for p in ["Mean", "Rstd"]:
        address = b.op(f"tt.addptr %{p}, {row} : !tt.ptr<f32>, i32")
        stats.append(b.op(f"tt.load {address} : !tt.ptr<f32>"))
    mean, rstd = stats
    ms, rs = [b.splat(v, "f32", size) for v in stats]
    nf = b.op("arith.sitofp %N : i32 to f32")
    one = b.op("arith.constant 1.0 : f32")
    inv = b.op(f"arith.divf {one}, {nf} : f32")
    ns = b.splat("%N", "i32", size)
    cols = b.op(f"tt.make_range {{end = {size} : i32, start = 0 : i32}} : {idx}")
    stored = []
    acc0 = acc1 = zero
    for tile in range(block // size):
        c = cols
        if tile:
            v = b.op(f"arith.constant dense<{tile*size}> : {idx}")
            c = b.op(f"arith.addi {cols}, {v} : {idx}")
        mask = b.op(f"arith.cmpi slt, {c}, {ns} : {idx}")
        values = []
        for p in [xp, dp, "%W"]:
            ps = b.splat(p, "!tt.ptr<f16>", size)
            address = b.op(f"tt.addptr {ps}, {c} : {ptr}, {idx}")
            val = b.op(f"tt.load {address}, {mask}, {hz} : {ptr}")
            values.append(b.op(f"arith.extf {val} : {half} to {ty}"))
        x, dy, w = values
        centered = b.op(f"arith.subf {x}, {ms} : {ty}")
        hat = b.op(f"arith.mulf {centered}, {rs} : {ty}")
        hat = b.op(f"arith.select {mask}, {hat}, {zero} : tensor<{size}xi1>, {ty}")
        wd = b.op(f"arith.mulf {w}, {dy} : {ty}")
        prod = b.op(f"arith.mulf {hat}, {wd} : {ty}")
        acc0 = b.op(f"arith.addf {acc0}, {prod} : {ty}")
        acc1 = b.op(f"arith.addf {acc1}, {wd} : {ty}")
        stored.append((c, mask, hat, wd))
    b.emit(
        f"""%sums:2 = "tt.reduce"({acc0}, {acc1}) <{{axis = 0 : i32}}> ({{
    ^bb0(%l0: f32, %l1: f32, %r0: f32, %r1: f32):
      %s0 = arith.addf %l0, %r0 : f32
      %s1 = arith.addf %l1, %r1 : f32
      tt.reduce.return %s0, %s1 : f32, f32
    }}) : ({ty}, {ty}) -> (f32, f32)"""
    )
    coef = []
    for i in range(2):
        c = b.op(f"arith.mulf %sums#{i}, {inv} : f32")
        c = b.op(f"arith.mulf {c}, {rstd} : f32")
        coef.append(b.splat(c, "f32", size))
    for c, mask, hat, wd in stored:
        scaled = b.op(f"arith.mulf {wd}, {rs} : {ty}")
        centered = b.op(f"arith.subf {scaled}, {coef[1]} : {ty}")
        term = b.op(f"arith.mulf {hat}, {coef[0]} : {ty}")
        dx = b.op(f"arith.subf {centered}, {term} : {ty}")
        dx = b.op(f"arith.truncf {dx} : {ty} to {half}")
        ps = b.splat(op, "!tt.ptr<f16>", size)
        address = b.op(f"tt.addptr {ps}, {c} : {ptr}, {idx}")
        b.emit(f"tt.store {address}, {dx}, {mask} cacheModifier = cs : {ptr}")
    b.emit("tt.return")
    return base[: base.index("    %cst =")] + "\n".join(b.lines) + "\n  }\n}\n"


def main():
    manifest = {}
    for block in BLOCKS:
        parent = (ROOT / f"ir/b{block}/r4_parent.ttir").read_text()
        # Fold uses immutable pair/coefficient form in every family.
        pair = (ROOT / f"ir/b{block}/joint_coefficients.ttir").read_text()
        cg = re.sub(r"(tt.load [^\n]+)( : tensor<)", r"\1 cacheModifier = cg\2", parent)
        stream = re.sub(
            r"(tt.store [^\n]+?)(?: cacheModifier = cs)?( : tensor<)",
            r"\1 cacheModifier = cs\2",
            cg,
        )
        variants = {
            "r4_cg": cg,
            "r4_cg_stream": stream,
            "r4_fold256_v2": fold(pair, block, 256),
            "r4_fold1024_v2": fold(pair, block, 1024),
            "r4_keep1024": keep(pair, block, 1024),
            "r4_keep2048": keep(pair, block, 2048),
        }
        for name, text in variants.items():
            path = ROOT / f"ir/b{block}/{name}.ttir"
            if path.exists():
                assert path.read_text() == text
            else:
                path.write_text(text)
            manifest[f"{name}/b{block}"] = digest(path)
    save(ROOT / "results/round4/candidates_batch1_v2.json", manifest)


if __name__ == "__main__":
    main()
