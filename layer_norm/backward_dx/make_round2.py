"""Bounded second-round TTIR edits; frozen tutorial, options and launch remain unchanged."""

from common import BLOCKS, ROOT, digest, save
from grouped_ir import IR


def reload_output(text, block, x_reload=False, wdy_reload=False):
    t = f"tensor<{block}xf32>"
    h = f"tensor<{block}xf16>"
    p = f"tensor<{block}x!tt.ptr<f16>>"
    prolog = []
    xhat = "%xhat_20"
    wdy = "%wdy_21"
    if x_reload:
        prolog += [
            f"    %newx_h = tt.load %x_6, %mask_1, %cst cacheModifier = ca : {p}",
            f"    %newx = arith.extf %newx_h : {h} to {t}",
            f"    %newcenter = arith.subf %newx, %xhat : {t}",
            f"    %newhat = arith.mulf %newcenter, %xhat_18 : {t}",
            f"    %newmasked = arith.select %mask_1, %newhat, %cst_0 : tensor<{block}xi1>, {t}",
        ]
        xhat = "%newmasked"
    if wdy_reload:
        prolog += [
            f"    %newdy_h = tt.load %dy_9, %mask_1, %cst cacheModifier = ca : {p}",
            f"    %neww_h = tt.load %w_12, %mask_1, %cst cacheModifier = ca : {p}",
            f"    %newdy = arith.extf %newdy_h : {h} to {t}",
            f"    %neww = arith.extf %neww_h : {h} to {t}",
            f"    %newwdy = arith.mulf %newdy, %neww : {t}",
        ]
        wdy = "%newwdy"
    begin = text.index("    %alpha =")
    suffix = text[begin:].replace("%xhat_20", xhat).replace("%wdy_21", wdy)
    return text[:begin] + "\n".join(prolog) + "\n" + suffix


def fold(text, block):
    width = min(block, 2048)
    parts = block // width
    if parts == 1:
        return text
    begin = text.index("    %pair:2 =")
    end = text.index("    %c1_23 =", begin)

    def reduce(inp, out):
        return f"""    %{out}_shaped = tt.reshape {inp} : tensor<{block}xf32> -> tensor<{parts}x{width}xf32>
    %{out} = "tt.reduce"(%{out}_shaped) <{{axis = 0 : i32}}> ({{
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }}) : (tensor<{parts}x{width}xf32>) -> tensor<{width}xf32>
"""

    old = (
        text[begin:end]
        .replace("%c1, %wdy_21", "%fold0, %fold1")
        .replace(f"tensor<{block}xf32>", f"tensor<{width}xf32>")
    )
    return (
        text[:begin]
        + reduce("%c1", "fold0")
        + reduce("%wdy_21", "fold1")
        + old
        + text[end:]
    )


def loop(base, block, chunk):
    size = min(block, chunk)
    b = IR()
    ty = f"tensor<{size}xf32>"
    half = f"tensor<{size}xf16>"
    idx = f"tensor<{size}xi32>"
    ptr = f"tensor<{size}x!tt.ptr<f16>>"
    zero = b.op("arith.constant 0 : i32")
    step = b.op(f"arith.constant {size} : i32")
    zeros = b.op(f"arith.constant dense<0.0> : {ty}")
    hzero = b.op(f"arith.constant dense<0.0> : {half}")
    row = b.op("tt.get_program_id x : i32")
    off = b.op(f"arith.muli {row}, %stride : i32")
    xp = b.op(f"tt.addptr %X, {off} : !tt.ptr<f16>, i32")
    dp = b.op(f"tt.addptr %DY, {off} : !tt.ptr<f16>, i32")
    outp = b.op(f"tt.addptr %DX, {off} : !tt.ptr<f16>, i32")
    mp = b.op(f"tt.addptr %Mean, {row} : !tt.ptr<f32>, i32")
    rp = b.op(f"tt.addptr %Rstd, {row} : !tt.ptr<f32>, i32")
    mean = b.op(f"tt.load {mp} : !tt.ptr<f32>")
    rstd = b.op(f"tt.load {rp} : !tt.ptr<f32>")
    ms = b.splat(mean, "f32", size)
    rs = b.splat(rstd, "f32", size)
    nf = b.op("arith.sitofp %N : i32 to f32")
    one = b.op("arith.constant 1.0 : f32")
    inv = b.op(f"arith.divf {one}, {nf} : f32")
    ns = b.splat("%N", "i32", size)
    cols = b.op(f"tt.make_range {{end = {size} : i32, start = 0 : i32}} : {idx}")

    def data(offset, ca):
        os = b.splat(offset, "i32", size)
        c = b.op(f"arith.addi {cols}, {os} : {idx}")
        mask = b.op(f"arith.cmpi slt, {c}, {ns} : {idx}")
        values = []
        for pointer in [xp, dp, "%W"]:
            sp = b.splat(pointer, "!tt.ptr<f16>", size)
            address = b.op(f"tt.addptr {sp}, {c} : {ptr}, {idx}")
            attr = " cacheModifier = ca" if ca else ""
            h = b.op(f"tt.load {address}, {mask}, {hzero}{attr} : {ptr}")
            values.append(b.op(f"arith.extf {h} : {half} to {ty}"))
        x, dy, w = values
        center = b.op(f"arith.subf {x}, {ms} : {ty}")
        hat = b.op(f"arith.mulf {center}, {rs} : {ty}")
        hat = b.op(f"arith.select {mask}, {hat}, {zeros} : tensor<{size}xi1>, {ty}")
        wdy = b.op(f"arith.mulf {w}, {dy} : {ty}")
        return c, mask, hat, wdy

    b.emit(
        f"%accs:2 = scf.for %offstat = {zero} to %N step {step} "
        f"iter_args(%acc0 = {zeros}, %acc1 = {zeros}) -> ({ty}, {ty}) : i32 {{"
    )
    _, _, hat, wdy = data("%offstat", False)
    prod = b.op(f"arith.mulf {hat}, {wdy} : {ty}")
    a0 = b.op(f"arith.addf %acc0, {prod} : {ty}")
    a1 = b.op(f"arith.addf %acc1, {wdy} : {ty}")
    b.emit(f"scf.yield {a0}, {a1} : {ty}, {ty}\n    }}")
    b.emit(
        f"""%sums:2 = "tt.reduce"(%accs#0, %accs#1) <{{axis = 0 : i32}}> ({{
    ^bb0(%l0: f32, %l1: f32, %r0: f32, %r1: f32):
      %s0 = arith.addf %l0, %r0 : f32
      %s1 = arith.addf %l1, %r1 : f32
      tt.reduce.return %s0, %s1 : f32, f32
    }}) : ({ty}, {ty}) -> (f32, f32)"""
    )
    c1 = b.op(f"arith.mulf %sums#0, {inv} : f32")
    c2 = b.op(f"arith.mulf %sums#1, {inv} : f32")
    alpha = b.op(f"arith.mulf {c1}, {rstd} : f32")
    beta = b.op(f"arith.mulf {c2}, {rstd} : f32")
    c1s = b.splat(alpha, "f32", size)
    c2s = b.splat(beta, "f32", size)
    b.emit(f"scf.for %offout = {zero} to %N step {step} : i32 {{")
    c, mask, hat, wdy = data("%offout", True)
    scaled = b.op(f"arith.mulf {wdy}, {rs} : {ty}")
    center = b.op(f"arith.subf {scaled}, {c2s} : {ty}")
    term = b.op(f"arith.mulf {hat}, {c1s} : {ty}")
    dx = b.op(f"arith.subf {center}, {term} : {ty}")
    dx = b.op(f"arith.truncf {dx} : {ty} to {half}")
    sp = b.splat(outp, "!tt.ptr<f16>", size)
    address = b.op(f"tt.addptr {sp}, {c} : {ptr}, {idx}")
    b.emit(f"tt.store {address}, {dx}, {mask} cacheModifier = cs : {ptr}\n    }}")
    b.emit("tt.return")
    return base[: base.index("    %cst =")] + "\n".join(b.lines) + "\n  }\n}\n"


def range_dispatch(base, block, chunk):
    if block != 16384:
        return (ROOT / f"ir/b{block}/r1_parent.ttir").read_text()
    header = base[: base.index("    %cst =")]
    dynamic = loop(base, block, chunk)
    fallback = (ROOT / f"ir/b{block}/grouped2048.ttir").read_text()

    def body(text):
        head = text.index("\n", text.index("tt.func")) + 1
        return text[head : text.rindex("    tt.return")]

    return (
        header + "    %range_limit = arith.constant 10240 : i32\n"
        "    %range_fits = arith.cmpi sle, %N, %range_limit : i32\n"
        "    scf.if %range_fits {\n"
        + body(dynamic)
        + "    } else {\n"
        + body(fallback)
        + "    }\n    tt.return\n  }\n}\n"
    )


def main():
    manifest = {}
    for block in BLOCKS:
        base = (ROOT / f"ir/b{block}/joint_coefficients.ttir").read_text()
        variants = {
            "r2_reload_x": reload_output(base, block, x_reload=True),
            "r2_reload_wdy": reload_output(base, block, wdy_reload=True),
            "r2_reload_all": reload_output(base, block, True, True),
            "r2_loop2048": loop(base, block, 2048),
            "r2_loop4096": loop(base, block, 4096),
        }
        variants["r2_fold_reload"] = fold(variants["r2_reload_all"], block)
        variants["r2_range2048"] = range_dispatch(base, block, 2048)
        variants["r2_range4096"] = range_dispatch(base, block, 4096)
        manifest[str(block)] = {}
        for name, text in variants.items():
            path = ROOT / f"ir/b{block}/{name}.ttir"
            if path.exists():
                assert path.read_text() == text
            else:
                path.write_text(text)
            manifest[str(block)][name] = digest(path)
    save(ROOT / "results/round2/candidates.json", manifest)


if __name__ == "__main__":
    main()
