"""Map active CTAs to independent row groups inside TTIR; preserve external launch ABI."""

from common import BLOCKS, ROOT, digest, save
from grouped_ir import IR


def grouped_rows(base, block, rows, tile):
    b = IR()
    tile = min(tile, block)
    ty = f"tensor<{rows}x{tile}xf32>"
    hty = f"tensor<{rows}x{tile}xf16>"
    ity = f"tensor<{rows}x{tile}xi32>"
    pty = f"tensor<{rows}x{tile}x!tt.ptr<f16>>"
    rty, cty = f"tensor<{rows}xi32>", f"tensor<{tile}xi32>"
    stats = f"tensor<{rows}xf32>"
    pid = b.op("tt.get_program_id x : i32")
    m = b.op("tt.get_num_programs x : i32")
    r = b.op(f"arith.constant {rows} : i32")
    first = b.op(f"arith.muli {pid}, {r} : i32")
    active = b.op(f"arith.cmpi slt, {first}, {m} : i32")
    b.emit(f"scf.if {active} {{")
    rr = b.op(f"tt.make_range {{start = 0 : i32, end = {rows} : i32}} : {rty}")
    rs = b.splat(first, "i32", rows)
    rr = b.op(f"arith.addi {rr}, {rs} : {rty}")
    rm = b.splat(m, "i32", rows)
    validrows = b.op(f"arith.cmpi slt, {rr}, {rm} : {rty}")
    stride = b.splat("%stride", "i32", rows)
    offsets = b.op(f"arith.muli {rr}, {stride} : {rty}")
    expanded = b.op(
        f"tt.expand_dims {offsets} {{axis = 1 : i32}} : {rty} -> tensor<{rows}x1xi32>"
    )
    offsets = b.op(f"tt.broadcast {expanded} : tensor<{rows}x1xi32> -> {ity}")
    rmask = b.op(
        f"tt.expand_dims {validrows} {{axis = 1 : i32}} : tensor<{rows}xi1> -> tensor<{rows}x1xi1>"
    )
    rmask = b.op(
        f"tt.broadcast {rmask} : tensor<{rows}x1xi1> -> tensor<{rows}x{tile}xi1>"
    )
    col = b.op(f"tt.make_range {{start = 0 : i32, end = {tile} : i32}} : {cty}")
    cexpand = b.op(
        f"tt.expand_dims {col} {{axis = 0 : i32}} : {cty} -> tensor<1x{tile}xi32>"
    )
    col = b.op(f"tt.broadcast {cexpand} : tensor<1x{tile}xi32> -> {ity}")
    n = b.op(f"tt.splat %N : i32 -> {ity}")
    zeros = b.op(f"arith.constant dense<0.0> : {ty}")
    hz = b.op(f"arith.constant dense<0.0> : {hty}")
    sz = b.op("arith.constant 0.0 : f32")
    meanstd = []
    for pointer in ["%Mean", "%Rstd"]:
        p = b.splat(pointer, "!tt.ptr<f32>", rows)
        p = b.op(f"tt.addptr {p}, {rr} : tensor<{rows}x!tt.ptr<f32>>, {rty}")
        z = b.op(f"tt.splat {sz} : f32 -> {stats}")
        v = b.op(f"tt.load {p}, {validrows}, {z} : tensor<{rows}x!tt.ptr<f32>>")
        meanstd.append(v)
    mean, rstd = meanstd

    def wide(v):
        e = b.op(
            f"tt.expand_dims {v} {{axis = 1 : i32}} : {stats} -> tensor<{rows}x1xf32>"
        )
        return b.op(f"tt.broadcast {e} : tensor<{rows}x1xf32> -> {ty}")

    means, rstds = wide(mean), wide(rstd)
    ptrs = {}
    for pointer in ["X", "DY", "DX", "W"]:
        p = b.op(f"tt.splat %{pointer} : !tt.ptr<f16> -> {pty}")
        if pointer != "W":
            p = b.op(f"tt.addptr {p}, {offsets} : {pty}, {ity}")
        ptrs[pointer] = p

    def data(offset):
        cs = col
        if offset is not None:
            o = b.op(f"tt.splat {offset} : i32 -> {ity}")
            cs = b.op(f"arith.addi {col}, {o} : {ity}")
        cm = b.op(f"arith.cmpi slt, {cs}, {n} : {ity}")
        mask = b.op(f"arith.andi {cm}, {rmask} : tensor<{rows}x{tile}xi1>")
        values = []
        for pointer in ["X", "DY", "W"]:
            p = b.op(f"tt.addptr {ptrs[pointer]}, {cs} : {pty}, {ity}")
            v = b.op(f"tt.load {p}, {mask}, {hz} : {pty}")
            values.append(b.op(f"arith.extf {v} : {hty} to {ty}"))
        x, dy, w = values
        center = b.op(f"arith.subf {x}, {means} : {ty}")
        hat = b.op(f"arith.mulf {center}, {rstds} : {ty}")
        wd = b.op(f"arith.mulf {dy}, {w} : {ty}")
        return cs, mask, hat, wd

    zero = b.op("arith.constant 0 : i32")
    step = b.op(f"arith.constant {tile} : i32")
    full = tile == block
    if not full:
        b.emit(
            f"%loop:2 = scf.for %off = {zero} to %N step {step} "
            f"iter_args(%a0 = {zeros}, %a1 = {zeros}) -> ({ty}, {ty}) : i32 {{"
        )
    cs, mask, hat, wd = data(None if full else "%off")
    prod = b.op(f"arith.mulf {hat}, {wd} : {ty}")
    if not full:
        a = b.op(f"arith.addf %a0, {prod} : {ty}")
        d = b.op(f"arith.addf %a1, {wd} : {ty}")
        b.emit(f"scf.yield {a}, {d} : {ty}, {ty}\n    }}")
        prod, wd = "%loop#0", "%loop#1"
    b.emit(
        f"""%sums:2 = "tt.reduce"({prod}, {wd}) <{{axis = 1 : i32}}> ({{
    ^bb0(%l0: f32, %l1: f32, %r0: f32, %r1: f32):
      %s0 = arith.addf %l0, %r0 : f32
      %s1 = arith.addf %l1, %r1 : f32
      tt.reduce.return %s0, %s1 : f32, f32
    }}) : ({ty}, {ty}) -> ({stats}, {stats})"""
    )
    nf = b.op("arith.sitofp %N : i32 to f32")
    one = b.op("arith.constant 1.0 : f32")
    inv = b.op(f"arith.divf {one}, {nf} : f32")
    invs = b.splat(inv, "f32", rows)
    coefs = []
    for i in range(2):
        v = b.op(f"arith.mulf %sums#{i}, {invs} : {stats}")
        coefs.append(wide(v))
    if not full:
        b.emit(f"scf.for %outoff = {zero} to %N step {step} : i32 {{")
        cs, mask, hat, wd = data("%outoff")
    term = b.op(f"arith.mulf {hat}, {coefs[0]} : {ty}")
    total = b.op(f"arith.addf {term}, {coefs[1]} : {ty}")
    sub = b.op(f"arith.subf {wd}, {total} : {ty}")
    dx = b.op(f"arith.mulf {sub}, {rstds} : {ty}")
    h = b.op(f"arith.truncf {dx} : {ty} to {hty}")
    p = b.op(f'tt.addptr {ptrs["DX"]}, {cs} : {pty}, {ity}')
    b.emit(f"tt.store {p}, {h}, {mask} cacheModifier = cs : {pty}")
    if not full:
        b.emit("}")
    b.emit("}\n    tt.return")
    return base[: base.index("    %cst =")] + "\n".join(b.lines) + "\n  }\n}\n"


def main():
    manifest = {}
    for block in BLOCKS:
        base = (ROOT / f"ir/b{block}/joint_coefficients.ttir").read_text()
        for rows, tile in [(2, 512), (4, 512), (8, 256), (2, block)]:
            name = f"r5_rows{rows}_" + ("full" if tile == block else f"tile{tile}")
            path = ROOT / f"ir/b{block}/{name}.ttir"
            text = grouped_rows(base, block, rows, tile)
            if path.exists():
                assert path.read_text() == text
            else:
                path.write_text(text)
            manifest[f"{name}/b{block}"] = digest(path)
    save(ROOT / "results/round5/candidates.json", manifest)


if __name__ == "__main__":
    main()
