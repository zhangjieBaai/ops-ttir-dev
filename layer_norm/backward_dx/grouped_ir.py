"""Construct grouped TTIR directly from the frozen function header."""


class IR:
    def __init__(self):
        self.lines = []
        self.i = 0

    def emit(self, line):
        self.lines.append("    " + line)

    def op(self, expr):
        name = f"%g{self.i}"
        self.i += 1
        self.emit(f"{name} = {expr}")
        return name

    def splat(self, x, t, size):
        return self.op(f"tt.splat {x} : {t} -> tensor<{size}x{t}>")


def grouped(base, block):
    b = IR()
    size = min(block, 2048)
    count = block // size
    ty = f"tensor<{size}xf32>"
    half = f"tensor<{size}xf16>"
    idx = f"tensor<{size}xi32>"
    ptr = f"tensor<{size}x!tt.ptr<f16>>"
    zero = b.op(f"arith.constant dense<0.0> : {ty}")
    hzero = b.op(f"arith.constant dense<0.0> : {half}")
    row = b.op("tt.get_program_id x : i32")
    offset = b.op(f"arith.muli {row}, %stride : i32")
    xp = b.op(f"tt.addptr %X, {offset} : !tt.ptr<f16>, i32")
    dp = b.op(f"tt.addptr %DY, {offset} : !tt.ptr<f16>, i32")
    op = b.op(f"tt.addptr %DX, {offset} : !tt.ptr<f16>, i32")
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

    def data(tile, ca):
        c = cols
        if tile:
            off = b.op(f"arith.constant dense<{tile*size}> : {idx}")
            c = b.op(f"arith.addi {cols}, {off} : {idx}")
        mask = b.op(f"arith.cmpi slt, {c}, {ns} : {idx}")
        values = []
        for p in [xp, dp, "%W"]:
            splat = b.splat(p, "!tt.ptr<f16>", size)
            address = b.op(f"tt.addptr {splat}, {c} : {ptr}, {idx}")
            attr = " cacheModifier = ca" if ca else ""
            v = b.op(f"tt.load {address}, {mask}, {hzero}{attr} : {ptr}")
            values.append(b.op(f"arith.extf {v} : {half} to {ty}"))
        x, dy, w = values
        centered = b.op(f"arith.subf {x}, {ms} : {ty}")
        xhat = b.op(f"arith.mulf {centered}, {rs} : {ty}")
        xhat = b.op(f"arith.select {mask}, {xhat}, {zero} : tensor<{size}xi1>, {ty}")
        wdy = b.op(f"arith.mulf {w}, {dy} : {ty}")
        return c, mask, xhat, wdy

    acc1 = zero
    acc2 = zero
    for tile in range(count):
        _, _, xhat, wdy = data(tile, True)
        prod = b.op(f"arith.mulf {xhat}, {wdy} : {ty}")
        acc1 = b.op(f"arith.addf {acc1}, {prod} : {ty}")
        acc2 = b.op(f"arith.addf {acc2}, {wdy} : {ty}")
    b.emit(
        f"""%sums:2 = "tt.reduce"({acc1}, {acc2}) <{{axis = 0 : i32}}> ({{
    ^bb0(%l0: f32, %l1: f32, %r0: f32, %r1: f32):
      %s0 = arith.addf %l0, %r0 : f32
      %s1 = arith.addf %l1, %r1 : f32
      tt.reduce.return %s0, %s1 : f32, f32
    }}) : ({ty}, {ty}) -> (f32, f32)"""
    )
    c1 = b.op(f"arith.mulf %sums#0, {inv} : f32")
    c2 = b.op(f"arith.mulf %sums#1, {inv} : f32")
    cs1 = b.splat(c1, "f32", size)
    cs2 = b.splat(c2, "f32", size)
    for tile in range(count):
        c, mask, xhat, wdy = data(tile, False)
        mul = b.op(f"arith.mulf {xhat}, {cs1} : {ty}")
        add = b.op(f"arith.addf {mul}, {cs2} : {ty}")
        sub = b.op(f"arith.subf {wdy}, {add} : {ty}")
        dx = b.op(f"arith.mulf {sub}, {rs} : {ty}")
        dx = b.op(f"arith.truncf {dx} : {ty} to {half}")
        splat = b.splat(op, "!tt.ptr<f16>", size)
        address = b.op(f"tt.addptr {splat}, {c} : {ptr}, {idx}")
        b.emit(f"tt.store {address}, {dx}, {mask} cacheModifier = cs : {ptr}")
    b.emit("tt.return")
    header = base[: base.index("    %cst =")]
    return header + "\n".join(b.lines) + "\n  }\n}\n"
