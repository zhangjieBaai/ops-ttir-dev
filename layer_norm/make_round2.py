"""Second-round direct TTIR transformations; no Python device kernel is generated."""

import re

from common import ROOT, save_json, sha256
from make_candidates import without_locations


class Builder:
    def __init__(self):
        self.lines = []
        self.seq = 0

    def emit(self, text):
        self.lines.append(text)

    def op(self, expr):
        name = f"%v{self.seq}"
        self.seq += 1
        self.emit(f"    {name} = {expr}")
        return name

    def constant(self, value, ty):
        return self.op(f"arith.constant {value} : {ty}")

    def splat(self, value, scalar_ty, count):
        return self.op(f"tt.splat {value} : {scalar_ty} -> tensor<{count}x{scalar_ty}>")

    def reduce(self, value, count):
        name = f"%v{self.seq}"
        self.seq += 1
        self.emit(
            f"""    {name} = "tt.reduce"({value}) <{{axis = 0 : i32}}> ({{
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      tt.reduce.return %sum : f32
    }}) : (tensor<{count}xf32>) -> f32"""
        )
        return name


def delayed_statistics(text):
    lines = text.splitlines()
    stores = [line for line in lines if re.match(r"\s*tt.store %[01], ", line)]
    assert len(stores) == 12
    # Each branch has exactly one Mean and Rstd store and its own lexical scope.
    result = []
    pending = []
    for line in lines:
        if (
            pending
            and line.lstrip().startswith("}")
            and len(line) - len(line.lstrip())
            == len(pending[0]) - len(pending[0].lstrip()) - 2
        ):
            result.extend(pending)
            pending = []
        if line in stores:
            pending.append(line)
        else:
            result.append(line)
    assert not pending
    return "\n".join(result) + "\n"


def affine(text):
    # All copies of the tutorial expression use x_hat_20 = x - mean, then *rstd.
    pattern = re.compile(
        r"^(\s*)(%\w+) = arith.subf (%\w+), (%\w+) : (tensor<\d+xf32>)\n"
        r"\s*(%\w+) = tt.splat (%\w+) : f32 -> \5\n"
        r"\s*(%\w+) = arith.mulf \2, \6 : \5",
        re.M,
    )

    def replace(m):
        indent, sub, x, mean, ty, rs, scalar, out = m.groups()
        return (
            f"{indent}{rs} = tt.splat {scalar} : f32 -> {ty}\n"
            f"{indent}{sub} = arith.mulf {mean}, {rs} : {ty}\n"
            f"{indent}{sub}_neg = arith.negf {sub} : {ty}\n"
            f"{indent}{out} = math.fma {x}, {rs}, {sub}_neg : {ty}"
        )

    result, count = pattern.subn(replace, text)
    assert count == 7, count
    return result


def grouped_body(
    b,
    chunk,
    count=None,
    volatile=False,
    delay=False,
    fold=None,
    out_chunk=None,
):
    f32 = f"tensor<{chunk}xf32>"
    f16 = f"tensor<{chunk}xf16>"
    i32 = f"tensor<{chunk}xi32>"
    ptr = f"tensor<{chunk}x!tt.ptr<f16>>"
    zero = b.constant("0", "i32")
    step = b.constant(str(chunk), "i32")
    one = b.constant("1.0", "f32")
    zero16 = b.constant("dense<0.0>", f16)
    zero32 = b.constant("dense<0.0>", f32)
    row = b.op("tt.get_program_id x : i32")
    offset = b.op(f"arith.muli {row}, %stride : i32")
    xp = b.op(f"tt.addptr %X, {offset} : !tt.ptr<f16>, i32")
    yp = b.op(f"tt.addptr %Y, {offset} : !tt.ptr<f16>, i32")
    arange = b.op(f"tt.make_range {{end = {chunk} : i32, start = 0 : i32}} : {i32}")
    ns = b.splat("%N", "i32", chunk)
    inv = b.op("arith.sitofp %N : i32 to f32")
    inv = b.op(f"arith.divf {one}, {inv} : f32")

    def cols_mask(off):
        if off == zero:
            cols = arange
        else:
            os = b.splat(off, "i32", chunk)
            cols = b.op(f"arith.addi {os}, {arange} : {i32}")
        mask = b.op(f"arith.cmpi slt, {cols}, {ns} : {i32}")
        return cols, mask

    def load(p, cols, mask, vol=False):
        ps = b.splat(p, "!tt.ptr<f16>", chunk)
        addr = b.op(f"tt.addptr {ps}, {cols} : {ptr}, {i32}")
        attrs = " {isVolatile = true}" if vol else ""
        val = b.op(f"tt.load {addr}, {mask}, {zero16}{attrs} : {ptr}")
        return b.op(f"arith.extf {val} : {f16} to {f32}")

    def accumulate(stage, mean=None):
        width = fold or chunk
        acc_ty = f"tensor<{width}xf32>"
        init = zero32 if not fold else b.constant("dense<0.0>", acc_ty)
        acc = init
        if count is None:
            result = f"%acc_result{b.seq}"
            off = f"%off{b.seq}"
            iv = f"%acc{b.seq}"
            b.seq += 1
            b.emit(
                f"    {result} = scf.for {off} = {zero} to %N step {step} "
                f"iter_args({iv} = {init}) -> ({acc_ty}) : i32 {{"
            )
            acc = iv
            iters = [off]
        else:
            iters = [
                zero if i == 0 else b.constant(str(i * chunk), "i32")
                for i in range(count)
            ]
        for off in iters:
            cols, mask = cols_mask(off)
            x = load(xp, cols, mask, volatile and stage == "mean")
            if stage == "var":
                ms = b.splat(mean, "f32", chunk)
                centered = b.op(f"arith.subf {x}, {ms} : {f32}")
                centered = b.op(
                    f"arith.select {mask}, {centered}, {zero32} : tensor<{chunk}xi1>, {f32}"
                )
                x = b.op(f"arith.mulf {centered}, {centered} : {f32}")
            if fold:
                shaped = b.op(
                    f"tt.reshape {x} : {f32} -> tensor<{chunk//fold}x{fold}xf32>"
                )
                value = f"%fold{b.seq}"
                b.seq += 1
                b.emit(
                    f"""    {value} = "tt.reduce"({shaped}) <{{axis = 0 : i32}}> ({{
    ^bb0(%fl: f32, %fr: f32):
      %fs = arith.addf %fl, %fr : f32
      tt.reduce.return %fs : f32
    }}) : (tensor<{chunk//fold}x{fold}xf32>) -> {acc_ty}"""
                )
                x = value
            acc = b.op(f"arith.addf {acc}, {x} : {acc_ty}")
        if count is None:
            b.emit(f"      scf.yield {acc} : {acc_ty}\n    }}")
            acc = result
        total = b.reduce(acc, width)
        return b.op(f"arith.mulf {total}, {inv} : f32")

    mean = accumulate("mean")
    var = accumulate("var", mean)
    var_eps = b.op(f"arith.addf {var}, %eps : f32")
    rstd = b.op(f"math.rsqrt {var_eps} : f32")

    def statistics():
        mp = b.op(f"tt.addptr %Mean, {row} : !tt.ptr<f32>, i32")
        rp = b.op(f"tt.addptr %Rstd, {row} : !tt.ptr<f32>, i32")
        b.emit(f"    tt.store {mp}, {mean} : !tt.ptr<f32>")
        b.emit(f"    tt.store {rp}, {rstd} : !tt.ptr<f32>")

    if not delay:
        statistics()
    if out_chunk:
        assert count is None
        chunk = out_chunk
        f32 = f"tensor<{chunk}xf32>"
        f16 = f"tensor<{chunk}xf16>"
        i32 = f"tensor<{chunk}xi32>"
        ptr = f"tensor<{chunk}x!tt.ptr<f16>>"
        step = b.constant(str(chunk), "i32")
        zero16 = b.constant("dense<0.0>", f16)
        arange = b.op(f"tt.make_range {{end = {chunk} : i32, start = 0 : i32}} : {i32}")
        ns = b.splat("%N", "i32", chunk)
    if count is None:
        off = f"%outoff{b.seq}"
        b.seq += 1
        b.emit(f"    scf.for {off} = {zero} to %N step {step} : i32 {{")
        iters = [off]
    else:
        iters = [
            zero if i == 0 else b.constant(str(i * chunk), "i32") for i in range(count)
        ]
    for off in iters:
        cols, mask = cols_mask(off)
        x = load(xp, cols, mask)
        w = load("%W", cols, mask)
        bias = load("%B", cols, mask)
        ms = b.splat(mean, "f32", chunk)
        rs = b.splat(rstd, "f32", chunk)
        center = b.op(f"arith.subf {x}, {ms} : {f32}")
        norm = b.op(f"arith.mulf {center}, {rs} : {f32}")
        scaled = b.op(f"arith.mulf {norm}, {w} : {f32}")
        y = b.op(f"arith.addf {scaled}, {bias} : {f32}")
        y = b.op(f"arith.truncf {y} : {f32} to {f16}")
        ps = b.splat(yp, "!tt.ptr<f16>", chunk)
        addr = b.op(f"tt.addptr {ps}, {cols} : {ptr}, {i32}")
        b.emit(f"    tt.store {addr}, {y}, {mask} : {ptr}")
    if count is None:
        b.emit("    }")
    if delay:
        statistics()


def grouped(clean, chunk, limits=None, volatile=False, delay=False, small_chunk=None):
    b = Builder()
    b.emit(clean[: clean.index("    %cst =")].rstrip())

    def branch(index):
        if limits is None or index == len(limits):
            grouped_body(b, chunk, volatile=volatile, delay=delay)
            return
        n = limits[index]
        lim = b.constant(str(n), "i32")
        fits = b.op(f"arith.cmpi sle, %N, {lim} : i32")
        b.emit(f"    scf.if {fits} {{")
        block = small_chunk if small_chunk and n <= 3072 else chunk
        grouped_body(
            b, min(block, n), (n + block - 1) // block, volatile=volatile, delay=delay
        )
        b.emit("    } else {")
        branch(index + 1)
        b.emit("    }")

    branch(0)
    b.emit("    tt.return\n  }\n}")
    return "\n".join(b.lines) + "\n"


def affine_v2(text):
    return re.sub(
        r"^(\s*)(%\w+) = arith.negf (%\w+) : (tensor<\d+xf32>)$",
        lambda m: f"{m[1]}{m[2]}_zero = arith.constant dense<-0.0> : {m[4]}\n"
        f"{m[1]}{m[2]} = arith.subf {m[2]}_zero, {m[3]} : {m[4]}",
        text,
        flags=re.M,
    )


def folded(clean):
    b = Builder()
    b.emit(clean[: clean.index("    %cst =")].rstrip())
    grouped_body(b, 8192, fold=2048, out_chunk=2048)
    b.emit("    tt.return\n  }\n}")
    return "\n".join(b.lines) + "\n"


def main():
    clean = without_locations((ROOT / "ir/baseline.ttir").read_text())
    parent = (ROOT / "ir/tile_dispatch_math_v2.ttir").read_text()
    fine = [1024, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 14336, 16384]
    candidates = {
        "r2_delay": delayed_statistics(parent),
        "r2_affine": affine(parent),
        "r2_delay_affine": affine(delayed_statistics(parent)),
        "r2_loop1024": grouped(clean, 1024),
        "r2_loop2048": grouped(clean, 2048),
        "r2_loop4096": grouped(clean, 4096),
        "r2_fine1024": grouped(clean, 1024, fine),
        "r2_fine1024_reload": grouped(clean, 1024, fine, volatile=True),
    }
    candidates.update(
        {
            "r2_affine_v2": affine_v2(candidates["r2_affine"]),
            "r2_delay_affine_v2": affine_v2(candidates["r2_delay_affine"]),
            "r2_fine2048_reload": grouped(clean, 2048, fine, volatile=True),
            "r2_fine2048_delay": grouped(clean, 2048, fine, delay=True),
            "r2_fold2048": folded(clean),
        }
    )
    candidates["r2_fine2048_ca"] = candidates["r2_fine2048_reload"].replace(
        "{isVolatile = true}", "cacheModifier = ca"
    )
    candidates["r2_fine2048_stream"] = re.sub(
        r"^(\s*tt.store .*?, %v\d+)( : tensor<)",
        r"\1 cacheModifier = cs\2",
        candidates["r2_fine2048_ca"],
        flags=re.M,
    )
    candidates["r2_mixed_stream"] = re.sub(
        r"^(\s*tt.store .*?, %v\d+)( : tensor<)",
        r"\1 cacheModifier = cs\2",
        grouped(clean, 2048, fine, volatile=True, small_chunk=1024).replace(
            "{isVolatile = true}", "cacheModifier = ca"
        ),
        flags=re.M,
    )
    manifest = dict(baseline_sha256=sha256(ROOT / "ir/baseline.ttir"), candidates={})
    for name, text in candidates.items():
        path = ROOT / f"ir/{name}.ttir"
        if path.exists():
            assert path.read_text() == text, name
        else:
            path.write_text(text)
        manifest["candidates"][name] = dict(sha256=sha256(path), bytes=len(text))
    save_json(ROOT / "results/round2/candidates.json", manifest)
    print(manifest)


if __name__ == "__main__":
    main()
