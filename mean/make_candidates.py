"""Construct only TTIR: split serial N accumulation across independent tile rows."""

from common import ROOT, ir_path, manifest, save_json, sha256


def tiled(original, rows):
    header = next(line for line in original.splitlines() if "tt.func public" in line)
    body = []

    def emit(line):
        body.append("    " + line)

    for param in ["N", "K"]:
        if f"%{param}:" not in header:
            emit(f"%{param} = arith.constant 1 : i32")
    types = dict(
        i=f"tensor<{rows}x1024xi32>",
        f=f"tensor<{rows}x1024xf32>",
        h=f"tensor<{rows}x1024xf16>",
        p=f"tensor<{rows}x1024x!tt.ptr<f16>>",
        b=f"tensor<{rows}x1024xi1>",
    )
    for name, value in [("zero", 0), ("step", rows), ("width", 1024)]:
        emit(f"%{name} = arith.constant {value} : i32")
    emit("%pm = tt.get_program_id x : i32")
    emit("%pk = tt.get_program_id y : i32")
    emit("%kb = arith.muli %pk, %width : i32")
    emit("%mn = arith.muli %pm, %N : i32")
    emit("%base = arith.muli %mn, %K : i32")
    emit("%kr = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>")
    emit("%kbs = tt.splat %kb : i32 -> tensor<1024xi32>")
    emit("%ko = arith.addi %kr, %kbs : tensor<1024xi32>")
    emit("%ks = tt.splat %K : i32 -> tensor<1024xi32>")
    emit("%km = arith.cmpi slt, %ko, %ks : tensor<1024xi32>")
    emit(
        f"%nr = tt.make_range {{end = {rows} : i32, start = 0 : i32}} : tensor<{rows}xi32>"
    )
    emit(
        f"%ne = tt.expand_dims %nr {{axis = 1 : i32}} : tensor<{rows}xi32> -> tensor<{rows}x1xi32>"
    )
    emit(
        "%ke = tt.expand_dims %ko {axis = 0 : i32} : tensor<1024xi32> -> tensor<1x1024xi32>"
    )
    emit(f'%kt = tt.broadcast %ke : tensor<1x1024xi32> -> {types["i"]}')
    emit(f'%ns = tt.splat %N : i32 -> {types["i"]}')
    emit(f'%stride = tt.splat %K : i32 -> {types["i"]}')
    emit(f'%bs = tt.splat %base : i32 -> {types["i"]}')
    emit(f'%zero_f = arith.constant dense<0.0> : {types["f"]}')
    emit(f'%zero_h = arith.constant dense<0.0> : {types["h"]}')
    emit(f'%ip = tt.splat %input_ptr : !tt.ptr<f16> -> {types["p"]}')
    emit(
        f'%acc = scf.for %n = %zero to %N step %step iter_args(%sum = %zero_f) -> ({types["f"]}) : i32 {{'
    )
    emit(f"  %nscalar = tt.splat %n : i32 -> tensor<{rows}x1xi32>")
    emit(f"  %ni = arith.addi %nscalar, %ne : tensor<{rows}x1xi32>")
    emit(f'  %nt = tt.broadcast %ni : tensor<{rows}x1xi32> -> {types["i"]}')
    emit(f'  %nm = arith.cmpi slt, %nt, %ns : {types["i"]}')
    emit(f'  %kmt = arith.cmpi slt, %kt, %stride : {types["i"]}')
    emit(f'  %mask = arith.andi %nm, %kmt : {types["b"]}')
    emit(f'  %no = arith.muli %nt, %stride : {types["i"]}')
    emit(f'  %off0 = arith.addi %no, %kt : {types["i"]}')
    emit(f'  %off = arith.addi %off0, %bs : {types["i"]}')
    emit(f'  %ptr = tt.addptr %ip, %off : {types["p"]}, {types["i"]}')
    emit(f'  %v = tt.load %ptr, %mask, %zero_h : {types["p"]}')
    emit(f'  %vf = arith.extf %v : {types["h"]} to {types["f"]}')
    emit(f'  %add = arith.addf %sum, %vf : {types["f"]}')
    emit(f'  scf.yield %add : {types["f"]}')
    emit("}")
    emit('%reduced = "tt.reduce"(%acc) ({')
    emit("^bb0(%a: f32, %b: f32):")
    emit("  %c = arith.addf %a, %b : f32")
    emit("  tt.reduce.return %c : f32")
    emit(f'}}) {{axis = 0 : i32}} : ({types["f"]}) -> tensor<1024xf32>')
    emit("%nf = arith.sitofp %N : i32 to f32")
    emit("%nfs = tt.splat %nf : f32 -> tensor<1024xf32>")
    emit("%avg = arith.divf %reduced, %nfs : tensor<1024xf32>")
    emit("%avg_h = arith.truncf %avg : tensor<1024xf32> to tensor<1024xf16>")
    emit("%ob = arith.muli %pm, %K : i32")
    emit("%obs = tt.splat %ob : i32 -> tensor<1024xi32>")
    emit("%oo = arith.addi %obs, %ko : tensor<1024xi32>")
    emit("%op = tt.splat %output_ptr : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>")
    emit("%optr = tt.addptr %op, %oo : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>")
    emit("tt.store %optr, %avg_h, %km : tensor<1024x!tt.ptr<f16>>")
    emit("tt.return")
    start = original.index(header) + len(header)
    end = original.index("\n  }", start)
    return original[:start] + "\n" + "\n".join(body) + original[end:]


def main():
    records = []
    for group in manifest()["groups"]:
        original = ir_path(group, "baseline").read_text()
        for rows in [1, 4, 16, 32]:
            name = f"r1_tile{rows}"
            result = (
                tiled(original, rows)
                if "@mean_dim_kernel_non_inner_vec" in original
                else original
            )
            path = ir_path(group, name)
            with path.open("x") as f:
                f.write(result)
            records.append(
                dict(
                    group=group,
                    name=name,
                    sha256=sha256(path),
                    changed=result != original,
                )
            )
    save_json(ROOT / "results/candidates.json", records)


if __name__ == "__main__":
    main()
