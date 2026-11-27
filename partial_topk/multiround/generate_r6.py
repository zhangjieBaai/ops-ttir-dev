"""Direct TTIR sorted-list merge with one active head per local chunk."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from complex_suite.support import digest, freeze, save  # noqa: E402


def merge_ir(shape, packed):
    _, n, k = shape
    chunks = (n + 1023) // 1024
    b = 1 << (chunks - 1).bit_length()
    lines = ["module {"]
    args = ", ".join(
        f"%{name}: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}"
        for name, ty in (("X", "f32"), ("I", "i32"), ("V", "f32"), ("O", "i32"))
    )
    lines.append(
        f"  tt.func public @partial_select({args}) attributes {{noinline = false}} {{"
    )

    def emit(s):
        lines.append(s)

    emit(
        f"""    %row = tt.get_program_id x : i32
    %stride = arith.constant {chunks * k} : i32
    %k = arith.constant {k} : i32
    %rowoff = arith.muli %row, %stride : i32
    %outoff = arith.muli %row, %k : i32
    %xbase = tt.addptr %X, %rowoff : !tt.ptr<f32>, i32
    %ibase = tt.addptr %I, %rowoff : !tt.ptr<i32>, i32
    %vbase = tt.addptr %V, %outoff : !tt.ptr<f32>, i32
    %obase = tt.addptr %O, %outoff : !tt.ptr<i32>, i32
    %xvec = tt.splat %xbase : !tt.ptr<f32> -> tensor<{b}x!tt.ptr<f32>>
    %ivec = tt.splat %ibase : !tt.ptr<i32> -> tensor<{b}x!tt.ptr<i32>>
    %chunkids = tt.make_range {{end = {b} : i32, start = 0 : i32}} : tensor<{b}xi32>
    %nchunks = arith.constant dense<{chunks}> : tensor<{b}xi32>
    %kvec = arith.constant dense<{k}> : tensor<{b}xi32>
    %validchunk = arith.cmpi slt, %chunkids, %nchunks : tensor<{b}xi32>
    %offsets = arith.muli %chunkids, %kvec : tensor<{b}xi32>
    %xp = tt.addptr %xvec, %offsets : tensor<{b}x!tt.ptr<f32>>, tensor<{b}xi32>
    %ip = tt.addptr %ivec, %offsets : tensor<{b}x!tt.ptr<i32>>, tensor<{b}xi32>
    %neginf = arith.constant dense<0xFF800000> : tensor<{b}xf32>
    %intmax = arith.constant dense<2147483647> : tensor<{b}xi32>
    %zero = arith.constant dense<0.0> : tensor<{b}xf32>
    %izero = arith.constant dense<0> : tensor<{b}xi32>
    %one = arith.constant dense<1> : tensor<{b}xi32>
    %shift31 = arith.constant dense<31> : tensor<{b}xi32>
    %signbit = arith.constant dense<-2147483648> : tensor<{b}xi32>
    %values0 = tt.load %xp, %validchunk, %neginf : tensor<{b}x!tt.ptr<f32>>
    %ids0 = tt.load %ip, %validchunk, %intmax : tensor<{b}x!tt.ptr<i32>>
    %szero = arith.constant 0 : i32
    %sall = arith.constant -1 : i32
    %ssign = arith.constant -2147483648 : i32
    %smax = arith.constant 2147483647 : i32
"""
    )
    if packed:
        emit(
            f"""    %shift32v = arith.constant dense<32> : tensor<{b}xi64>
    %shift32 = arith.constant 32 : i64
"""
        )

    def encode(values, prefix):
        emit(
            f"""    %{prefix}_iszero = arith.cmpf oeq, {values}, %zero : tensor<{b}xf32>
    %{prefix}_normal = arith.select %{prefix}_iszero, %zero, {values} : tensor<{b}xi1>, tensor<{b}xf32>
    %{prefix}_bits = tt.bitcast %{prefix}_normal : tensor<{b}xf32> -> tensor<{b}xi32>
    %{prefix}_sign = arith.shrsi %{prefix}_bits, %shift31 : tensor<{b}xi32>
    %{prefix}_flip = arith.ori %{prefix}_sign, %signbit : tensor<{b}xi32>
    %{prefix}_keys = arith.xori %{prefix}_bits, %{prefix}_flip : tensor<{b}xi32>
"""
        )
        return f"%{prefix}_keys"

    def pack(keys, ids, prefix):
        emit(
            f"""    %{prefix}_reverse = arith.subi %intmax, {ids} : tensor<{b}xi32>
    %{prefix}_high = arith.extui {keys} : tensor<{b}xi32> to tensor<{b}xi64>
    %{prefix}_low = arith.extui %{prefix}_reverse : tensor<{b}xi32> to tensor<{b}xi64>
    %{prefix}_shifted = arith.shli %{prefix}_high, %shift32v : tensor<{b}xi64>
    %{prefix}_packed = arith.ori %{prefix}_shifted, %{prefix}_low : tensor<{b}xi64>
"""
        )
        return f"%{prefix}_packed"

    keys = encode("%values0", "init")
    ids = "%ids0"
    cursor = "%izero"
    state = pack(keys, ids, "init") if packed else keys
    for i in range(k):
        p = f"h{i}"
        ty = "i64" if packed else "i32"
        emit(
            f"""    %{p}_maximum = "tt.reduce"({state}) <{{axis = 0 : i32}}> ({{
    ^bb0(%{p}_a: {ty}, %{p}_b: {ty}):
      %{p}_m = arith.maxui %{p}_a, %{p}_b : {ty}
      tt.reduce.return %{p}_m : {ty}
    }}) : (tensor<{b}x{ty}>) -> {ty}
"""
        )
        if packed:
            emit(
                f"""    %{p}_high = arith.shrui %{p}_maximum, %shift32 : i64
    %{p}_maxkey = arith.trunci %{p}_high : i64 to i32
    %{p}_low = arith.trunci %{p}_maximum : i64 to i32
    %{p}_bestid = arith.subi %smax, %{p}_low : i32
"""
            )
            maxkey = f"%{p}_maxkey"
        else:
            maxkey = f"%{p}_maximum"
            emit(
                f"""    %{p}_maxvec = tt.splat {maxkey} : i32 -> tensor<{b}xi32>
    %{p}_match = arith.cmpi eq, {state}, %{p}_maxvec : tensor<{b}xi32>
    %{p}_choices = arith.select %{p}_match, {ids}, %intmax : tensor<{b}xi1>, tensor<{b}xi32>
    %{p}_bestid = "tt.reduce"(%{p}_choices) <{{axis = 0 : i32}}> ({{
    ^bb0(%{p}_ia: i32, %{p}_ib: i32):
      %{p}_im = arith.minsi %{p}_ia, %{p}_ib : i32
      tt.reduce.return %{p}_im : i32
    }}) : (tensor<{b}xi32>) -> i32
"""
            )
        emit(
            f"""    %{p}_positive = arith.cmpi slt, {maxkey}, %szero : i32
    %{p}_undo = arith.select %{p}_positive, %ssign, %sall : i32
    %{p}_valuebits = arith.xori {maxkey}, %{p}_undo : i32
    %{p}_value = tt.bitcast %{p}_valuebits : i32 -> f32
    %{p}_outidx = arith.constant {i} : i32
    %{p}_vp = tt.addptr %vbase, %{p}_outidx : !tt.ptr<f32>, i32
    %{p}_op = tt.addptr %obase, %{p}_outidx : !tt.ptr<i32>, i32
    tt.store %{p}_vp, %{p}_value : !tt.ptr<f32>
    tt.store %{p}_op, %{p}_bestid : !tt.ptr<i32>
"""
        )
        if i == k - 1:
            break
        emit(
            f"""    %{p}_idvec = tt.splat %{p}_bestid : i32 -> tensor<{b}xi32>
    %{p}_chosen = arith.cmpi eq, {ids}, %{p}_idvec : tensor<{b}xi32>
    %{p}_advance = arith.select %{p}_chosen, %one, %izero : tensor<{b}xi1>, tensor<{b}xi32>
    %{p}_cursor = arith.addi {cursor}, %{p}_advance : tensor<{b}xi32>
    %{p}_remains = arith.cmpi slt, %{p}_cursor, %kvec : tensor<{b}xi32>
    %{p}_valid = arith.andi %validchunk, %{p}_remains : tensor<{b}xi1>
    %{p}_read = arith.andi %{p}_chosen, %{p}_valid : tensor<{b}xi1>
    %{p}_offset = arith.addi %offsets, %{p}_cursor : tensor<{b}xi32>
    %{p}_xp = tt.addptr %xvec, %{p}_offset : tensor<{b}x!tt.ptr<f32>>, tensor<{b}xi32>
    %{p}_ip = tt.addptr %ivec, %{p}_offset : tensor<{b}x!tt.ptr<i32>>, tensor<{b}xi32>
    %{p}_loaded = tt.load %{p}_xp, %{p}_read, %neginf : tensor<{b}x!tt.ptr<f32>>
    %{p}_loadedid = tt.load %{p}_ip, %{p}_read, %intmax : tensor<{b}x!tt.ptr<i32>>
    %{p}_ids = arith.select %{p}_chosen, %{p}_loadedid, {ids} : tensor<{b}xi1>, tensor<{b}xi32>
"""
        )
        newkeys = encode(f"%{p}_loaded", p)
        fresh = pack(newkeys, f"%{p}_loadedid", p) if packed else newkeys
        emit(
            f"    %{p}_state = arith.select %{p}_chosen, {fresh}, {state} : "
            f"tensor<{b}xi1>, tensor<{b}x{ty}>"
        )
        state, ids, cursor = f"%{p}_state", f"%{p}_ids", f"%{p}_cursor"
    emit("    tt.return\n  }\n}\n")
    return "\n".join(lines)


def main():
    records = []
    manifest = json.loads((ROOT.parent / "round2/expanded/manifest.json").read_text())
    for item in manifest["records"]:
        for stage in ("local", "merge"):
            group = item["group"]
            parent = ROOT / "r5/ir" / group / stage / "optimized.ttir"
            text = parent.read_text()
            folder = ROOT / "r6/ir" / group / stage
            freeze(folder / "parent.ttir", text)
            for name, packed in (("merge_heads", False), ("merge_heads_packed", True)):
                value = text if stage == "local" else merge_ir(item["shape"], packed)
                path = folder / f"{name}.ttir"
                freeze(path, value)
                records.append(
                    {
                        "group": group,
                        "stage": stage,
                        "name": name,
                        "sha256": digest(path),
                        "parent_sha256": digest(parent),
                    }
                )
    save(ROOT / "r6/candidates.json", records)


if __name__ == "__main__":
    main()
