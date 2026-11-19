"""Borrow three mechanisms from matched FlagGems TTIR without changing the launch contract."""

import re

from common import ROOT, frozen, save_json, sha256

FAMILY = "bm64_bn128_bk64_g8_w4_s4"
CANDIDATES = {
    "r2_pid_direct": ["pid"],
    "r2_zero_pad": ["zero"],
    "r2_absolute": ["absolute"],
    "r2_zero_absolute": ["zero", "absolute"],
    "r2_zero_pid": ["zero", "pid"],
    "r2_absolute_pid": ["absolute", "pid"],
    "r2_zero_absolute_pid": ["zero", "absolute", "pid"],
}


def once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def pid_direct(text):
    return once(
        text,
        "%pid_m_6 = arith.remsi %pid_m, %group_size_m_5",
        "%pid_m_6 = arith.remsi %pid, %group_size_m_5",
    )


def zero_pad(text):
    for result, raw in [("%offs_am_12", "%offs_am_10"), ("%offs_bn_17", "%offs_bn_15")]:
        text, count = re.subn(
            r"^    " + result + r" = arith.remsi[^\n]+\n", "", text, flags=re.M
        )
        assert count == 1
        text = re.sub(re.escape(result) + r"\b", raw, text)
    marker = "    %a_ptrs = tt.expand_dims"
    masks = """    %r2_am_ok = arith.cmpi slt, %offs_am_10, %offs_am_11 : tensor<64xi32>
    %r2_am_x = tt.expand_dims %r2_am_ok {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %r2_am_mask = tt.broadcast %r2_am_x : tensor<64x1xi1> -> tensor<64x64xi1>
    %r2_bn_ok = arith.cmpi slt, %offs_bn_15, %offs_bn_16 : tensor<128xi32>
    %r2_bn_x = tt.expand_dims %r2_bn_ok {axis = 0 : i32} : tensor<128xi1> -> tensor<1x128xi1>
    %r2_bn_mask = tt.broadcast %r2_bn_x : tensor<1x128xi1> -> tensor<64x128xi1>
"""
    text = once(text, marker, masks + marker)
    text = once(
        text,
        "      %a_56 = tt.load %a_ptrs_49, %a_55,",
        "      %r2_a_mask = arith.andi %a_55, %r2_am_mask : tensor<64x64xi1>\n"
        "      %a_56 = tt.load %a_ptrs_49, %r2_a_mask,",
    )
    text = once(
        text,
        "      %b_59 = tt.load %b_ptrs_50, %b_58,",
        "      %r2_b_mask = arith.andi %b_58, %r2_bn_mask : tensor<64x128xi1>\n"
        "      %b_59 = tt.load %b_ptrs_50, %r2_b_mask,",
    )
    return text


def absolute(text):
    lines = text.splitlines()
    index = next(i for i, line in enumerate(lines) if "= scf.for" in line)
    header = lines[index]
    newheader = (
        "    %r2_acc = scf.for %k = %c0_i32 to %6 step %c1_i32 "
        "iter_args(%accumulator_51 = %accumulator) -> (tensor<64x128xf32>) : i32 {"
    )
    text = once(text, header, newheader)
    marker = "      %a_52 = arith.subi"
    addressing = """      %r2_a_delta = tt.splat %a : i32 -> tensor<64x64xi32>
      %r2_a_iter = tt.addptr %a_ptrs_25, %r2_a_delta : tensor<64x64x!tt.ptr<f16>>, tensor<64x64xi32>
      %r2_b_offset = arith.muli %a, %stride_bk : i32
      %r2_b_delta = tt.splat %r2_b_offset : i32 -> tensor<64x128xi32>
      %r2_b_iter = tt.addptr %b_ptrs_33, %r2_b_delta : tensor<64x128x!tt.ptr<f16>>, tensor<64x128xi32>
"""
    text = once(text, marker, addressing + marker)
    text = text.replace("tt.load %a_ptrs_49,", "tt.load %r2_a_iter,").replace(
        "tt.load %b_ptrs_50,", "tt.load %r2_b_iter,"
    )
    text, count = re.subn(
        r"^      %(a_ptrs_61|b_ptrs_62|b_ptrs_63|b_ptrs_64) =[^\n]+\n",
        "",
        text,
        flags=re.M,
    )
    assert count == 4
    text, count = re.subn(
        r"^      scf.yield %a_ptrs_61,[^\n]+",
        "      scf.yield %accumulator_60 : tensor<64x128xf32>",
        text,
        flags=re.M,
    )
    assert count == 1
    text = once(text, "%accumulator_34#2", "%r2_acc")
    return text


def descriptor_probe(text):
    # Compilation-only diagnostic. Its body is hand-written TTIR under the
    # unchanged pointer/scalar signature; any global workspace bars ranking.
    # Reuse the frozen signature; strip only its source-location annotations.
    entry = next(line for line in text.splitlines() if "tt.func public @" in line)
    entry = re.sub(r' loc\("[^"\n]*"\(#loc\w*\)\)', "", entry)
    entry = re.sub(r" loc\(#loc\w*\)", "", entry)
    return (
        """module {
"""
        + entry
        + """
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c8 = arith.constant 8 : i32
    %c64 = arith.constant 64 : i32
    %c128 = arith.constant 128 : i32
    %c63 = arith.constant 63 : i32
    %c127 = arith.constant 127 : i32
    %one64 = arith.constant 1 : i64
    %sa = arith.extsi %stride_am : i32 to i64
    %sb = arith.extsi %stride_bk : i32 to i64
    %sc = arith.extsi %stride_cm : i32 to i64
    %da = tt.make_tensor_descriptor %a_ptr, [%M, %K], [%sa, %one64] : <f16>, <tensor<64x64xf16>>
    %db = tt.make_tensor_descriptor %b_ptr, [%K, %N], [%sb, %one64] : <f16>, <tensor<64x128xf16>>
    %dc = tt.make_tensor_descriptor %c_ptr, [%M, %N], [%sc, %one64] : <f16>, <tensor<64x128xf16>>
    %pid = tt.get_program_id x : i32
    %madd = arith.addi %M, %c63 : i32
    %nm = arith.divsi %madd, %c64 : i32
    %nadd = arith.addi %N, %c127 : i32
    %nn = arith.divsi %nadd, %c128 : i32
    %width = arith.muli %nn, %c8 : i32
    %group = arith.divsi %pid, %width : i32
    %first = arith.muli %group, %c8 : i32
    %remaining = arith.subi %nm, %first : i32
    %size = arith.minsi %remaining, %c8 : i32
    %local = arith.remsi %pid, %width : i32
    %rm = arith.remsi %local, %size : i32
    %pm = arith.addi %first, %rm : i32
    %pn = arith.divsi %local, %size : i32
    %om = arith.muli %pm, %c64 : i32
    %on = arith.muli %pn, %c128 : i32
    %kadd = arith.addi %K, %c63 : i32
    %iters = arith.divsi %kadd, %c64 : i32
    %zero = arith.constant dense<0.0> : tensor<64x128xf32>
    %result = scf.for %k = %c0 to %iters step %c1 iter_args(%acc = %zero) -> (tensor<64x128xf32>) : i32 {
      %ok = arith.muli %k, %c64 : i32
      %a = tt.descriptor_load %da[%om, %ok] : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16>
      %b = tt.descriptor_load %db[%ok, %on] : !tt.tensordesc<tensor<64x128xf16>> -> tensor<64x128xf16>
      %next = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>
      scf.yield %next : tensor<64x128xf32>
    }
    %out = arith.truncf %result : tensor<64x128xf32> to tensor<64x128xf16>
    tt.descriptor_store %dc[%om, %on], %out : !tt.tensordesc<tensor<64x128xf16>>, tensor<64x128xf16>
    tt.return
  }
}
"""
    )


def main():
    manifest = frozen()
    config = manifest["families"][FAMILY]
    path = ROOT / config["ttir_path"]
    source = path.read_text()
    records = []
    transforms = {"pid": pid_direct, "zero": zero_pad, "absolute": absolute}
    for name, steps in CANDIDATES.items():
        value = source
        for step in steps:
            value = transforms[step](value)
        output = path.with_name(name + ".ttir")
        if output.exists():
            assert output.read_text() == value
        else:
            with output.open("x") as f:
                f.write(value)
        records.append(
            dict(
                name=name,
                steps=steps,
                ttir_path=str(output.relative_to(ROOT)),
                ttir_sha256=sha256(output),
                parent_sha256=sha256(path),
            )
        )
    out = ROOT / "results/runs/mm_opt_r2_20261119"
    probe = out / "device_descriptor_probe.ttir"
    value = descriptor_probe(source)
    if probe.exists():
        assert probe.read_text() == value
    else:
        probe.write_text(value)
    save_json(
        out / "candidates.json",
        dict(
            family=FAMILY,
            plan_sha256=sha256(ROOT / "optimization_round2.md"),
            candidates=records,
            diagnostic_only=str(probe),
        ),
    )
    print(
        "Generated7 direct TTIR functional candidates and1 compilation-only descriptor diagnostic"
    )


if __name__ == "__main__":
    main()
