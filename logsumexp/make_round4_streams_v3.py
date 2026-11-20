"""Adaptive hypothesis: independent 128-lane streams for single-row medium inputs."""

from common import ROOT, ir_path, manifest, save_json, sha256
from make_round3 import header


def stream128(text, config):
    width = config["meta"]["TILE_N"]
    if (
        "TILE_K" in config["meta"]
        or "%M:" in text
        or not config["meta"]["ONE_TILE_PER_CTA"]
        or width > 8192
        or width < 4096
    ):
        return text
    body = """    %r4_inf = arith.constant 0xFF800000 : f32
    %r4_zero = arith.constant 0.000000e+00 : f32
    %r4_other = arith.constant dense<0xFC00> : tensor<128xf16>
    %r4_range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %r4_nv = tt.splat %N : i32 -> tensor<128xi32>
    %r4_ptr = tt.splat %input_ptr : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>>
"""
    for i in range(width // 128):
        body += f"""    %c{i} = arith.constant dense<{i*128}> : tensor<128xi32>
    %idx{i} = arith.addi %r4_range, %c{i} : tensor<128xi32>
    %mask{i} = arith.cmpi slt, %idx{i}, %r4_nv : tensor<128xi32>
    %ptr{i} = tt.addptr %r4_ptr, %idx{i} : tensor<128x!tt.ptr<f16>>, tensor<128xi32>
    %x{i} = tt.load %ptr{i}, %mask{i}, %r4_other : tensor<128x!tt.ptr<f16>>
"""
        if i:
            body += (
                f"    %max{i} = arith.maxnumf %x{i}, "
                + ("%x0" if i == 1 else f"%max{i-1}")
                + " : tensor<128xf16>\n"
            )
    last = f"%max{width//128-1}"
    body += f"""    %r4_max16 = "tt.reduce"({last}) <{{axis = 0 : i32}}> ({{
    ^bb0(%a: f16, %b: f16):
      %m = arith.maxnumf %a, %b : f16
      tt.reduce.return %m : f16
    }}) : (tensor<128xf16>) -> f16
    %r4_max = arith.extf %r4_max16 : f16 to f32
    %r4_allinf = arith.cmpf oeq, %r4_max, %r4_inf : f32
    %r4_safe = arith.select %r4_allinf, %r4_zero, %r4_max : f32
    %r4_bmax = tt.splat %r4_safe : f32 -> tensor<128xf32>
"""
    for i in range(width // 128):
        body += f"""    %xf{i} = arith.extf %x{i} : tensor<128xf16> to tensor<128xf32>
    %delta{i} = arith.subf %xf{i}, %r4_bmax : tensor<128xf32>
    %exp{i} = math.exp %delta{i} : tensor<128xf32>
"""
        if i:
            body += (
                f"    %sum{i} = arith.addf %exp{i}, "
                + ("%exp0" if i == 1 else f"%sum{i-1}")
                + " : tensor<128xf32>\n"
            )
    last = f"%sum{width//128-1}"
    body += f"""    %r4_sum = "tt.reduce"({last}) <{{axis = 0 : i32}}> ({{
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }}) : (tensor<128xf32>) -> f32
    %r4_log = math.log %r4_sum : f32
    %r4_val = arith.addf %r4_safe, %r4_log : f32
    %r4_out = arith.select %r4_allinf, %r4_max, %r4_val : f32
    %r4_y16 = arith.truncf %r4_out : f32 to f16
    tt.store %output_ptr, %r4_y16 : !tt.ptr<f16>
    tt.return
  }}
}}
"""
    return header(text) + body


def main():
    output = {}
    for group, config in manifest()["groups"].items():
        parent = ir_path(group, "r3_mix")
        path = ir_path(group, "r4_stream128_v3")
        with path.open("x") as stream:
            stream.write(stream128(parent.read_text(), config))
        output[group] = dict(sha256=sha256(path), parent_sha256=sha256(parent))
    save_json(
        ROOT / "results/runs/r4_adaptive/candidate_stream128_v3.json",
        dict(
            groups=output,
            generator_sha256=sha256(__file__),
            hypothesis="128-lane independent streams for M1,N4096/8192; "
            "test communication reduction with unchanged launch config",
        ),
    )


if __name__ == "__main__":
    main()
