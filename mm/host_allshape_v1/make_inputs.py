"""Generate immutable six-family host descriptor TTIR inputs."""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MM = ROOT.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def store(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text() == value, path
    else:
        path.write_text(value)


def build(config, base, pid):
    bm, bn, bk = [config["meta"]["BLOCK_SIZE_" + k] for k in "MNK"]
    fields = []
    for name, x, y in [("ad", bm, bk), ("bd", bk, bn), ("cd", bm, bn)]:
        fields += [
            f"%{name}: !tt.tensordesc<tensor<{x}x{y}xf16>>",
            f"%{name}_s0: i32",
            f"%{name}_s1: i32",
            f"%{name}_st0: i64",
            f"%{name}_st1: i64",
        ]
    header = next(line for line in base.splitlines() if "tt.func public" in line)
    header = re.sub(r' loc\("[^"\n]*"\(#loc\w*\)\)', "", header)
    header = re.sub(r" loc\(#loc\w*\)", "", header)
    header = header.replace(
        "@matmul_kernel(", "@matmul_kernel(" + ", ".join(fields) + ", "
    )
    a, b, c = f"{bm}x{bk}", f"{bk}x{bn}", f"{bm}x{bn}"
    value = f"""module {{
{header}
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %g = arith.constant 8 : i32
    %bm = arith.constant {bm} : i32
    %bn = arith.constant {bn} : i32
    %bk = arith.constant {bk} : i32
    %bm1 = arith.constant {bm - 1} : i32
    %bn1 = arith.constant {bn - 1} : i32
    %bk1 = arith.constant {bk - 1} : i32
    %pid = tt.get_program_id x : i32
    %madd = arith.addi %M, %bm1 : i32
    %nm = arith.divsi %madd, %bm : i32
    %nadd = arith.addi %N, %bn1 : i32
    %nn = arith.divsi %nadd, %bn : i32
    %width = arith.muli %nn, %g : i32
    %group = arith.divsi %pid, %width : i32
    %first = arith.muli %group, %g : i32
    %remaining = arith.subi %nm, %first : i32
    %size = arith.minsi %remaining, %g : i32
    %local = arith.remsi %pid, %width : i32
    %rm = arith.remsi %{pid}, %size : i32
    %pm = arith.addi %first, %rm : i32
    %pn = arith.divsi %local, %size : i32
    %om = arith.muli %pm, %bm : i32
    %on = arith.muli %pn, %bn : i32
    %kadd = arith.addi %K, %bk1 : i32
    %iters = arith.divsi %kadd, %bk : i32
    %zero = arith.constant dense<0.0> : tensor<{c}xf32>
    %result = scf.for %k = %c0 to %iters step %c1 iter_args(%acc = %zero) -> (tensor<{c}xf32>) : i32 {{
      %ok = arith.muli %k, %bk : i32
      %a = tt.descriptor_load %ad[%om, %ok] : !tt.tensordesc<tensor<{a}xf16>> -> tensor<{a}xf16>
      %b = tt.descriptor_load %bd[%ok, %on] : !tt.tensordesc<tensor<{b}xf16>> -> tensor<{b}xf16>
      %next = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<{a}xf16> * tensor<{b}xf16> -> tensor<{c}xf32>
      scf.yield %next : tensor<{c}xf32>
    }}
    %out = arith.truncf %result : tensor<{c}xf32> to tensor<{c}xf16>
    tt.descriptor_store %cd[%om, %on], %out : !tt.tensordesc<tensor<{c}xf16>>, tensor<{c}xf16>
    tt.return
  }}
}}
"""
    old = next(line for line in base.splitlines() if "tt.func public" in line)
    return base.replace(old, header, 1), value


def main():
    manifest = json.loads((MM / "baseline.json").read_text())
    records = {}
    for family, config in manifest["families"].items():
        source = MM / config["ttir_path"]
        assert sha(source) == config["ttir_sha256"]
        group = {}
        for name, pid in [("host_tma", "local"), ("host_pid", "pid")]:
            baseline, candidate = build(config, source.read_text(), pid)
            for label, text in [("baseline", baseline), (name, candidate)]:
                path = ROOT / "ir" / family / (label + ".ttir")
                store(path, text)
                group[label] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path))
        records[family] = group
    store(
        ROOT / "inputs_v4.json",
        json.dumps(
            dict(
                baseline_manifest_sha256=sha(MM / "baseline.json"),
                generator_sha256=sha(Path(__file__)),
                plan_sha256=sha(ROOT / "plan.md"),
                families=records,
            ),
            indent=2,
        )
        + "\n",
    )
    print("Frozen 6 matched baselines and 12 host-TMA candidates")


if __name__ == "__main__":
    main()
