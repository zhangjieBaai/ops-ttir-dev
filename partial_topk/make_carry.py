"""Keep ordered score keys in the loop-carried SSA state; encode only once."""

import json
import re
import sys
from pathlib import Path

from make_ordered import ordered

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from complex_suite.support import digest, freeze, save  # noqa: E402


def carry(base):
    text = ordered(base)
    loop = re.search(
        r"    %ids_\d+:2 = scf.for.*?iter_args\((%values_\d+) = (%values_\d+),.*?-> \(tensor<(\d+)xf32>,",
        text,
    )
    current, initial, b = loop.groups()
    begin = text.index("      %bits =")
    end = text.index('      %maxkey = "tt.reduce"', begin)
    init = (
        text[begin:end]
        .replace(current, "%normalized_values")
        .replace("%key =", "%init_keys =")
    )
    init = (
        f"""    %zerof = arith.constant dense<0.0> : tensor<{b}xf32>
    %iszero = arith.cmpf oeq, {initial}, %zerof : tensor<{b}xf32>
    %normalized_values = arith.select %iszero, %zerof, {initial} : tensor<{b}xi1>, tensor<{b}xf32>
    %sentinel_key = arith.constant dense<8388607> : tensor<{b}xi32>
"""
        + init
    )
    text = text[:begin] + text[end:]
    text = text[: loop.start()] + init + text[loop.start() :]
    text = text.replace(f"{current} = {initial}", f"{current} = %init_keys")
    text = text.replace(
        f"-> (tensor<{b}xf32>, tensor<{b}xi32>)",
        f"-> (tensor<{b}xi32>, tensor<{b}xi32>)",
    )
    text = text.replace('"tt.reduce"(%key)', f'"tt.reduce"({current})')
    text = re.sub(
        rf"%matches = tt.splat %best : f32 -> tensor<{b}xf32>",
        f"%matches = tt.splat %maxkey : i32 -> tensor<{b}xi32>",
        text,
    )
    text = re.sub(
        rf"arith.cmpf oeq, {current}, %matches : tensor<{b}xf32>",
        f"arith.cmpi eq, {current}, %matches : tensor<{b}xi32>",
        text,
    )
    text = re.sub(
        rf"(arith.select %selected_\d+, )%cst(?:_\d+)?(, {current} : tensor<{b}xi1>, )tensor<{b}xf32>",
        rf"\1%sentinel_key\2tensor<{b}xi32>",
        text,
    )
    text = re.sub(
        rf"(scf.yield %values_\d+, %ids_\d+ : )tensor<{b}xf32>",
        rf"\1tensor<{b}xi32>",
        text,
    )
    return text


def main():
    records = []
    for row in json.loads((ROOT / "manifest.json").read_text())["records"]:
        for stage in ("local", "merge"):
            folder = ROOT / "ir" / row["group"] / stage
            path = folder / "ordered_carry_v3.ttir"
            freeze(path, carry((folder / "baseline.ttir").read_text()))
            records.append(
                {
                    "group": row["group"],
                    "stage": stage,
                    "baseline_sha256": digest(folder / "baseline.ttir"),
                    "sha256": digest(path),
                }
            )
    save(ROOT / "carry_candidates.json", records)


if __name__ == "__main__":
    main()
