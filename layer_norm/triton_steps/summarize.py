"""Audit source witness receipts and render their own performance record."""

import argparse
import csv
import hashlib
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads((args.run / "results.json").read_text())
    manifest = json.loads((ROOT / "manifest.json").read_text())
    assert data["passed"] and data["accuracy_passed"] and data["aa_passed"]
    assert len(data["accuracy"]) == 262 * 10 and all(
        r["passed"] for r in data["accuracy"]
    )
    assert len(data["pipeline"]) == len(manifest["variants"]) == 9
    assert len(data["benchmark"]) == 30
    assert data["source_manifest_sha256"] == sha(ROOT / "manifest.json")
    for name, entry in manifest["variants"].items():
        kernel = data["pipeline"][name]
        assert kernel["source_file_identical"]
        assert kernel["source_sha256"] == entry["sha256"] == sha(ROOT / entry["source"])
        assert kernel["ttir_sha256"] == sha(args.run / f"{name}.ttir")
    lines = [
        "# Triton 逐项源码示例验证",
        "",
        "九种源码表达和原始冻结TTIR基线共十条路径，每条完成142个原验证用例及120个范围边界用例，"
        "合计2620项检查通过。包含native ATen输出、FP64均值/方差倒数参考、padding canary和输入不修改。",
        "",
        "全部九个源码版本正常JIT后导出TTIR，文件入口重编译的LLVM/PTX/cubin与各自源码入口逐字节一致；"
        "这一检查不代表这些源码与历史手写TTIR逐字节相同。原教程编译链一致性另行通过。",
        "",
        "H100，官方Triton3.7.0，固定8warps/3stages/1CTA，原FP16合同。"
        "所有数值检查先于性能。30点M4096、N1024…15872步长512，5轮rep10ms、"
        "每版100ms预热、随机顺序配对和每点A/A；仅预分配CUDA graph kernel时间。",
        "",
        "| 版本 | 相对冻结原始baseline的几何加速 | registers | spills |",
        "|---|---:|---:|---:|",
    ]
    csv_rows = []
    for name in manifest["variants"]:
        values = []
        for row in data["benchmark"]:
            assert 0.97 <= row["timing"]["aa_median"] <= 1.03
            samples = row["timing"]["samples"][name]
            assert len(samples) == 5
            ratios = []
            for sample in samples:
                ratio = sample["us"]["frozen_baseline"] / sample["us"][name]
                assert math.isclose(sample["speedup"], ratio, rel_tol=1e-12)
                ratios.append(ratio)
            speed = st.median(ratios)
            values.append(speed)
            csv_rows.append(
                [
                    name,
                    row["M"],
                    row["N"],
                    st.median(s["us"]["frozen_baseline"] for s in samples),
                    st.median(s["us"][name] for s in samples),
                    speed,
                ]
            )
        geometric = st.geometric_mean(values)
        assert math.isclose(
            data["benchmark_summary"][name]["geometric"], geometric, rel_tol=1e-12
        )
        kernel = data["pipeline"][name]
        lines.append(
            f"| {name} | {geometric:.6f}× | {kernel['n_regs']} | {kernel['n_spills']} |"
        )
    with (args.run / "all_shapes.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["version", "M", "N", "baseline_us", "optimized_us", "paired_speedup"]
        )
        writer.writerows(csv_rows)
    aa = [r["timing"]["aa_median"] for r in data["benchmark"]]
    lines.extend(
        [
            "",
            f"A/A中位数范围[{min(aa):.6f}, {max(aa):.6f}]，全部通过。",
            "",
            "本表为新源码示例的单会话完整矩阵观测（B级），没有独立进程复测，不替换此前TTIR正式实验加速比。"
            "寄存器/spill是C级静态资源，不能单独解释性能原因；未对全部新源码版本重跑sanitizer。",
            "",
            "[完整结果和哈希](results.json) · [全部shape CSV](all_shapes.csv)",
            "",
        ]
    )
    (args.run / "report.md").write_text("\n".join(lines))
    print(
        "Verified 9 source witnesses, 2620 numerical checks and 30 paired performance points"
    )


if __name__ == "__main__":
    main()
