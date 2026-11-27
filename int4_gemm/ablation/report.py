"""Render the operator report from paired full-matrix timings."""

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_run(path, rounds):
    data = json.loads(path.read_text())
    assert data["passed"] and data["aa_passed"] and not data["failures"]
    assert data["rounds"] == rounds and data["rep_ms"] == 20
    by_shape = {}
    for row in data["records"]:
        shape = tuple(row["shape"])
        samples = row["timing"]["samples"]
        assert len(samples) == rounds and shape not in by_shape
        assert 0.97 <= row["timing"]["aa_median"] <= 1.03
        by_shape[shape] = [sample["us"] for sample in samples]
    assert len(by_shape) == 30
    return by_shape


def ratios(run, before, after):
    return {
        shape: statistics.median(t[before] / t[after] for t in samples)
        for shape, samples in run.items()
    }


def main():
    final = []
    ablation = []
    for phase, rounds in (("main", 7), ("repeat", 5)):
        final.append(
            ratios(
                read_run(ROOT / f"multiround/r6/results/{phase}/results.json", rounds),
                "baseline",
                "parent",
            )
        )
        ablation.append(
            read_run(ROOT / f"round2/results/ablation_v1_{phase}/results.json", rounds)
        )
    assert (
        final[0].keys() == final[1].keys() == ablation[0].keys() == ablation[1].keys()
    )
    geo = [statistics.geometric_mean(run.values()) for run in final]
    mean = [statistics.mean(run.values()) for run in final]
    rows = [
        "# INT4 反量化融合 GEMM 的 TTIR 优化",
        "",
        "来源为 FlagGems 的 weight_int4pack_mm_with_scales_and_zeros。冻结基线在同一 kernel 内完成 nibble 解包、"
        "按 128 个 K 元素一组反量化及 GEMM。输入为 FP16 激活和 uint8 打包权重，反量化在 FP32 计算后舍入至 FP16，"
        "dot 使用 FP32 累加并输出 FP16。[源码快照](upstream/source.py.snapshot) · [基线配置](round2/expanded/manifest.json)",
        "",
        "环境为 H100、官方 Triton 3.7.0。各 shape 的 tile、grid 和编译选项冻结。预分配 CUDA graph 测单 kernel 设备耗时，"
        "主测 7 轮、独立进程复测 5 轮，每轮 rep=20ms。第六轮同场比较原始基线和最终保留的第三轮 group128。",
        "",
        f"**平均加速比（A级，30 点）**　主测几何均值 {geo[0]:.4f}×，独立复测 {geo[1]:.4f}×；"
        f"算术均值分别为 {mean[0]:.4f}×、{mean[1]:.4f}×。每个 shape 对同轮耗时比取中位数。",
        "",
        "| Shape (M, N, K) | 主测加速比 | 独立复测加速比 |",
        "|---|---:|---:|",
    ]
    for shape in sorted(final[0]):
        rows.append(f"| {shape} | {final[0][shape]:.4f}× | {final[1][shape]:.4f}× |")
    rows += [
        "",
        "## 优化方法与条件收益",
        "",
        "首轮将同一 byte 的两支 scale/zero 合并，逐 K tile 标量化量化组索引，对已证非负的索引使用无符号运算。"
        "冻结 7 点的组合加速为 1.0948× / 1.0958×。第二轮扩至 30 点，以下消融使用同一会话内的逐轮配对观测。"
        "parent 为首轮组合在 30 点上的实现。[首轮证据](report_round1.md) · [第二轮 TTIR 变换](round2/make_candidates.py)",
        "",
        "| 对照（起点 → 终点） | 主测几何均值 | 独立复测 |",
        "|---|---:|---:|",
    ]
    for label, before, after in [
        ("原始基线 → 首轮组合", "baseline", "parent"),
        ("parent → 缩窄 scale/zero 加载后广播", "parent", "r2_narrow"),
        ("parent → 去掉恒真的 K mask", "parent", "r2_full_k"),
        ("已缩窄加载 → 再去 K mask", "r2_narrow", "r2_narrow_full_k"),
        ("已去 K mask → 再缩窄加载", "r2_full_k", "r2_narrow_full_k"),
        ("两项均启用 → BK64 两次 dot 共用组参数", "r2_narrow_full_k", "r2_group_pair"),
        ("parent → 完整第二轮组合", "parent", "r2_group_pair"),
    ]:
        values = [
            statistics.geometric_mean(ratios(r, before, after).values())
            for r in ablation
        ]
        rows.append(f"| {label} | {values[0]:.4f}× | {values[1]:.4f}× |")
    rows += [
        "",
        "组内复用保持两个 dot 的顺序，BK128 路径在这一步保持相同 TTIR。第三轮利用组宽 128 将除法替换为右移，"
        "对第二轮路径的主测 / 复测增量为 1.0306× / 1.0302×。第四至六轮保留该实现；第六轮整数构造 FP32 的候选"
        "相对该实现仅 0.9536× / 0.9537×。[第三轮](multiround/r3/report.md) · [第六轮原始报告](multiround/r6/report.md)",
        "",
        "新消融在 GPU7 上执行。各会话 30 个 shape × 6 条路径 × 5 种输入分布，与基线输出逐位相同，"
        "并通过 FP32 参考、小尺寸 CPU FP64 参考、输出 canary 和输入不变检查。每个 shape 的基线源码与 TTIR 文件入口"
        "生成相同 LLVM/PTX/cubin；全部 A/A 中位数落在 [0.97, 1.03]。第六轮已对正式路径执行 memcheck/racecheck。",
        "",
        "[消融协议与复现命令](ablation/plan.md) · [消融主测样本和哈希](round2/results/ablation_v1_main/results.json) · "
        "[消融独立复测](round2/results/ablation_v1_repeat/results.json) · [第六轮选择与原始数据索引](multiround/r6/report.md)",
    ]
    (ROOT / "report.md").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
