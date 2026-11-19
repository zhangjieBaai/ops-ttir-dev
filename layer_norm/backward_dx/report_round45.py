"""Report measured round-four selection and round-five rejected structural candidates."""

import json
import math
import statistics

from common import ROOT, save

R = ROOT / "results/round4"


def geo(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


def main():
    decision = json.load(open(R / "selection_final.json"))
    rows = [
        "# backward dX 第四、五轮：目标均值 1.10×",
        "",
        "延续30个shape、FP16输入输出、FP32计算、官方Triton3.7.0固定options/grid/ABI和原始baseline。"
        "第四轮12个算法候选（另有2份C0命名修复文件），第五轮4个内部多行调度候选，共16种新方案。",
        "",
        "## 完整矩阵与最终结果（A级）",
        "",
        "| 测量 | 最终组合 / baseline 几何 | 算术 | 最终组合 / parent 几何 | A/A |",
        "|---|---:|---:|---:|---|",
    ]
    summary = {}
    for label, file in [("主测", "full.json"), ("独立复测", "repeat.json")]:
        data = json.load(open(R / file))
        assert data["aa_passed"]
        parent = {v["N"]: v for v in data["records"] if v["kernel"] == "r4_parent"}
        selected = []
        for v in data["records"]:
            b = str(v["block"])
            name = (
                decision["families"][b]["candidate"]
                if v["block"] in decision["promoted_blocks"]
                else "r4_parent"
            )
            if v["kernel"] == name:
                selected.append(v)
        assert len(selected) == 30
        speedups = [v["speedup"] for v in selected]
        s = dict(
            geometric_mean=geo(speedups),
            arithmetic_mean=statistics.mean(speedups),
            vs_parent_geometric=geo(
                [parent[v["N"]]["candidate_us"] / v["candidate_us"] for v in selected]
            ),
        )
        s["target_met"] = s["geometric_mean"] > 1.10 and s["arithmetic_mean"] > 1.10
        summary[file] = s
        rows.append(
            f"| {label} | {s['geometric_mean']:.4f}× | {s['arithmetic_mean']:.4f}× | "
            f"{s['vs_parent_geometric']:.4f}× | 30/30通过 |"
        )
    all_met = all(v["target_met"] for v in summary.values())
    rows += [
        "",
        f"**是否达到用户要求的平均提升超过10%：{'是' if all_met else '否'}。**",
        f"本轮晋升family：{decision['promoted_blocks']}。仅局部晋升不等于整体目标达成。",
        "主测7轮rep20ms，独立seed123/5轮rep10ms；先按主测冻结候选，再用独立复测验收。"
        "最终组合按各shape实际选中候选汇总；vs parent为同会话中位耗时之比，并非伪造直接配对样本。",
        "",
        "| family | 当前版本 |",
        "|---|---|",
    ]
    for block, v in decision["families"].items():
        rows.append(f"| b{block} | {v['candidate']} |")
    rows += [
        "",
        "## 第四轮初筛（B级）",
        "",
        "初筛仅6个代表点、3轮10ms，不能用来声称最终全矩阵加速。下表都相对原始baseline；原始样本含同会话parent。",
        "",
        "| 候选 | 几何均值 | 最差点 | b16384 registers/spills/shared |",
        "|---|---:|---:|---|",
    ]
    resources = json.load(open(R / "binary_analysis.json"))
    for filename in ["screen_batch1.json", "screen_batch2.json", "screen_batch3.json"]:
        data = json.load(open(R / filename))
        assert data["aa_passed"]
        for name, v in data["summary"].items():
            if name == "r4_parent":
                continue
            r = resources[f"{name}/b16384"]
            rows.append(
                f"| {name} | {v['geometric_mean']:.4f}× | {v['worst']:.4f}× | "
                f"{r['n_regs']}/{r['n_spills']}/{r['shared']}B |"
            )
    rows += [
        "",
        "r4_affine虽然通过135个常规用例，但72个额外压力用例中3个失败，已淘汰，未进行计时。"
        "它将中心化表达式展开为两个FMA，大偏移/低方差时抵消误差不能用低平均RMS掩盖。",
        "",
        "## 第五轮多行调度（B级负结果）",
        "",
        "外部grid仍为(M,1,1)，内部由get_num_programs取得M，仅前ceil(M/R)个CTA处理R行，其他退出。"
        "行边界和列边界均mask。相对于原先每CTA一行，这是单独声明的调度变化，未改变compiler/options/launcher。",
        "",
        "| 候选 | 6点几何 / baseline | 最差 | b16384 registers/spills/shared |",
        "|---|---:|---:|---|",
    ]
    data = json.load(open(ROOT / "results/round5/screen.json"))
    assert data["aa_passed"]
    for name, v in data["summary"].items():
        if name == "r4_parent":
            continue
        r = resources[f"{name}/b16384"]
        rows.append(
            f"| {name} | {v['geometric_mean']:.4f}× | {v['worst']:.4f}× | "
            f"{r['n_regs']}/{r['n_spills']}/{r['shared']}B |"
        )
    rows += [
        "",
        "8行×256列版本的TTGIR使用warpsPerCTA=[8,1]，共享内存为0；"
        "仍未取得收益。完整双行向量在b16384产生大量spill，也未晋升。未将这些6点数据称为A级结果。",
        "",
        "## NCU与产物诊断（C级）",
        "",
        "base clock、cache-control=all，单次launch；不是常规CUDA graph时间，不能混算加速比。",
        "",
        "| N | 版本 | NCU µs | DRAM读 MB | DRAM写 MB | active warps % | long-scoreboard / issue |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for v in json.load(open(R / "ncu_summary.json")):
        if v["kernel"] == "baseline":
            continue

        def value(k):
            return v[k]["value"]

        rows.append(
            f"| {v['N']} | {v['kernel']} | {value('gpu__time_duration.sum')/1000:.3f} | "
            f"{value('dram__bytes_read.sum')/1e6:.2f} | {value('dram__bytes_write.sum')/1e6:.2f} | "
            f"{value('sm__warps_active.avg.pct_of_peak_sustained_active'):.2f} | "
            f"{value('smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio'):.2f} |"
        )
    rows += [
        "",
        "局部标量化、cache提示和降寄存器都不是充分条件。折叠规约还引入了16KB共享内存，"
        "而原b8192只有32B；保留所有分组状态将b16384推到254/255寄存器；这些产物变化解释了为何不能只看TTIR代码更短。"
        "确切单因素因果仍未完全拆解，不将任何一个指标当成性能证明。",
        "",
        "## 正确性、记录与复现",
        "",
        "- 四轮首批8版本×135=1080次检查、第二批6×135=810、第三批4×135=540；第五轮6×135=810。"
        "所有有效候选均做72例saved-stats压力检查；affine按原阈值失败。",
        "- 入围版本额外48边界、seed123全用例及memcheck/racecheck；最终别名重新做pipeline、准确性和直接性能。",
        "- C0命名冲突保留原始失败IR/日志，使用新名字_v2修复；所有被拒绝候选及原始数据保留。",
        "- baseline、冻结教程源码均未修改。upstream Triton3.7.0；无FlagTree、独立llc、inline PTX、低层IR替换。",
        "",
        "主要证据：[selection_final](results/round4/selection_final.json)、[主测](results/round4/full.json)、"
        "[独立复测](results/round4/repeat.json)、[NCU](results/round4/ncu_summary.json)、"
        "[第五轮初筛](results/round5/screen.json)。",
    ]
    direct = json.load(open(R / "selected_benchmark.json"))
    assert direct["aa_passed"] and len(direct["records"]) == 30
    stats = direct["summary"]["optimized"]
    rows += [
        "",
        "## 最终 optimized 入口直接复核",
        "",
        f"3轮rep10ms，完整30点A/A全部通过；几何{stats['geometric_mean']:.4f}×，"
        f"算术{stats['arithmetic_mean']:.4f}×。这是最终入口验证，晋升仍依据7轮主测和5轮独立复测。",
        "",
        "| Shape | baseline µs | optimized µs | baseline/optimized |",
        "|---|---:|---:|---:|",
    ]
    for v in direct["records"]:
        rows.append(
            f"| ({v['M']}, {v['N']}) | {v['baseline_us']:.3f} | {v['candidate_us']:.3f} | {v['speedup']:.4f}× |"
        )
    rows += [
        "",
        "比值为配对样本比值中位数，可能与两个耗时中位数直接相除有微小差异。",
        "",
        "[最终数据](results/round4/selected_benchmark.json)、[最终编译链](results/round4/pipeline_final.json)。",
    ]
    summary["direct_final"] = stats
    (ROOT / "report_round45.md").write_text("\n".join(rows) + "\n")
    save(R / "composite_summary.json", dict(target_met=all_met, measurements=summary))


if __name__ == "__main__":
    main()
