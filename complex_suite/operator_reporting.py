"""Build current operator reports without mixing historical matrices."""

import json
import math
import statistics


def render(op, title, contract, steps, negatives):
    selection = json.loads((op / "multiround/r6/selection.json").read_text())
    paths = [
        op / "multiround/r6/results" / phase / "results.json"
        for phase in ("main", "repeat")
    ]
    main, repeat = [json.loads(p.read_text()) for p in paths]
    name = selection["selected"]
    for x in (main, repeat):
        assert x["passed"] and x["aa_passed"] and not x["failures"]
        assert len(x["records"]) == 30
    values = [
        [r["timing"]["baseline_over"][name] for r in x["records"]]
        for x in (main, repeat)
    ]
    geos = [math.exp(statistics.mean(math.log(v) for v in vs)) for vs in values]
    lines = [
        f"# {title}：直接 TTIR 优化",
        "",
        "## 当前最终版本与实验契约",
        "",
        contract,
        "",
        "当前最终IR为`multiround/r6/ir/<group>/optimized.ttir`，"
        "根目录`ir/optimized.ttir`及`selection.json`属于首轮历史，不能作为当前复现选择。"
        "[当前选择与IR哈希](multiround/r6/selection.json) · "
        "[30点冻结矩阵和配置](round2/expanded/manifest.json)",
        "",
        "## 环境与计时协议",
        "",
        "官方Triton3.7.0、H100 80GB；完整backend options与各点baseline相同。"
        "主测7轮、独立进程seed123复测5轮，rep20ms；随机交错、每shape每轮A/A。"
        "计时前源码/文件baseline的LLVM/PTX/cubin和options逐组一致、全部候选数值通过。"
        "预分配CUDA graph设备序列，不含编译、分配、CPU launch；未主动锁频或清cache。",
        "",
        "## 最终版完整30点结果",
        "",
        f"第六轮最终有效路径相对原始baseline：几何均值 **{geos[0]:.6f}× / {geos[1]:.6f}×**，"
        f"最差点{min(values[0]):.6f}× / {min(values[1]):.6f}×（主测/复测，A级）。"
        "每shape取同轮时间比的中位数，再跨shape求几何平均；不乘历史加速比。",
        "",
        "| shape | baseline µs（复测） | 最终 µs（复测） | 主测加速 | 独立复测 |",
        "|---|---:|---:|---:|---:|",
    ]
    for a, b in zip(main["records"], repeat["records"]):
        assert a["shape"] == b["shape"]
        t = b["timing"]
        lines.append(
            f"| {'×'.join(map(str, a['shape']))} | {t['median_us']['baseline']:.4f} | "
            f"{t['median_us'][name]:.4f} | {a['timing']['baseline_over'][name]:.6f}× | "
            f"{t['baseline_over'][name]:.6f}× |"
        )
    lines.extend(
        [
            "",
            "时延中位数之比不必等于配对比值中位数。"
            f"[主测]({paths[0].relative_to(op)}) · [独立复测]({paths[1].relative_to(op)}) · "
            "[第六轮完整验证和资源](multiround/r6/report.md)",
            "",
            "## 优化过程和最终组合消融",
            "",
            steps,
            "",
            "[独立消融报告](report_ablation.md)按照LayerNorm形式记录各版本相对原baseline、"
            "逐项加入、最终移除、完整shape、资源与复现。新消融是保持同一最终版的另一次会话，"
            "数字不混入上面的第六轮正式结果，也没有根据新对照重选最终实现。"
            "[预声明消融计划](ablation_plan.md)",
            "",
            "## 数值验证、负结果与解释边界",
            "",
            negatives,
            "",
            "最终路线已有r6 memcheck零错误、racecheck零错误/零警告的存证。"
            "normal检查完整M，其他压力分布及sanitizer按原验证器缩减行数；"
            "不将此描述为所有原始M都完整跑过sanitizer。新增消融没有额外跑全部sanitizer，"
            "实际检查范围在独立消融报告中列明。性能仅限本矩阵/研究baseline，"
            "不代表冷cache、生产端到端或最强库比较。",
            "",
            "第一轮曾执行源码回写，[源码对照](source_port/report.md)仅适用于原小矩阵/首轮改动。"
            "当前最终版的公开Triton表达能力和源码版性能/机器指令一致性是不同问题；"
            "后两者本轮未重新验证。",
            "",
            "## 复现与历史",
            "",
            f"仓库根设置独占`CUDA_VISIBLE_DEVICES`后运行`bash {op.name}/run_experiment.sh`。"
            "它核对候选不变，依次执行全部pipeline/正确性、主测、独立复测和报告生成。"
            "设置新`RUN_ID`保存重复实验，已有结果目录拒绝覆盖；命令细节见[README](README.md)。",
            "",
            "[首轮合同](optimization_plan.md) · [首轮报告](report_round1.md) · "
            "[第二轮](round2/report.md) · [第三至五轮](multiround/report.md) · "
            "[第六轮](multiround/r6/report.md)",
            "",
        ]
    )
    (op / "report.md").write_text("\n".join(lines))
