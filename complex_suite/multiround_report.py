"""Assemble the five-operator r3-r5 report from qualified local selections."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPS = {
    "int4_gemm": "INT4 反量化＋GEMM",
    "moe_routing": "Grouped／biased MoE",
    "norm_quant": "Norm＋动态 INT8",
    "attention_dkdv": "Attention dK／dV",
    "partial_topk": "大 N partial Top-K",
}


def read(path):
    return json.loads(path.read_text())


def source(path):
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def geo(metrics, key):
    value = metrics[key]
    return value.get("geomean", value.get("geo"))


def main():
    records = {}
    evidence = []
    rounds = []
    for op in OPS:
        directory = ROOT / op / "multiround"
        summary_path = directory / "summary.json"
        summary = read(summary_path)
        evidence.append(source(summary_path))
        if op == "attention_dkdv":
            final = {
                run: {
                    "anchor_over_final": geo(
                        summary["final"][run], "anchor_over_final"
                    ),
                    "baseline_over_final": geo(
                        summary["final"][run], "baseline_over_final"
                    ),
                }
                for run in ("main", "repeat")
            }
            selected = "保留第二轮分组选择"
            count = 32
            for rid, stage in summary["rounds"].items():
                for group, data in stage["groups"].items():
                    rounds.append(
                        {
                            "operator": op,
                            "round": rid,
                            "group": group,
                            "candidate": data["tested"],
                            "main_incremental": geo(data["main"], "parent_over"),
                            "repeat_incremental": geo(data["repeat"], "parent_over"),
                            "accepted": data["accepted"],
                        }
                    )
            assert not any(summary["accepted_chain"].values())
        else:
            count = 30
            selected = "第二轮版本"
            stages = []
            for rid in ("r3", "r4", "r5"):
                path = directory / rid / "selection.json"
                data = read(path)
                evidence.append(source(path))
                stages.append(data)
                key = (
                    "incremental"
                    if op in ("int4_gemm", "moe_routing")
                    else "parent_over"
                )
                if data["accepted"]:
                    selected = f"{rid}/{data['candidate_tested']}"
                rounds.append(
                    {
                        "operator": op,
                        "round": rid,
                        "candidate": data["candidate_tested"],
                        "main_incremental": geo(data["main"], key),
                        "repeat_incremental": geo(data["repeat"], key),
                        "accepted": data["accepted"],
                    }
                )
            final = {}
            for run in ("main", "repeat"):
                data = stages[-1][f"selected_{run}"]
                anchor_key, base_key = (
                    ("anchor_relative", "cumulative")
                    if op in ("int4_gemm", "moe_routing")
                    else ("anchor_over", "baseline_over")
                )
                final[run] = {
                    "anchor_over_final": geo(data, anchor_key),
                    "baseline_over_final": geo(data, base_key),
                }
        records[op] = {
            "label": OPS[op],
            "shape_count": count,
            "selected": selected,
            "final": final,
            "report": f"{op}/multiround/report.md",
        }
    assert sum(item["shape_count"] for item in records.values()) == 152
    output = {
        "report_date": "2026-11-24",
        "round_ids": ["r3", "r4", "r5"],
        "unique_shapes": 152,
        "shape_round_evaluations": 456,
        "metric": "Geomean of per-shape medians of same-round paired time ratios",
        "final_measurement": "Actual selected routes in each operator's r5 main/repeat",
        "source_port_measured": False,
        "operators": records,
        "round_decisions": rounds,
        "attention_subsets": read(ROOT / "attention_dkdv/multiround/summary.json")[
            "final_subsets"
        ],
        "sources": evidence,
        "limitations": [
            "Fixed shapes, dtype, ABI, grid and compiler configuration; no new autotune.",
            "Attention measured at locked1200MHz; automatic clocks restored.",
            "MoE rejected r3/r4 finalists lack fresh sanitizer runs; accepted r5 is sanitized.",
            "Norm/TopK sanitizer checks cover all30 mappings with reduced row counts.",
            "No source-port performance or executable equivalence claim for r2-r5 changes.",
        ],
    }
    (ROOT / "complex_suite/multiround_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    )
    lines = [
        "# 五类算子连续三轮 TTIR 优化（r3–r5）",
        "",
        "报告更新：2026-11-24。第二轮之后继续完成三个顺序搜索轮，每轮均覆盖原有152个唯一性能点："
        "INT4、MoE、Norm、Top-K各30点，Attention32点。三轮共456次shape评估，不是456个不同shape。",
        "",
        "本次新增晋升为INT4的GROUP128常量特化、MoE的top2归约与整数常量组合、"
        "Top-K的合同特化。Norm与Attention的三轮新候选未达门槛，最终保留第二轮版本。",
        "",
        "## 最终有效版本",
        "",
        "所有数字均来自最后一轮同场计时：baseline为原始实现，anchor为第二轮验收实现。"
        "下表是各自固定矩阵内A级性能证据；原始baseline／最终版为累计收益，"
        "anchor／最终版为本次三轮之后的净增量。没有相乘历史加速比。",
        "",
        "| 算子 | 点数 | 最终版本 | 相对第二轮：主测／复测 | 相对原始baseline：主测／复测 |",
        "|---|---:|---|---:|---:|",
    ]
    for op, item in records.items():
        a, b = item["final"]["main"], item["final"]["repeat"]
        lines.append(
            f"| [{item['label']}](../{item['report']}) | {item['shape_count']} | {item['selected']} | "
            f"{a['anchor_over_final']:.6f}×／{b['anchor_over_final']:.6f}× | "
            f"{a['baseline_over_final']:.6f}×／{b['baseline_over_final']:.6f}× |"
        )
    subsets = output["attention_subsets"]
    lines += [
        "",
        "保留parent时使用本轮实际测得的parent耗时，接近1的anchor比值是重复计时波动，不是新收益。"
        "Attention完整32点包含D64和D128；独立复测相对原baseline分别为"
        f"D64的{subsets['d64']['repeat']['baseline_over_final']['geomean']:.6f}×（16点）和"
        f"D128的{subsets['d128']['repeat']['baseline_over_final']['geomean']:.6f}×（16点）。"
        "这些结果均在SM1200MHz下测得，不能与旧自动频率实验的绝对时间混用。",
        "",
        "## 每轮实际尝试与决策",
        "",
        "以下为冻结进入正式确认的候选相对当轮parent的直接增量，负结果也保留。"
        "只有主测和独立复测均满足几何均值≥1.02、最差点≥0.95，"
        "并通过数值、A/A和sanitizer，才替换parent。",
        "",
        "| 算子／组 | 轮次 | 正式候选 | 主测增量 | 复测增量 | 决策 |",
        "|---|---|---|---:|---:|---|",
    ]
    for item in rounds:
        label = OPS[item["operator"]]
        if "group" in item:
            label += " / " + item["group"]
        decision = "接受" if item["accepted"] else "保留parent"
        lines.append(
            f"| {label} | {item['round']} | `{item['candidate']}` | "
            f"{item['main_incremental']:.6f}× | {item['repeat_incremental']:.6f}× | {decision} |"
        )
    lines += [
        "",
        "## 优化了什么，Triton能否表达",
        "",
        "| 算子 | 三轮探索与结果 | 公开Triton表达方式 |",
        "|---|---|---|",
        "| INT4 | r3把既定GROUP128的非负除法改为右移7位并晋升；r4的PID分组、"
        "r5的预加载／两group展开均未进一步获益 | 常量或`tl.constexpr`、整数右移、"
        "`tl.program_id`重排、循环与加载顺序 |",
        "| MoE | r3的gather属性／选择展开不获益；r4单独tuple top2复测1.0194×未达门槛；"
        "r5将其与固定G、topK、stride等整数常量组合后晋升 | `tl.reduce`自定义双值组合器、"
        "`tl.constexpr`或已知常量。未晋升的`efficient_layout`属性在3.7的`tl.gather`无直接公开参数 |",
        "| Norm | r3合并或消去量化gain；r4混合sum／uint max归约；r5重载或1024分块输出，"
        "均未达门槛 | 标量算术、`tl.reduce`、bitcast、循环和`tl.load` |",
        "| Attention | r3组合展开／指针重算或查询迭代；r4调整Q/dO缓存和delta读取；"
        "r5调整存储／驱逐提示，均未晋升 | 循环、表达式次序、`tl.load`／`tl.store`缓存与驱逐参数 |",
        "| Top-K | r3分层uint32／uint64选择、r4批量store退步；r5在有限值、固定grid及"
        "完整块条件下删除恒真mask／耗尽判断后晋升，尾块保护保留 | `tl.constexpr`分支、"
        "逻辑索引、`tl.load` mask、归约与reshape |",
        "",
        "上述可表达性是结构与API对应关系，不是新的源码性能实验。第一轮41点源码回写曾测得"
        "与TTIR相同的可执行ELF `.text`；第二轮及r3–r5的新改动尚未做源码回写计时，"
        "不能直接继承其机器指令或性能相等结论。",
        "",
        "## 有价值的负结果与限制",
        "",
        "MoE的单独top2归约没有达到晋升门槛，第三轮将它与常量特化重新组合后才获得稳定收益；"
        "这是新的组合实测。INT4消除变量除法时寄存器与spill也改变，不能把收益全部归因于一条指令。",
        "",
        "Attention查询展开增加活跃状态，部分组spill和shared显著上升并回退；"
        "Q/dO的`.cg`以及`.ca+evict_last`候选虽然改变上层属性，四组最终可执行`.text`仍与parent相同。"
        "这些是本编译器与配置下的静态证据，不代表缓存提示普遍无效，也没有证明实际occupancy或动态瓶颈。",
        "",
        "Norm没有因更少的算术或更窄的输出循环而获得整体提升；Top-K分层归约与批量存储也慢于parent。"
        "因此只保留通过完整矩阵门槛的版本，不按复测结果事后挑选单shape赢家。",
        "",
        "## 验证与复现范围",
        "",
        "官方Triton3.7.0／H100，保持第二轮dtype、固定编译配置、ABI、grid、kernel数与workspace，"
        "未新增autotune或pipeline-depth调参。每轮先做正确性和原始源码／文件baseline的"
        "LLVM、PTX、cubin一致性检查，再初筛、冻结选择、主测7轮、独立进程复测5轮。",
        "",
        "正式计时为预分配CUDA graph设备时间，随机交错同输入配对，`rep=20ms`为计时库参数，"
        "保留外层全部样本；每点A/A中位要求在[0.97,1.03]。Attention等条件预热并锁SM1200MHz，"
        "结束后已恢复自动频率；其他四类不主动锁频。",
        "",
        "最终新增晋升均有memcheck／racecheck零错误证据。INT4、Norm、Top-K、Attention"
        "三轮正式候选均有新sanitizer记录；MoE仅晋升的r5新增sanitizer，r3/r4未晋升候选"
        "没有新sanitizer，保留的parent沿用已有资格。Norm／Top-K的sanitizer覆盖30个映射，"
        "为控制开销使用缩减行数；不是每个压力分布都验证原始完整M。具体数值合同、覆盖范围与哈希见分报告。",
        "",
        "数值合同各不相同：Norm沿用量化code误差、scale／重构误差及输出范围检查；"
        "Top-K要求稳定索引和值数值相等，±0等价。未宣称二者所有输出均保持原始位型。",
        "",
        "[共同协议](multiround_plan.md) · [机器可读汇总](multiround_summary.json) · "
        "[第二轮152点报告](round2_report.md) · [第一轮源码回写实测](source_port_comparison.md)",
        "",
        "复现汇总：`/root/venvs/ir-ops-dev/bin/python complex_suite/multiround_report.py`。"
        "各算子报告链接给出原始配对样本、选择与IR哈希、候选生成器及GPU复现命令。",
        "",
    ]
    if (ROOT / "complex_suite/round6_report.md").exists():
        lines[2:2] = [
            "> 后续已完成[第六轮TTIR优化](round6_report.md)，新增MoE、Attention D128与Top-K收益。"
            "本文保留r3–r5结束时的历史结果。",
            "",
        ]
    (ROOT / "complex_suite/multiround_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
