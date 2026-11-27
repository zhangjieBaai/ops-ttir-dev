"""Consolidate qualified round6 results without combining historical ratios."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPS = {
    "int4_gemm": "INT4反量化＋GEMM",
    "moe_routing": "Grouped／biased MoE",
    "norm_quant": "Norm＋动态INT8",
    "attention_dkdv": "Attention dK／dV",
    "partial_topk": "大N partial Top-K",
}


def main():
    sources, records = {}, {}
    for op, label in OPS.items():
        path = ROOT / op / "multiround/r6/summary.json"
        data = json.loads(path.read_text())
        assert data["source_port_measured"] is False
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        final = {}
        if op == "attention_dkdv":
            assert data["clocks"]["automatic_restored"]
            assert all(len(data["final_rows"][p]) == 32 for p in ("main", "repeat"))
            for phase in ("main", "repeat"):
                final[phase] = {
                    f"{key}_over_final": data["final"][phase][f"{key}_over_final"][
                        "geomean"
                    ]
                    for key in ("parent", "anchor", "baseline")
                }
            accepted = {g: x["accepted"] for g, x in data["groups"].items()}
            selected = {g: x["selected"] for g, x in data["groups"].items()}
            details = {
                "groups": data["groups"],
                "subsets": data["final_subsets"],
                "clocks": data["clocks"],
            }
        else:
            assert data["shape_count"] == 30
            for phase in ("main", "repeat"):
                metrics = (
                    data[f"final_{phase}"]
                    if op == "int4_gemm"
                    else data[f"selected_{phase}"]
                )
                keys = (
                    ("incremental", "anchor_relative", "cumulative")
                    if op in ("int4_gemm", "moe_routing")
                    else ("parent_over", "anchor_over", "baseline_over")
                )
                final[phase] = {
                    f"{key}_over_final": metrics[field]["geomean"]
                    for key, field in zip(("parent", "anchor", "baseline"), keys)
                }
            accepted, selected = data["accepted"], data["selected"]
            key = "incremental" if op in ("int4_gemm", "moe_routing") else "parent_over"
            details = {
                "candidate_tested": data["candidate_tested"],
                "candidate_main": data["main"][key],
                "candidate_repeat": data["repeat"][key],
            }
        records[op] = {
            "label": label,
            "shape_count": 32 if op == "attention_dkdv" else 30,
            "accepted": accepted,
            "selected": selected,
            "final": final,
            "details": details,
        }
    output = {
        "round": 6,
        "unique_shapes": 152,
        "date": "2026-11-24",
        "operators": records,
        "sources": sources,
        "source_port_measured": False,
        "metric": "Geomean of per-shape median paired ratios; final effective routes measured in r6",
    }
    (ROOT / "complex_suite/round6_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    )
    rows = [
        "# 五类算子第六轮TTIR优化",
        "",
        "更新：2026-11-24。在r3–r5完成后继续进行新一轮结构搜索，仍覆盖152个唯一shape（四类各30、Attention32）。"
        "本轮从r5最终验收版本出发，只改变TTIR，不改变dtype、编译配置、ABI、grid、kernel数或workspace。",
        "",
        "## 最终有效路线的直接实测",
        "",
        "parent指r5最终有效选择，anchor指第二轮验收选择，baseline指原始冻结实现。"
        "下面的主测／独立复测均来自第六轮同场随机配对，未相乘历史加速比。",
        "",
        "| 算子 | 点数 | 相对r5 parent：主测／复测 | 相对原始baseline：主测／复测 |",
        "|---|---:|---:|---:|",
    ]
    for op, record in records.items():
        m, r = record["final"]["main"], record["final"]["repeat"]
        delta = (
            f"{m['parent_over_final']:.6f}×／{r['parent_over_final']:.6f}×"
            if (
                any(record["accepted"].values())
                if isinstance(record["accepted"], dict)
                else record["accepted"]
            )
            else "保留parent"
        )
        rows.append(
            f"| [{record['label']}](../{op}/multiround/r6/report.md) | {record['shape_count']} | {delta} | "
            f"{m['baseline_over_final']:.6f}×／{r['baseline_over_final']:.6f}× |"
        )
    rows += [
        "",
        "没有晋升时使用本轮实测parent时间计算累计收益；相同代码在不同测量轮次的微小波动不算新增收益。"
        "Attention仅在原D/causal分组上决定晋升，全32点平均包含D64的未改路径，具体如下。",
        "",
        "| Attention组 | 正式候选 | 主测增量 | 复测增量 | 决策 |",
        "|---|---|---:|---:|---|",
    ]
    attention = records["attention_dkdv"]["details"]
    for group, rec in attention["groups"].items():
        m = rec["main"]["parent_over"]["geo"]
        r = rec["repeat"]["parent_over"]["geo"]
        rows.append(
            f"| {group} | {rec['tested']} | {m:.6f}× | {r:.6f}× | {'接受' if rec['accepted'] else '保留parent'} |"
        )
    rows += [
        "",
        "D64表中的候选名称对应与parent字节相同的控制文件，没有对D64执行拆分遍历。"
        f"D128合计16点的复测增量为{attention['subsets']['d128']['repeat']['parent_over_final']['geomean']:.6f}×，"
        f"累计为{attention['subsets']['d128']['repeat']['baseline_over_final']['geomean']:.6f}×；"
        f"D64的16点累计为{attention['subsets']['d64']['repeat']['baseline_over_final']['geomean']:.6f}×。",
        "",
        "Attention本轮仍临时固定SM1200MHz，结束后恢复自动频率；其余四类未主动锁频。"
        "不能将自动频率历史绝对耗时与本轮锁频绝对耗时混用。",
        "",
        "## 本轮改动与发现",
        "",
        "| 算子 | 新方向 | 结果解释与表达方式 |",
        "|---|---|---|",
        "| INT4 | uint8先拼接再FP32反量化；以整数位构造精确0..15浮点值；组合 | "
        "精度通过但整体回退，保留parent。公开Triton join/trans/reshape、cast/bitcast和整数运算可表达 |",
        "| MoE | 同一活动CTA批量处理2/4/8个连续组的stage1 top2，stage2保持parent | "
        "保持原M×8 grid，非组首CTA返回；公开Triton分支、二维张量和axis1 tuple reduce可表达。"
        "工作分配变化不是实际occupancy的测量证据 |",
        "| Norm | 直接最近偶数舍入至i32，整数clamp；联合统计分4组 | "
        "近似持平或回退，保留parent。libdevice float2int_rn有公开Triton CUDA接口；"
        "分组reduce和整数clamp有公开语法 |",
        "| Attention | 在已接受64-key结构上后移dV；或将dV和dK分两遍查询计算 | "
        "两遍计算重复QK、概率及加载，减少部分spill却整体回退；后移dV在新parent上重新检验。"
        "公开表达式顺序与循环可表达 |",
        "| Top-K | merge仅保存每个局部有序列表当前头部和cursor；比较uint32两次归约与packed uint64稳定选择 | "
        "local保持原样，推进胜出列表后masked load下一项；公开Triton可表达，"
        "以更小的merge活跃状态换取额外依赖加载 |",
        "",
        "这些是新的结构实验：Attention第一轮曾在128-key基线上测试dV后移，本轮是在第二轮形成的64-key parent上重新验证；"
        "Top-K利用已有局部排序约束缩小merge状态，不是改变kernel数或用完整排序替代partial Top-K。",
        "",
        "## 资格、负结果与复现",
        "",
        "完整矩阵先做原始源码／文件baseline LLVM、PTX、cubin和完整options核验，再做数值检查。"
        "初筛3轮rep10，冻结候选后主测7轮rep20、独立进程seed123复测5轮rep20，"
        "A/A中位要求[0.97,1.03]。晋升要求两次增量几何均值≥1.02、最差点≥0.95，并通过sanitizer。",
        "",
        "所有新增晋升均有memcheck零错误和racecheck零错误／零警告证据。"
        "Norm／Top-K正常分布使用完整M验证，其他压力分布及sanitizer使用min(M,3)／min(M,2)行；"
        "覆盖30映射，不把缩减行数当作每个完整M均在sanitizer下执行。"
        "Top-K保持稳定id与数值exact，±0等价；Norm沿用code误差≤1及scale／重构误差合同。",
        "",
        "解析失败版本与初筛负结果均保留。Top-K packed初版SSA命名冲突在v2修复，选拔使用完整screen_v2。"
        "Attention fission的解析修正与无收益记录见分报告。仅主/复测确认结果属于固定矩阵A级性能证据，"
        "静态寄存器、spill、shared及.text身份属于C级，不能单独证明动态性能原因。",
        "",
        "**r2–r6新增结构尚未做Triton源码回写实测。** 可表达性不等于已验证源码性能或机器指令相同；"
        "第一轮41点源码回写的结论仅适用于当时版本。",
        "",
        "[预声明计划](round6_plan.md) · [机器可读汇总](round6_summary.json) · [r3–r5历史报告](multiround_report.md) · "
        "[第一轮源码回写](source_port_comparison.md)",
        "",
        "复现汇总：`/root/venvs/ir-ops-dev/bin/python complex_suite/round6_report.py`。"
        "各算子报告和workflow记录包含完整测量命令、原始样本、每shape选择及哈希。",
        "",
    ]
    rows[2:2] = [
        "[全部152个shape的加速比](round6_per_shape.md) · [CSV表格](round6_per_shape.csv)",
        "",
    ]
    (ROOT / "complex_suite/round6_report.md").write_text("\n".join(rows))


if __name__ == "__main__":
    main()
