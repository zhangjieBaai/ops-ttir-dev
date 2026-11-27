"""Combine qualified expanded-matrix round2 evidence without pooling old shapes."""

import json
import math
import statistics
from pathlib import Path

from support import digest, save

ROOT = Path(__file__).resolve().parents[1]


def main():
    ops = ["int4_gemm", "moe_routing", "norm_quant", "attention_dkdv", "partial_topk"]
    data = {
        op: json.loads((ROOT / op / "round2/summary.json").read_text()) for op in ops
    }
    records = []
    mapping = {
        "int4_gemm": ("incremental", "cumulative", "matrix_size"),
        "moe_routing": (
            "parent_over_candidate",
            "baseline_over_candidate",
            "matrix_count",
        ),
        "norm_quant": ("parent_over", "baseline_over", "shape_count"),
        "partial_topk": ("parent_over", "baseline_over", "shape_count"),
    }
    for op in ops:
        d = data[op]
        if op == "attention_dkdv":
            accepted = {g: x for g, x in d["groups"].items() if x["accepted"]}
            assert accepted, "Report no accepted Attention group separately"
            row = {
                "op": op,
                "matrix_points": d["matrix_points"],
                "accepted_points": len(accepted) * 8,
                "groups": d["groups"],
                "accepted_groups": list(accepted),
            }
            for phase in ("main", "repeat"):
                row[phase] = {
                    "incremental": math.exp(
                        statistics.mean(
                            math.log(g[phase]["parent_over_candidate"]["geo"])
                            for g in accepted.values()
                        )
                    ),
                    "cumulative": math.exp(
                        statistics.mean(
                            math.log(g[phase]["baseline_over_candidate"]["geo"])
                            for g in accepted.values()
                        )
                    ),
                }
        else:
            inc, cum, count = mapping[op]
            assert d["accepted"] and d[count] == 30
            row = {
                "op": op,
                "matrix_points": 30,
                "accepted_points": 30,
                "candidate": d.get("candidate", d.get("selected")),
            }
            for phase in ("main", "repeat"):
                row[phase] = {
                    "incremental": d[phase][inc]["geomean"],
                    "cumulative": d[phase][cum]["geomean"],
                }
        row["summary_sha256"] = digest(ROOT / op / "round2/summary.json")
        records.append(row)
    assert sum(x["matrix_points"] for x in records) == 152
    save(
        ROOT / "complex_suite/round2_summary.json",
        {"total_points": 152, "records": records},
    )
    names = {
        "int4_gemm": "INT4反量化＋GEMM",
        "moe_routing": "Grouped/biased MoE",
        "norm_quant": "Norm＋动态INT8",
        "attention_dkdv": "Attention dK/dV",
        "partial_topk": "大N partial Top-K",
    }
    lines = [
        "# 五类算子第二轮TTIR优化：152个性能点",
        "",
        "本轮在上一轮已验收版本上继续直接修改TTIR，并按用户要求把每类矩阵扩充到约30个shape。",
        "INT4、MoE、Norm和Top-K各30点，Attention32点，共152点；所有原测试点保留。",
        "下表区分增量收益（parent/candidate）与累计收益（原baseline/candidate），两者均来自同轮配对测量。",
        "不能将本轮扩展矩阵平均与上一轮小矩阵平均相除，不能把旧收益重新记为本轮增量。",
        "",
        "| 算子 | 正式性能点 | 新版本收益覆盖 | 主测增量 | 独立复测增量 | 主测累计 | 独立复测累计 |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in records:
        scope = (
            "全部30点"
            if row["op"] != "attention_dkdv"
            else f"{row['accepted_points']}点，组：{', '.join(row['accepted_groups'])}"
        )
        lines.append(
            f"| [{names[row['op']]}](../{row['op']}/round2/report.md) | {row['matrix_points']} | {scope} | "
            f"{row['main']['incremental']:.4f}× | {row['repeat']['incremental']:.4f}× | "
            f"{row['main']['cumulative']:.4f}× | {row['repeat']['cumulative']:.4f}× |"
        )
    lines += [
        "",
        "表中性能为各自冻结矩阵或明确组子集的A级证据。Attention未接受组保留parent，",
        "其候选负结果仍列在分报告中；不把子集收益冒充全32点收益，也不把回退项伪装成实测1×。",
        "",
        "## 本轮优化内容",
        "",
        "| 算子 | 新结构 | 关键取舍与负结果 |",
        "|---|---|---|",
        "| INT4 | scale/zero由宽张量加载改为窄加载后广播；两次64-K迭代复用同组参数；保留原dot顺序 | 主要收益不等于TTIR load个数减少；单独删K mask在部分点回退 |",
        "| MoE | 固定8组的稳定排名、uint32选择状态、选择后一次gather复用原始权重 | 保留原tanh sigmoid、未加bias权重和稳定tie；仍为两个kernel |",
        "| Norm＋INT8 | 每行一次precise reciprocal＋逐元素乘法；Z提前写出 | 量化舍入路径改变，但沿用原±1 code/scale/重构合同；fold256归约明显回退 |",
        "| Attention | 同一CTA内64-key子块；非因果组重算指针，因果组展开两次子块 | D128获益；32-key虽去掉spill却损失WGMMA，部分候选变慢 |",
        "| Top-K | 固定原始ids，哨兵仅更新score key；展开K次选择 | 尾块耗尽显式INTMAX；packed64联合key减少reduce却更慢 |",
        "",
        "所有方案保持原设备ABI、grid、kernel数和工作区约定，只改TTIR内容；没有改TTGIR/LLVM/PTX/cubin或用内联PTX。",
        "Attention改变的是同一CTA内部计算子块，INT4改变的是同一kernel内部循环，并非改launch配置或省略阶段。",
        "",
        "这些结构原则上也能通过公开Triton的窄load/broadcast、循环、tl.gather/tl.reduce、tl.div_rn及tl.static_range表达。",
        "**本轮新方案没有再次做源码回写计时**；不能把上一轮源码与TTIR的机器码一致性结论直接继承给本轮。",
        "",
        "## 扩展矩阵与配置",
        "",
        "| 算子 | 覆盖 | 新点配置策略 |",
        "|---|---|---|",
        "| INT4 | M1..256，N64..11008，K128..8192及合法M/N尾部 | 原7点既有配置；新增23点使用明确固定tile族，无重新autotune |",
        "| MoE | 15种tokens（16..8192）× E256/512 | G8/top4/top8、FP32，两个kernel均4warps/3stages |",
        "| Norm | N257/1024/4096/4097/8192/16384，rows1..2048，保留原8点 | 原width-family：N4097为4warps，其余8warps，3stages |",
        "| Attention | 8种B/H/N × D64/128 × causal/noncausal | 原4warps/5stages，N为128倍数；完整组固定8点 |",
        "| Top-K | 5种rows × 6种N/K，最大N1048576，含65537尾块 | 原chunk1024、4warps/3stages，两阶段合并 |",
        "",
        "固定配置不是全面调优后的最强基线。新30/32点的parent来自原已验收变换的同合同扩展，",
        "新增点重新检查baseline/parent正确性和编译链，不把上一轮性能证据外推到它们。",
        "",
        "## 验证与归因",
        "",
        "官方Triton3.7.0/H100；每个算子独占GPU。所有候选先完整原数值/压力合同，再初筛，",
        "四类30点实验未主动锁频；Attention因同一parent在未锁频时A/A不稳定，",
        "采用GPU4固定SM1200MHz的新协议，所有路线和完整32点在同频下重测，结束后恢复自动频率。",
        "Attention锁频结果不与旧未锁频样本混合，也不拿其他算子的绝对时间作跨算子比较。",
        "冻结候选后主测7轮20ms、独立seed复测5轮20ms。全部输入seed/哈希、IR/产物哈希、完整options、",
        "资源、随机执行顺序和原始配对样本均保留。指标为预分配CUDA graph设备序列，不含编译、分配或Python启动。",
        "10ms/20ms指do_bench_cudagraph的rep参数；该版本内部10次replay返回均值，保存的是全部外层配对样本。",
        "原源码到baseline TTIR文件的LLVM/PTX/cubin一致性逐点核查；A/A门槛为[0.97,1.03]，无效轮不进入汇总。",
        "最终候选均有memcheck/racecheck，0 errors、racecheck 0 warnings。各工具实际测试shape和压力rows以分报告/JSON为准。",
        "",
        "Norm另有90项CPU FP64极端输入检查。Top-K首次main/repeat被主控误并发，两个完整轮次均排除，",
        "只使用隔离后的main_isolated/repeat_isolated；Attention筛选中的A/A失败与稳定化重测也保留。",
        "没有因候选失败放宽精度阈值。INT4/MoE/Attention维持原逐位对照，Norm维持原量化容差，Top-K维持稳定索引和数值精确（±0等价）。",
        "",
        "寄存器、spill、shared和PTX计数是C级静态事实，不能单项解释全部加速；未用动态NCU证据的地方不声称DRAM字节或stall因果已证明。",
        "已保存失败候选、显著回退和配置局限，不计算跨算子的总平均，也不声称TTIR独占能力。",
        "",
        "[共同协议](round2_plan.md) · [机器可读汇总](round2_summary.json) · [Norm/Top-K静态产物](round2_static_analysis.json)",
    ]
    (ROOT / "complex_suite/round2_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
