"""Render conditional forward ablation effects and source-expression parity."""

import csv
import json
import math
import statistics

from common import ROOT, save_json

R = ROOT / "results/ablation"
ORDER = [
    "abl_recip_only",
    "abl_rsqrt_only",
    "abl_arithmetic",
    "abl_grouped",
    "abl_unroll",
    "abl_ca",
    "abl_cs",
    "abl_final",
    "abl_no_ca",
    "abl_no_cs",
    "abl_no_unroll",
    "abl_source_formatted",
]
LABEL = dict(
    zip(
        ORDER,
        [
            "只复用1/N",
            "只改rsqrt",
            "两项算术合并",
            "2048动态分组+算术",
            "范围静态展开",
            "再加均值阶段.ca",
            "再加输出.cs",
            "最终：再加小N1024分组",
            "最终去掉.ca",
            "最终去掉.cs",
            "去静态展开及小N混合",
            "公开Triton源码版",
        ],
    )
)


def geo(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


def compare(data, a, b):
    """Duration(a) / duration(b); each kernel independently paired with baseline."""
    ar = {r["N"]: r for r in data["records"] if r["kernel"] == a}
    br = {r["N"]: r for r in data["records"] if r["kernel"] == b}
    assert ar.keys() == br.keys() and len(ar) == 30
    ratios = [ar[n]["candidate_us"] / br[n]["candidate_us"] for n in ar]
    return dict(
        geometric_mean=geo(ratios),
        worst=min(ratios),
        best=max(ratios),
        wins=sum(v > 1 for v in ratios),
    )


def main():
    datasets = [json.load(open(R / p)) for p in ["full.json", "repeat.json"]]
    for d in datasets:
        assert d["aa_passed"] and len(d["records"]) == 30 * len(ORDER)
    resources = json.load(open(R / "resources.json"))
    rows = [
        "# LayerNorm 前向 TTIR 消融实验",
        "",
        "本报告研究已有前向最终版r2_mixed_stream，不是backward dX。原baseline、optimized及冻结教程均不修改。"
        "官方Triton3.7.0、H100、FP16、M4096、N1024..15872步长512、固定8 warps/3 stages/1 CTA。",
        "",
        "## 各版本相对原始baseline",
        "",
        "| 版本 | 主测几何加速 | 独立复测几何加速 | 寄存器 | spill |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ORDER:
        a, b = [d["summary"][name]["geometric_mean"] for d in datasets]
        r = resources[name]
        rows.append(
            f"| {LABEL[name]} | {a:.4f}× | {b:.4f}× | {r['registers']} | {r['spills']} |"
        )
    rows += [
        "",
        "主测5轮、独立进程seed123复测3轮，每轮rep10ms；所有版本100ms无计时稳定阶段。"
        "每个shape/每轮都有A/A控制，两次完整矩阵均通过0.97..1.03门槛。"
        "这里只计预分配CUDA graph kernel时间，不含分配、编译、CPU launch，也不代表冷cache或端到端性能。",
        "",
        "## 逐项加入的条件收益",
        "",
        "收益定义为前一版本耗时/后一版本耗时的几何平均。各版本都与共同baseline交错测量，"
        "下列差分使用同会话每shape耗时中位数，未声称是直接相邻版本配对；百分比表示吞吐提升，不是耗时减少。",
        "",
        "| 对照 | 改动 | 主测加速 / 吞吐提升 | 复测加速 / 吞吐提升 |",
        "|---|---|---:|---:|",
    ]
    transitions = [
        ("abl_arithmetic", "abl_grouped", "窄化为2048并重构三遍动态循环"),
        ("abl_grouped", "abl_unroll", "按N范围选择组数并静态展开"),
        ("abl_unroll", "abl_ca", "仅均值阶段使用.ca，改变跨阶段load合并"),
        ("abl_ca", "abl_cs", "输出使用.cs"),
        ("abl_cs", "abl_final", "N≤3072时改为1024分组"),
    ]
    effects = {}
    for a, b, label in transitions:
        values = [compare(d, a, b) for d in datasets]
        effects[label] = values
        cells = [
            f"{v['geometric_mean']:.4f}× / {(v['geometric_mean']-1)*100:+.2f}%"
            for v in values
        ]
        rows.append(f"| {a} → {b} | {label} | {cells[0]} | {cells[1]} |")
    rows += [
        "",
        "1/N复用和rsqrt分别以baseline为参照，见第一张表；合并后不是两个百分比相加。"
        "“窄分组”同时改变累加/循环的表达，不能进一步把其收益全部归因于寄存器下降。",
        "",
        "## 从最终版移除改动",
        "",
        "数值为“缺少该改动版本的耗时 / 最终版耗时”，大于1说明最终版更快。"
        "这里是给定其余变换后的条件贡献，不是可相加的独立贡献。",
        "",
        "| 去掉的变化 | 主测条件加速 | 复测条件加速 |",
        "|---|---:|---:|",
    ]
    for name, label in [
        ("abl_no_ca", ".ca"),
        ("abl_no_cs", ".cs"),
        ("abl_cs", "小N混合分组"),
        ("abl_no_unroll", "静态范围展开和小N混合（联合移除）"),
    ]:
        values = [compare(d, name, "abl_final") for d in datasets]
        effects["remove " + label] = values
        rows.append(
            f"| {label} | {values[0]['geometric_mean']:.4f}× | {values[1]['geometric_mean']:.4f}× |"
        )
    rows += [
        "",
        "最后一行同时回到统一2048动态循环，不能解释成“只移除展开”的纯独立贡献。"
        "展开本身的更干净对照为上一节abl_grouped→abl_unroll。",
        "",
        "这些是当前顺序下的条件效应，存在交互，不能相加为总收益。"
        "低于约1%且跨轮方向不稳定的差异不据此宣称可靠优化；本实验未给置信区间。"
        "没有把历史搜索中未进入最终版的延迟统计量store、volatile或输出affine重写计入最终改动。",
        "",
        "资源线索：动态窄分组为32 registers，静态展开为80，加入.ca后为48。"
        "这与加载复用/存活范围改变一致，但并未单独证明全部收益由寄存器导致。"
        "resources.json的静态load/barrier计数包含所有分支、循环体只计一次，不代表运行时指令总量。",
        "这里.ca既是缓存提示，也在该程序和编译器中改变了load合并及存活范围。"
        "因此该消融测到的是整个编译结果变化，不是单纯的L1缓存命中率贡献；"
        "公开Triton能写出.ca也不意味着它在任意程序中都能强制保留重载。",
        "",
        "## 能否用普通Triton实现",
        "",
        "| 改动 | 公开Triton表达 | 结论 |",
        "|---|---|---|",
        "| 倒数复用 | inv=1./N，mean=sum*inv，var=sum*inv | 能 |",
        "| sqrt+除法改rsqrt | tl.rsqrt(var+eps) | 能 |",
        "| 窄分组/动态循环 | tl.arange(0,C)、range/tl.range、向量累加 | 能 |",
        "| 按N范围展开 | 运行时if/elif、constexpr组数、tl.static_range | 能 |",
        '| 均值阶段.ca | tl.load(...,cache_modifier=".ca") | 能；不保证禁止任意CSE |',
        '| 输出.cs | tl.store(...,cache_modifier=".cs") | 能 |',
        "| 小N混合1024/大N2048 | 分支调用constexpr C不同的@triton.jit helper | 能 |",
        "",
        "可执行实现：[ablation_triton.py](ablation_triton.py)。源码版覆盖这些最终变化，"
        "不是仅凭API名称判断可行；已完成完整数值矩阵、边界检查与性能测试。"
        "源码导出TTIR后再走文件入口，与该源码路径LLVM/PTX/cubin逐字节一致。"
        "源码版与原手写TTIR的机器码.text哈希不同，因此可表达同一算法不等于同一机器码。",
        "",
    ]
    for label, d in zip(["主测", "复测"], datasets):
        v = compare(d, "abl_source_formatted", "abl_final")
        rows.append(
            f"{label}：源码版耗时 / 手写TTIR最终版耗时几何比={v['geometric_mean']:.4f}；"
            "大于1表示手写TTIR更快，小于1表示源码版更快。"
        )
    rows += [
        "",
        "本例最终优化没有必须依赖普通Triton无法表达的运算。TTIR的优势在精确控制编译截点和实验输入；"
        "真实机器码、CSE、寄存器与调度效果仍由下游决定。",
        "",
        "## 全部shape",
        "",
        "每列为独立复测相对baseline的配对加速比中位数。完整两轮时延、样本和IR/cubin哈希保存在JSON；CSV含两轮各版本。",
        "",
    ]
    short = [
        "abl_arithmetic",
        "abl_grouped",
        "abl_unroll",
        "abl_ca",
        "abl_cs",
        "abl_final",
        "abl_source_formatted",
    ]
    rows += [
        "| N（M=4096） | " + " | ".join(short) + " |",
        "|---:|" + "---:|" * len(short),
    ]
    table = {(r["N"], r["kernel"]): r for r in datasets[1]["records"]}
    for n in range(1024, 15873, 512):
        rows.append(
            f"| {n} | "
            + " | ".join(f"{table[n, k]['speedup']:.4f}×" for k in short)
            + " |"
        )
    rows += [
        "",
        "证据：[主测](results/ablation/full.json)、[独立复测](results/ablation/repeat.json)、"
        "[资源](results/ablation/resources.json)、[数值验证](results/ablation/accuracy_combined.json)、"
        "[源码路径一致性](results/ablation/source_formatted_parity.json)、[全部CSV](results/ablation/all_shapes.csv)。",
        "",
        "复现入口：make_ablation.py生成消融IR；ablation_triton.py生成并核查公开源码对照；"
        "validate.py进行数值/边界检查；ablation_benchmark.py按同协议测量；report_ablation.py生成本报告。"
        "用新的--output路径保存重复测试。初次格式化前的源码及导出物单独保存在source_initial.py.txt、"
        "source_parity.json和abl_source_final.ttir，仅为溯源；正式性能表使用abl_source_formatted。",
    ]
    (ROOT / "report_ablation.md").write_text("\n".join(rows) + "\n")
    save_json(R / "effects.json", effects)
    with (R / "all_shapes.csv").open("w") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "kernel",
                "M",
                "N",
                "baseline_us",
                "candidate_us",
                "speedup",
            ],
        )
        writer.writeheader()
        for run, d in zip(["main", "repeat"], datasets):
            for v in d["records"]:
                writer.writerow(
                    dict(
                        run=run,
                        **{
                            k: v[k]
                            for k in [
                                "kernel",
                                "M",
                                "N",
                                "baseline_us",
                                "candidate_us",
                                "speedup",
                            ]
                        },
                    )
                )


if __name__ == "__main__":
    main()
