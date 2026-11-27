"""Render the ablation report from paired factorial evidence and syntax witnesses."""

import csv
import json
import math
import statistics

from ablation_common import EDGES, NAMES
from common import ALL_SHAPES, ROOT, save_json, sha256

RUN = ROOT / "results/runs/ablation_v1"


def read(name):
    return json.loads((RUN / f"{name}.json").read_text())


def geometric(values):
    return math.exp(statistics.mean(map(math.log, values)))


def ratio_from_blocks(run, before, after, vector=False):
    points = [p for p in run["points"] if not vector or p["shape"][2] >= 1024]
    return geometric(
        [
            statistics.median(
                block["times_us"][before] / block["times_us"][after]
                for block in point["blocks"]
            )
            for point in points
        ]
    )


def main():
    analysis = read("analysis")
    timing = {name: read(name) for name in ("main", "repeat")}
    syntax = read("public_syntax")
    probes = read("branch_probes")
    pipeline = read("pipeline")
    checks = dict(
        analysis=analysis["passed"],
        syntax=syntax["passed"],
        branch_probes=probes["passed"],
    )
    for tool, marker in [
        ("memcheck", "ERROR SUMMARY: 0 errors"),
        ("racecheck", "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"),
    ]:
        data = read(tool)
        checks[tool] = (
            data["passed"]
            and len(data["records"]) == 42 * 2 * 13
            and marker in (RUN / f"{tool}.log").read_text()
        )
    checks["syntax_scope"] = (
        len(syntax["records"]) == sum(s[2] >= 1024 for s in ALL_SHAPES) * 8 * 13
    )
    checks["probe_scope"] = len(probes["records"]) == 8 * 8 * 13 * 2
    checks["probe_mapping"] = len(probes["mappings"]) == 8 and all(
        p["source_file_byte_equal"] for p in probes["mappings"]
    )
    checks["same_gpu"] = (
        len(
            {
                data["physical_uuid"]
                for data in [pipeline, syntax, probes, *timing.values()]
            }
        )
        == 1
    )
    checks["syntax_source"] = syntax["source_sha256"] == sha256(
        ROOT / "triton_syntax_ablation.py"
    )
    checks["plan"] = (
        syntax["plan_sha256"]
        == probes["plan_sha256"]
        == sha256(ROOT / "ablation_plan.md")
    )
    checks["supplement"] = probes["supplement_sha256"] == sha256(
        ROOT / "ablation_branch_probe_plan.md"
    )
    for group, versions in pipeline["kernels"].items():
        for name in NAMES:
            for tool in ("memcheck", "racecheck"):
                info = read(tool)["kernels"][group][name]
                assert info["input_sha256"] == versions[name]["input_sha256"]
                assert info["stages"]["cubin"] == versions[name]["stages"]["cubin"]
    assert all(checks.values()), checks
    effects = analysis["effects"]
    a = effects["main"]["all30"]
    b = effects["repeat"]["all30"]
    av = effects["main"]["vector18"]
    bv = effects["repeat"]["vector18"]
    lines = [
        "# mean_dim 两轮改动消融实验",
        "",
        "**主要收益来自32位索引和16次循环展开。**",
        "对齐K上，单独扁平化没有改变机器码；短N分支和尾列归约的收益依赖组合。",
        "五类改动都已用官方Triton3.7.0公开语法编译并运行验证。",
        "这证明算法可表达，不证明源码入口会生成与手写TTIR相同的机器码或性能。",
        "",
        "## 先说明如何读表",
        "",
        "原始基线指最初从FlagGems导出的baseline.ttir；第一轮版本指optimized.ttir。",
        "所有加速比均为“改动前耗时÷改动后耗时”，大于1表示更快。",
        "每点先取同轮配对比值中位数，再求测试点几何平均。",
        "主测7轮，独立进程不同种子复测5轮；每次CUDA graph计时20ms。",
        "下文区间是固定测试集配对轮次的95% bootstrap区间，不代表其他输入。",
        "",
        "F表示把128×8的逻辑K状态扁平化为1024；I表示地址索引由64位改为32位。",
        "U表示每轮预加载16行后按顺序累加；G表示N<8走第一轮路径。",
        "T表示仅对剩余一列的CTA采用256×1的并行N归约。CTA就是一个GPU线程块。",
        "",
        "第一轮采用F×I四组合，第二轮采用U×G×T八组合。",
        "所有组合都测完整30点；向量路径18点单列，避免未改动的12点稀释收益。",
        "没有挑选候选、删除退步点或修改此前最终文件。",
        "",
        "## 第一轮：扁平化和索引宽度",
        "",
        "| 对比 | 30点主测/复测 | 向量18点主测/复测 | 30点主测95%区间 |",
        "|---|---:|---:|---|",
    ]

    def row(label, description):
        ci = a[label]["ci95"]
        return (
            f"| {description} | {a[label]['geomean']:.4f}× / {b[label]['geomean']:.4f}× | "
            f"{av[label]['geomean']:.4f}× / {bv[label]['geomean']:.4f}× | [{ci[0]:.4f}, {ci[1]:.4f}] |"
        )

    for label, description in [
        ("I_at_F0", "只改32位索引，保留128×8"),
        ("I_at_F1", "已扁平化后，再改32位索引"),
        ("F_at_I0", "只扁平化，保留64位索引"),
        ("F_at_I1", "已改32位索引后，再扁平化"),
        ("FI_combined", "扁平化与32位索引同时应用"),
        ("round1_bridge", "上述两项之后，换成第一轮循环/掩码表达"),
        ("round1_total", "原始基线→第一轮最终版"),
    ]:
        lines.append(row(label, description))
    lines += [
        "",
        "这里没有把第一轮的全部收益直接分给F和I。",
        "额外对照单独测了第一轮循环、掩码及单行归约表达的重建影响。",
        "这些表达仍保留同样的求和顺序与归约后除法。",
        "",
        "| 向量配置组 | 只扁平化=原始cubin | 扁平化32位=未扁平化32位 | 仅索引32位=第一轮最终 |",
        "|---|---|---|---|",
    ]
    for info in analysis["resources_and_binary"]:
        lines.append(
            f"| {info['group']} | {info['flat64_equals_baseline']} | "
            f"{info['flat32_equals_narrow32']} | {info['narrow32_equals_round1']} |"
        )
    lines += [
        "",
        "上表比较整个cubin文件的SHA256，不是只看指令数。",
        "五个对齐组中，扁平化前后的cubin相同，索引32位版本与第一轮最终版也相同。",
        "两个非对齐组不同，不能将这条结论推广给奇数K。",
        "",
        "## 第二轮：展开、短N分支、单列尾块",
        "",
        "| 单项及其上下文 | 30点主测/复测 | 向量18点主测/复测 | 30点主测95%区间 |",
        "|---|---:|---:|---|",
    ]
    for label, description in [
        ("round2_scaffold", "仅改为第二轮单步循环表达"),
        ("U_at_u0g0t0", "单独加U：16次顺序展开"),
        ("G_at_u0g0t0", "单独加G：短N分支"),
        ("T_at_u0g0t0", "单独加T：单列尾块归约"),
        ("U_at_u0g1t1", "完整组合中增加U，G/T固定"),
        ("G_at_u1g0t1", "完整组合中增加G，U/T固定"),
        ("T_at_u1g1t0", "完整组合中增加T，U/G固定"),
        ("round2_total", "第一轮→第二轮完整组合"),
        ("all_total", "原始基线→第二轮完整组合"),
    ]:
        lines.append(row(label, description))
    lines += [
        "",
        "“单独增加”与“完整组合中增加”是两个不同问题，不能把百分比相加。",
        "完整组合中增加某因素，等价于比较“移除该因素”的版本与完整版本。",
        "完整22组对比和每组区间见results/runs/ablation_v1/analysis.json。",
        "",
        "| U | G | T | 相对第一轮：30点主测/复测 | 相对第一轮：向量18点主测/复测 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for name in [name for name in NAMES if name.startswith("ab_u")]:
        bits = [int(name[4]), int(name[6]), int(name[8])]
        values = [
            ratio_from_blocks(timing[run], "optimized", name, vector)
            for vector in (False, True)
            for run in ("main", "repeat")
        ]
        lines.append(
            f"| {bits[0]} | {bits[1]} | {bits[2]} | {values[0]:.4f}× / {values[1]:.4f}× | "
            f"{values[2]:.4f}× / {values[3]:.4f}× |"
        )
    lines += [
        "",
        "## K=1025退步修复到底来自哪里",
        "",
        "只有一个性能点使用奇数K，即(1,63,1025)。",
        "对该点的受控比较如下，不能把它外推成所有非对齐尺寸的结论。",
        "",
        "| 改动及上下文 | 主测 | 复测 |",
        "|---|---:|---:|",
    ]
    for label, description in [
        ("I_at_F0", "原128×8上只改32位索引"),
        ("F_at_I0", "保留64位索引，只扁平化"),
        ("F_at_I1", "32位索引上再扁平化"),
        ("round1_total", "原始→第一轮"),
        ("U_at_u0g0t0", "单步循环上只加16次展开"),
        ("T_at_u1g1t0", "已有U/G时再加尾列归约"),
        ("all_total", "原始→第二轮完整组合"),
    ]:
        am = effects["main"]["oddK"][label]["geomean"]
        bm = effects["repeat"]["oddK"][label]["geomean"]
        lines.append(f"| {description} | {am:.4f}× | {bm:.4f}× |")
    lines += [
        "",
        "因此，第二轮该点从退步变加速，不能直接归功于尾列并行归约。",
        "应根据上面的展开单项和尾列条件对比判断；此前报告只验证了整个组合。",
        "",
        "原30点中，向量路径N要么为单独特化的1，要么至少为8。",
        "所以G在这些点没有实际执行N=2..7的回退路径。",
        "这里测到的G影响主要属于分支存在引发的代码生成变化，不能称为命中短N快路径的收益。",
        "补充8个正确性点：(M=3,N∈{2,7,8,9},K∈{1024,1025})。",
        "先逐点验证源码导出与已有配置组一致，再建立独立准确性映射，不加入性能平均。",
        "",
        "## 每种改动在Triton中能否实现",
        "",
        "以下是实测编译与数值验证，不只依据API名称推断。",
        "源码见triton_syntax_ablation.py，运行证据见public_syntax.json和branch_probes.json。",
        "",
        "| 改动 | 公开Triton表达 | 实测结论 |",
        "|---|---|---|",
        "| F：扁平化 | tl.arange(0,1024)，替代两个arange的笛卡尔组合 | 可以 |",
        "| I：索引窄化 | program_id.to(tl.int32)，标量/索引用tl.cast显式选位宽 | 可以，必须证明范围不溢出 |",
        "| U：16次展开 | 外层range步长16，tl.static_range收集load，再顺序累加 | 可以 |",
        "| G：短N分支 | 运行时if N<8选择原单步循环 | 可以 |",
        "| T：单列尾块 | 运行时尾列判断，tl.arange(0,256)+tl.sum | 可以 |",
        "",
        "这些源码使用常规@triton.jit和公开triton.language接口。",
        "没有使用Gluon、FlagTree、内联汇编、私有IR builder或编译器补丁。",
        "源码入口会经过前端规范化；TTIR文件入口从make_ttgir继续。",
        "因此“能表达相同算法”不等于“能指定相同SSA图、最终布局或机器码”。",
        "本实验没有测量源码见证版的性能，不声称它与TTIR版同速。",
        "",
        "## 验证与范围",
        "",
        f"主测和复测全部30点A/A控制通过，同一GPU：{timing['main']['physical_uuid']}。",
        "每轮随机测13种版本和一个重复原始基线，保存所有样本、顺序、哈希和遥测。",
        "比较的两种版本来自同一轮，不使用两轮历史结果相除。",
        "未锁定时钟、未清缓存；指标为预分配输出下的CUDA graph设备kernel耗时。",
        "",
        "两种数据种子分别4368项准确性检查通过。",
        "42点源代码/TTIR文件入口核查通过，原始基线的LLVM/PTX/cubin字节一致。",
        "所有版本的真实编译选项、寄存器和shared memory记录在pipeline.json。",
        "memcheck/racecheck各1092项通过，分别0 errors与0 hazards。",
        f"公开Triton源码见证通过{len(syntax['records'])}项检查；分支补充{len(probes['records'])}项。",
        "数值参考、容差、NaN/Inf和两侧输出保护区检查沿用既定契约。",
        "",
        "性能结论等级A仅覆盖当前连续FP16矩阵，全部输入此前已暴露，未进行新一轮调优。",
        "语法表达能力已编译和执行验证，但没有证明TTIR与源码的性能等价。",
        "没有profiler或寄存器/调度单项控制，不能把时间差进一步断言为某种stall的减少。",
        "前12点未改变实现；近1×的差异是计时波动。退步和条件效应全部保留。",
        "",
        "## 全30点与复现",
        "",
        "完整22条配对边的每点主测/复测数据见ablation_per_shape.csv。",
        "下面列出关键的“移除某因素”对照。",
        "",
        "| M,N,K | 原始→全组合 主/复 | 单独索引32位 主/复 | 全组合中加U 主/复 | 全组合中加G 主/复 | 全组合中加T 主/复 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = ["all_total", "I_at_F0", "U_at_u0g1t1", "G_at_u1g0t1", "T_at_u1g1t0"]
    for item in analysis["per_shape"]:
        cells = [",".join(map(str, item["shape"]))]
        cells.extend(
            f"{item['main'][label]:.4f} / {item['repeat'][label]:.4f}"
            for label in labels
        )
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "从仓库根目录执行（先选择空闲GPU）：",
        "",
        "```bash",
        "CUDA_VISIBLE_DEVICES=5 bash mean/test_ablation.sh",
        "```",
        "",
        "所有阶段写入新目录。已有TTIR及设计文件不会覆盖。",
        "消融没有替换第一轮或第二轮最终候选。",
    ]
    report = ROOT / "report_ablation.md"
    assert not report.exists()
    report.write_text("\n".join(lines) + "\n")
    csv_path = ROOT / "ablation_per_shape.csv"
    with csv_path.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "M",
                "N",
                "K",
                *[
                    f"{label}_{run}"
                    for label, _, _ in EDGES
                    for run in ("main", "repeat")
                ],
            ]
        )
        for item in analysis["per_shape"]:
            writer.writerow(
                [
                    *item["shape"],
                    *[
                        item[run][label]
                        for label, _, _ in EDGES
                        for run in ("main", "repeat")
                    ],
                ]
            )
    save_json(
        ROOT / "results/ablation_acceptance.json",
        dict(
            accepted=all(checks.values()),
            checks=checks,
            plan_sha256=sha256(ROOT / "ablation_plan.md"),
            report_sha256=sha256(report),
            csv_sha256=sha256(csv_path),
            evidence={
                str(path.relative_to(ROOT)): sha256(path) for path in RUN.glob("*.json")
            },
            key_effects={
                scope: {
                    label: {
                        run: effects[run][scope][label] for run in ("main", "repeat")
                    }
                    for label in labels
                }
                for scope in ("all30", "vector18", "oddK")
            },
        ),
    )
    print("PASS ablation report and acceptance", checks)


if __name__ == "__main__":
    main()
