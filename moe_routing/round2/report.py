"""Qualify one frozen finalist and report incremental and cumulative evidence."""

import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from complex_suite.support import digest, freeze, save  # noqa: E402


def bootstrap(data, name, factor):
    rng = random.Random(731)
    values = []
    for _ in range(3000):
        ratios = []
        for row in data["records"]:
            samples = row["timing"]["samples"]
            ratios.append(
                statistics.median(rng.choice(samples)[factor][name] for _ in samples)
            )
        values.append(math.exp(statistics.mean(math.log(x) for x in ratios)))
    values.sort()
    return [values[75], values[2925]]


def main():
    selection = json.loads((ROOT / "selection_pre_main.json").read_text())
    name = selection["candidate"]
    runs = {
        run: json.loads((ROOT / f"results/{run}/performance.json").read_text())
        for run in ("main", "repeat")
    }
    for run, data in runs.items():
        assert len(data["records"]) == 30 and data["aa_passed"]
        accuracy = json.loads((ROOT / f"results/{run}/accuracy.json").read_text())
        assert not accuracy["failed"] and len(accuracy["checks"]) == 540
        assert (
            len(json.loads((ROOT / f"results/{run}/pipeline.json").read_text())) == 30
        )
        for e in (256, 512):
            for variant in ("baseline", "parent", name):
                for stage, meta in enumerate(data["kernels"][f"e{e}/{variant}"], 1):
                    path = Path(meta["ir_path"])
                    assert meta["ir_sha256"] == digest(path)
    for tool in ("memcheck", "racecheck"):
        acc = json.loads((ROOT / f"results/{tool}/accuracy.json").read_text())
        assert not acc["failed"] and len(acc["checks"]) == 540
        log = (ROOT / f"results/{tool}.log").read_text()
        marker = (
            "ERROR SUMMARY: 0 errors" if tool == "memcheck" else "0 errors, 0 warnings"
        )
        assert marker in log
    accepted = all(
        d["summary"][name]["parent_over_candidate"]["geomean"] >= 1.02
        and d["summary"][name]["parent_over_candidate"]["worst"] >= 0.95
        for d in runs.values()
    )
    chosen = name if accepted else "parent"
    for e in (256, 512):
        folder = ROOT / "ir" / f"e{e}_w4"
        freeze(folder / "optimized_s2.ttir", (folder / f"{chosen}_s2.ttir").read_text())
    summary = {
        "operator": "moe_routing",
        "matrix_count": 30,
        "parent": "membership",
        "candidate_tested": name,
        "selected": chosen,
        "accepted": accepted,
        "evidence_level": "A, expanded frozen30 only; raw8 historical matrix not pooled",
        "main": runs["main"]["summary"][name],
        "repeat": runs["repeat"]["summary"][name],
        "bootstrap95": {
            run: {
                factor: bootstrap(data, name, factor)
                for factor in ("parent_over_candidate", "baseline_over_candidate")
            }
            for run, data in runs.items()
        },
        "aa_ranges": {
            run: [
                min(r["timing"]["aa_median"] for r in data["records"]),
                max(r["timing"]["aa_median"] for r in data["records"]),
            ]
            for run, data in runs.items()
        },
        "manifest_sha256": digest(ROOT / "manifest30.json"),
        "selection_pre_main_sha256": digest(ROOT / "selection_pre_main.json"),
        "runs_sha256": {
            run: digest(ROOT / f"results/{run}/performance.json") for run in runs
        },
        "sanitizer": {"memcheck": "0 errors", "racecheck": "0 errors, 0 warnings"},
        "accuracy_checks": {
            "screen": 1440,
            "main": 540,
            "repeat": 540,
            "memcheck": 540,
            "racecheck": 540,
        },
        "points": [],
    }
    for row, repeat in zip(runs["main"]["records"], runs["repeat"]["records"]):
        assert row["shape"] == repeat["shape"]
        summary["points"].append(
            {"shape": row["shape"], "main": row["timing"], "repeat": repeat["timing"]}
        )
    save(ROOT / "summary.json", summary)
    save(
        ROOT / "selection.json",
        {
            "selected": chosen,
            "accepted": accepted,
            "parent": "membership",
            "groups": {
                str(e): digest(ROOT / "ir" / f"e{e}_w4" / "optimized_s2.ttir")
                for e in (256, 512)
            },
        },
    )
    m, r = summary["main"], summary["repeat"]
    lines = [
        "# Grouped/biased MoE：第二轮 TTIR 优化与30点扩展验证",
        "",
        f"最终{'接受' if accepted else '不晋升'} `{name}`。"
        f"相对已验收父版本membership，主测增量几何 **{m['parent_over_candidate']['geomean']:.4f}×**，"
        f"",
        f"独立复测 **{r['parent_over_candidate']['geomean']:.4f}×**；"
        f"相对原baseline累计为 **{m['baseline_over_candidate']['geomean']:.4f}× "
        f"/ {r['baseline_over_candidate']['geomean']:.4f}×**。"
        f"",
        f"增量最差点主/复测 **{m['parent_over_candidate']['worst']:.4f}× "
        f"/ {r['parent_over_candidate']['worst']:.4f}×**。"
        f"这不是重复报告上一轮membership收益。",
        "",
        "## 新矩阵与冻结边界",
        "",
        "M={16,32,64,128,256,384,512,768,1024,1536,2048,3072,4096,6144,8192} × E={256,512}，共30点。",
        "用户在候选生成后、任何GPU验证/计时前提出扩展；映射先于本轮计时冻结于[manifest30.json](manifest30.json)。",
        "原8点不与本轮样本混算。所有M仍为正16倍数，满足原运行时参数属性；G8/top4组/top8专家、FP32、连续等长分组、",
        "sigmoid的原tanh表达、bias与原权重归一化合同不变。新M固定沿用4warps/3stages，不声称已逐新点autotune。",
        "原文件和source_port不变；本目录只复制旧baseline/parent并直接改写TTIR，没有从优化Python导出候选。",
        "",
        "## 最终组合改了什么",
        "",
        "1. group选择由4轮threshold选择及scan改为8×8稳定排名：计数比当前组分数更高、或同分但原组index更小的组，rank<4即选中。",
        "2. expert biased分数编码为有序uint32，循环持有该key并使用unsigned max；保留最小原始expert index的tie规则。负分数受支持，±0编码前归一化。",
        "3. expert选择循环只保存indices，不再逐个重载logit、再算一次tanh以及保存weight状态；循环结束后一次tt.gather提取已有原始sigmoid向量，按原顺序归约归一化。",
        "",
        "第三项没有从biased分数减回bias，也没有改tanh或归一化的舍入路径。最终indices、weights和group workspace均要求与原baseline严格相等。",
        "两个kernel、stage1计算、ABI、grid、workspace和完整backend options都不变；被测主机函数只launch。",
        "",
        "## 逐点主测与独立复测",
        "",
        "| (M,E) | baseline μs | parent μs | candidate μs | 主测增量 | 复测增量 | 主测累计 | 复测累计 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point in summary["points"]:
        a, b = point["main"], point["repeat"]
        us = a["median_us"]
        lines.append(
            f"| {tuple(point['shape'])} | {us['baseline']:.3f} "
            f"| {us['parent']:.3f} | {us[name]:.3f} |"
            f" {a['parent_over_candidate'][name]:.4f}× "
            f"| {b['parent_over_candidate'][name]:.4f}× "
            f"| {a['baseline_over_candidate'][name]:.4f}× "
            f"| {b['baseline_over_candidate'][name]:.4f}× "
            f"|"
        )
    lines += [
        "",
        "时间各自取样本中位；比值按同轮时间相除后取中位，因此表中时间相除不必恰好等于比值。所有比值均为本轮直接测量。",
        "",
        "## 初筛和资源证据",
        "",
        "六个候选均通过完整30×6分布准确性，没有因为正确性失败而调宽合同。下表性能仅为B级完整初筛，单项没有独立性能复测。",
        "",
        "| 候选 | 相对parent初筛geo | 最差 | 结果 |",
        "|---|---:|---:|---|",
    ]
    for variant, values in selection["screen"].items():
        d = values["parent_over_candidate"]
        lines.append(
            f"| {variant} | {d['geomean']:.4f}× | {d['worst']:.4f}× | {'冻结为finalist' if variant == name else '保留，未选'} |"
        )
    lines += [
        "",
        "最终stage2资源（C级静态事实）：",
        "",
        "| E | parent regs/shared/spill | candidate regs/shared/spill |",
        "|---|---|---|",
    ]
    for e in (256, 512):
        a = runs["main"]["kernels"][f"e{e}/parent"][1]
        b = runs["main"]["kernels"][f"e{e}/{name}"][1]
        lines.append(
            f"| {e} | {a['n_regs']}/{a['metadata']['shared']}B/{a['n_spills']} "
            f"| {b['n_regs']}/{b['metadata']['shared']}B/{b['n_spills']} "
            f"|"
        )
    lines += [
        "",
        "gather使原始score存活更久并引入布局通信，shared高于parent；但它消除了逐选择的原始logit重载和重复tanh，实际整体仍更快。",
        "这说明更少寄存器/shared不是唯一目标；未经NCU或完整单因素正式复测，不将全部收益分配给某一个资源指标。",
        "[static_analysis.json](static_analysis.json)记录TTIR的loop/reduce/tanh表达式和PTX的静态指令计数，不能当作动态执行次数。",
        "",
        "## 验证、计时和复现",
        "",
        "官方Triton3.7.0/H100/GPU2独占，所有路径从TTIR经同一原生编译链启动。每次主实验先逐30点核验原baseline源码与文件入口LLVM/PTX/cubin字节一致、输出相等。",
        "完整初筛1440项，主测、独立复测、memcheck、racecheck各540项，"
        "共3600项版本/分布检查通过，含CPU参考、ties、group ties、bias主导、饱和和全负饱和；"
        "检查输入不修改、sentinel与canary。",
        "新候选在全30点上memcheck 0 errors，racecheck 0 errors/0 warnings。filter只检查本实验topk kernel；日志和数值记录分别保存。",
        "主测7轮、独立seed123复测5轮、每轮20ms，三路随机顺序，同等100ms预热。A/A点中位数须在[.97,1.03]；原始输入seed/hash与全部样本保留。",
        f"A/A范围：{summary['aa_ranges']}。增量bootstrap95：{summary['bootstrap95']['main']['parent_over_candidate']} "
        f"/ {summary['bootstrap95']['repeat']['parent_over_candidate']}。"
        f"",
        "预分配完整两kernel CUDA graph设备时间，不含Python/分配/编译；未锁频、未主动清cache。bootstrap仅描述固定30点保存样本的波动。",
        "",
        "```bash",
        "CUDA_VISIBLE_DEVICES=2 /root/venvs/ir-ops-dev/bin/python moe_routing/round2/run.py \\",
        f"  --run my_repeat --variants {name} --rounds 5 --rep 20 --seed 234",
        "```",
        "",
        "输出目录必须新建，脚本先pipeline和完整准确性再计时；--check-only跳过计时。make_candidates.py可以逐字节重建候选，拒绝覆盖不同内容。",
        "[原始主测](results/main/performance.json) · "
        "[独立复测](results/repeat/performance.json) "
        "· [选择](selection.json) · [汇总](summary.json)",
        "",
        "这些结构也能用公开Triton的uint32/bitcast/for、tl.gather、广播比较及tl.sum表达。本轮只直接修改TTIR，未重新做第二轮源码回写或盲化A/B/C搜索效率对照。",
    ]
    (ROOT / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
