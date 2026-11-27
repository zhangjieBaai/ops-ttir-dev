"""Derive report and selection evidence from all original three-way samples."""

import json
import statistics

from common_port import ROOT, digest, save


def load(path):
    return json.loads((ROOT / path).read_text())


def summary(points):
    return {
        ratio: statistics.geometric_mean(p["median_ratios"][ratio] for p in points)
        for ratio in ("baseline_over_ir", "baseline_over_source", "ir_over_source")
    }


def main():
    runs = {
        name: load(f"results/{name}_v2/benchmark.json") for name in ("main", "repeat")
    }
    pipeline = load("results/pipeline_main_v3/pipeline.json")
    assert all(len(r["points"]) == 12 for r in runs.values())
    assert all(p["source_ir_text_equal"] for p in pipeline["points"])
    mem = (ROOT / "results/memcheck.log").read_text()
    race = (ROOT / "results/racecheck.log").read_text()
    assert "ERROR SUMMARY: 0 errors" in mem
    assert "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)" in race
    accuracy = {
        name: load(f"results/{name}/accuracy.json")
        for name in ("accuracy_main_v3", "accuracy_repeat", "memcheck", "racecheck")
    }
    assert all(x["passed"] for x in accuracy.values())
    result = dict(
        source_sha256=digest(ROOT / "frozen_port.py"),
        evidence="A, fixed original 12-point matrix; main7/repeat5 randomized paired rounds",
        groups={},
        all={},
        d64={},
        aa={},
        timing_runs={"main": "main_v2", "repeat": "repeat_v2"},
        excluded_runs=["main", "repeat"],
        performance_validity="All point median A/A ratios pass [0.97, 1.03]",
        sanitizer={
            "memcheck": "0 errors",
            "racecheck": "0 hazards, 0 errors, 0 warnings",
        },
        source_ir_elf_text_equal_all_groups=True,
        accuracy_counts={k: len(v["checks"]) for k, v in accuracy.items()},
    )
    for name, run in runs.items():
        result["all"][name] = summary(run["points"])
        result["d64"][name] = summary([p for p in run["points"] if p["shape"][3] == 64])
        aa = [statistics.median(x["ratio"] for x in p["aa"]) for p in run["points"]]
        assert all(0.97 <= value <= 1.03 for value in aa), (name, aa)
        result["aa"][name] = dict(min=min(aa), max=max(aa))
    for group in ("d64_c0", "d64_c1", "d128_c0", "d128_c1"):
        result["groups"][group] = {
            name: summary([p for p in run["points"] if p["group"] == group])
            for name, run in runs.items()
        }
    save(ROOT / "summary.json", result)
    lines = [
        "# Attention dK/dV：现有 TTIR 优化的 Triton 源码回写",
        "",
        (
            "源码回写完整复现原 `r1_dv_late`：四组编译后的 ELF `.text` 与直接 TTIR 候选逐字节相同，三路配对测量的源码/I"
            "R差异在计时波动量级。D64收益与D128退步都保留；这说明本次优化可由公开Triton源码表达，不构成TTIR独占能力。"
        ),
        "",
        (
            "唯一回写是把概率FP16 cast与dV dot移到dK dot之后。没有进一步调优tile、warp、stage、数学表达或测试数据。原文"
            "件不变；本目录[frozen_port.py](frozen_port.py)在black/isort后冻结，哈希见[source_mani"
            "fest.json](source_manifest.json)。"
        ),
        "",
        "## 三路性能（A级，仅本12点固定配置）",
        "",
        "比值定义为前者耗时/后者耗时；IR/source>1表示源码更快。每点先取同轮比值中位数，再计算跨点几何均值。",
        "",
        (
            "baseline=原baseline.ttir；IR=原r1_dv_late.ttir，四组都测；source=正常Python JIT生成"
            "的CompiledKernel。source计时没有偷换成导出TTIR重编译版本。三路设备ABI、grid及全部后端options一致；Py"
            "thon绑定层为AST产物传递constexpr，但这些不是额外设备参数，计时不包含Python。"
        ),
        "",
        (
            "独占物理GPU4/H100，官方Triton3.7.0，4warps/5stages，完整原12点；每点同轮随机三路顺序，主测7轮、独立se"
            "ed123复测5轮，rep=20ms CUDA graph，每路500ms预热，无锁频、无主动缓存清理。不含forward、预缩放K、delta"
            "、分配、编译、Python。全部原始时间/顺序/输入与产物哈希见[主测](results/main_v2/benchmark.json)及[复测]"
            "(results/repeat_v2/benchmark.json)。"
        ),
        "",
        "| 组 | baseline/IR 主测 / 复测 | baseline/source 主测 / 复测 | IR/source 主测 / 复测 |",
        "|---|---:|---:|---:|",
    ]
    for group, values in result["groups"].items():
        a, b = values["main"], values["repeat"]
        lines.append(
            f"| {group} | {a['baseline_over_ir']:.4f} / {b['baseline_over_ir']:.4f}"
            f" | {a['baseline_over_source']:.4f} / {b['baseline_over_source']:.4f}"
            f" | {a['ir_over_source']:.4f} / {b['ir_over_source']:.4f} |"
        )
    a, b = result["all"]["main"], result["all"]["repeat"]
    lines.append(
        f"| 全12点 | {a['baseline_over_ir']:.4f} / {b['baseline_over_ir']:.4f}"
        f" | {a['baseline_over_source']:.4f} / {b['baseline_over_source']:.4f}"
        f" | {a['ir_over_source']:.4f} / {b['ir_over_source']:.4f} |"
    )
    a, b = result["d64"]["main"], result["d64"]["repeat"]
    lines += [
        "",
        (
            f"D64六点：baseline/source几何 **{a['baseline_over_source']:.4f}× / {b['baseline_over_source']:.4f}"
            f"×**；baseline/IR为{a['baseline_over_ir']:.4f}× / {b['baseline_over_ir']:.4f}"
            f"×。"
        ),
        "",
        (
            f"全12点IR/source几何 **{result['all']['main']['ir_over_source']:.4f}× /"
            f" {result['all']['repeat']['ir_over_source']:.4f}×**。D128两路线都存在退步，正"
            f"式使用应延续父实验D128保留baseline的选择；没有用人为1×覆盖负结果。"
        ),
        "",
        (
            "| (B,H,N,D,causal) | baseline μs | IR μs | source μs | baseline/IR 主/复 |"
            " baseline/source 主/复 | IR/source 主/复 |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for a, b in zip(runs["main"]["points"], runs["repeat"]["points"]):
        assert a["shape"] == b["shape"]
        t = a["median_us"]
        x = a["median_ratios"]
        y = b["median_ratios"]
        lines.append(
            f"| {tuple(a['shape'])} | {t['baseline']:.3f} | {t['ir']:.3f} | {t['source']:.3f}"
            f" | {x['baseline_over_ir']:.4f} / {y['baseline_over_ir']:.4f}"
            f" | {x['baseline_over_source']:.4f} / {y['baseline_over_source']:.4f}"
            f" | {x['ir_over_source']:.4f} / {y['ir_over_source']:.4f} |"
        )
    lines += [
        "",
        "同机器码不保证每轮耗时完全相等。保留全部配对样本，不把计时波动解释为不同的编译调度。",
        "原main/repeat的A/A在大D64非因果点分别为1.0361/1.0307，超出预声明[0.97,1.03]门槛，整轮排除。",
        "本报告仅使用完整main_v2/repeat_v2：所有路线统一500ms预热，仍保持7轮/5轮、rep20ms和原输入seed。",
        "",
        (
            f"A/A各点3对的中位比范围：主测{result['aa']['main']['min']:.4f}–{result['aa']['main']['max']:.4f}"
            f"，复测{result['aa']['repeat']['min']:.4f}–{result['aa']['repeat']['max']:.4f}"
            f"。"
        ),
        "",
        "## Pipeline与机器码证据",
        "",
        (
            "12点逐点验证原baseline源码与原TTIR的LLVM/PTX/cubin字节一致。新源码导出的TTIR与其重编译产物的LLVM/PTX"
            "/cubin也逐字节一致，输出逐位相同，见[pipeline](results/pipeline_main_v3/pipeline.json"
            ")和[独立pipeline](results/pipeline_repeat/pipeline.json)。"
        ),
        "",
        (
            "新源码与直接修改的IR文本因SSA命名及源码位置等存在差异，完整cubin也可能含不同调试信息；这里明确只宣称它们的ELF.text机器代码"
            "段完全一致。前端保留的loop顺序是QK→dP→dK→pT cast→dV，与直接TTIR变换一致，而baseline为QK→cast→dV"
            "→dP→dK。各组ELF.text哈希和完整产物哈希在pipeline中，逐组原始TTIR diff和SASS静态计数见[structura"
            "l_analysis.json](results/structural_analysis.json)。"
        ),
        "",
        "| 组 | baseline regs / spill | IR和source regs / spill | 三路shared bytes |",
        "|---|---:|---:|---:|",
    ]
    seen = set()
    for point in pipeline["points"]:
        g = point["group"]
        if g in seen:
            continue
        seen.add(g)
        m = point["metadata"]
        a, b = m["baseline"], m["source"]
        assert (b["n_regs"], b["n_spills"], b["shared"]) == (
            m["ir"]["n_regs"],
            m["ir"]["n_spills"],
            m["ir"]["shared"],
        )
        lines.append(
            f"| {g} | {a['n_regs']} / {a['n_spills']} | {b['n_regs']} / {b['n_spills']} | {a['shared']} |"
        )
    lines += [
        "",
        "资源/指令为C级静态证据，不把寄存器或spill的改变单独归因为性能因果；未运行NCU。",
        "",
        "## 验证、失败记录与边界",
        "",
        (
            "原精度合同完全继承：原28输入、原分布及FP64数学参考、rtol=0.03/atol=0.008；主测84项和独立seed复测84项（三路"
            "线×28）全通过且source/IR与baseline输出逐位相同。每轮4个小例另做CPU FP64 autograd；检查NaN sent"
            "inel覆盖、有限性及输入不修改，没有删压力case或放宽容差。"
        ),
        "",
        (
            "新源码完整28输入：[memcheck](results/memcheck.log) **0 errors**，[racecheck](re"
            "sults/racecheck.log) **0 hazards / 0 errors / 0 warnings**；各28项数值检查通过。"
            "测试setup为获取JIT产物会额外启动4次源码kernel，同样接受sanitizer检查。"
        ),
        "",
        (
            "早期harness失败完整保留：[debug_dispositions.json](results/debug_dispositions.j"
            "son)。首次错误地给AST CompiledKernel传TTIR绑定列表导致启动异常，随后修正constexpr绑定；v2精度全PASS"
            "但原失败进程未退出，主动排除并隔离重做v3。pipeline v2因tuple/list options比较产生false negative"
            "，改为逐字段JSON规范化比较后全部通过。没有任何数值失败通过放宽门槛处理，也未从失败/重叠运行采集正式性能。"
        ),
        "",
        (
            "原baseline未经逐shape autotune；结果只覆盖连续FP16、N为128倍数、D64/128、causal/noncausa"
            "l及既有LSE/delta/预缩放K，不是完整attention backward或训练step收益。不支持本实验未声明的ragged、dr"
            "opout或其它dtype。"
        ),
        "",
        "复现命令见[README.md](README.md)。本目录没有覆盖父实验的baseline、候选、选择或报告。",
    ]
    (ROOT / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
