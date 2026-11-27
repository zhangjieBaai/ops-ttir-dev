"""Audit raw operator-local evidence and render comparable ablation reports."""

import argparse
import csv
import hashlib
import json
import math
import statistics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geo(values):
    return math.exp(statistics.mean(math.log(x) for x in values))


def compare(result, before, after):
    return geo(
        statistics.median(
            s["times_us"][before] / s["times_us"][after] for s in r["timing"]["samples"]
        )
        for r in result["records"]
    )


def audit(op, path, manifest):
    result = json.loads(path.read_text())
    assert result["passed"] and result["aa_passed"]
    assert len(result["records"]) == len(manifest["records"]) == 30
    assert result["manifest_sha256"] == sha(op / "ablation_v1/manifest.json")
    assert result["names"] == manifest["names"]
    stages = ["kernel"] if op.name == "norm_quant" else ["local", "merge"]
    assert len(result["pipeline"]) == 30 * len(stages)
    assert all(
        x["gate"]["llir_ptx_cubin_identical"] and x["gate"]["full_options_identical"]
        for x in result["pipeline"]
    )
    gates = {(x["group"], x["stage"]): x["gate"] for x in result["pipeline"]}
    for record, item in zip(result["records"], manifest["records"]):
        assert record["group"] == item["group"] and record["shape"] == item["shape"]
        timing = record["timing"]
        aa_ratios = []
        for sample in timing["aa"]:
            ratio = sample["times_us"][0] / sample["times_us"][1]
            assert math.isclose(sample["ratio"], ratio, rel_tol=1e-12)
            aa_ratios.append(ratio)
        aa_median = statistics.median(aa_ratios)
        assert math.isclose(timing["aa_median"], aa_median, rel_tol=1e-12)
        assert 0.97 <= aa_median <= 1.03
        assert len(aa_ratios) == result["rounds"]
        assert len(record["timing"]["samples"]) == result["rounds"]
        assert len(record["input_sha256"]) == (3 if op.name == "norm_quant" else 1)
        for name in result["names"]:
            assert len(record["validation"][name]) == (
                7 if op.name == "norm_quant" else 8
            )
            for stage in stages:
                kernel = record["kernels"][name][stage]
                p = op.parent / item["paths"][name][stage]
                assert sha(p) == kernel["ir_sha256"] == item["sha256"][name][stage]
                assert kernel["elf_text_sha256"]
                gate = gates[(item["group"], stage)]
                assert {k: kernel["metadata"][k] for k in gate["options"]} == gate[
                    "options"
                ]
                if name == "final":
                    selected = op / "multiround/r6/ir" / item["group"]
                    if stage != "kernel":
                        selected /= stage
                    assert sha(selected / "optimized.ttir") == sha(p)
            for sample in record["timing"]["samples"]:
                assert math.isclose(
                    sample["baseline_over"][name],
                    sample["times_us"]["baseline"] / sample["times_us"][name],
                    rel_tol=1e-12,
                )
            ratios = [
                s["times_us"]["baseline"] / s["times_us"][name]
                for s in timing["samples"]
            ]
            assert math.isclose(
                timing["baseline_over"][name], statistics.median(ratios), rel_tol=1e-12
            )
            for route in ("baseline", name):
                median = statistics.median(
                    s["times_us"][route] for s in timing["samples"]
                )
                assert math.isclose(timing["median_us"][route], median, rel_tol=1e-12)
    return result


def render(op, title, descriptions, additions, removals, explanation):
    parser = argparse.ArgumentParser()
    prefix = "split_v2_20261124" if op.name == "partial_topk" else "split_20261124"
    parser.add_argument("--main", default=f"{prefix}_main")
    parser.add_argument("--repeat", default=f"{prefix}_repeat")
    args = parser.parse_args()
    folder = op / "ablation_v1"
    manifest = json.loads((folder / "manifest.json").read_text())
    paths = [
        folder / "results" / run / "results.json" for run in (args.main, args.repeat)
    ]
    main, repeat = [audit(op, p, manifest) for p in paths]
    assert main["seed"] != repeat["seed"]
    for a, b in zip(main["records"], repeat["records"]):
        for n in manifest["names"]:
            for stage, kernel in a["kernels"][n].items():
                other = b["kernels"][n][stage]
                assert kernel["stage_sha256"] == other["stage_sha256"]
                assert kernel["elf_text_sha256"] == other["elf_text_sha256"]
    lines = [
        f"# {title}：最终版 TTIR 消融",
        "",
        "固定第六轮最终版，全部30点；原baseline、冻结源码、最终别名和历史结果不改。"
        "每组final与r6 optimized逐字节相同。本轮是事后受控消融，没有重新选优。",
        "",
        f"官方Triton3.7.0/H100；主测{main['rounds']}轮、独立seed{repeat['seed']}复测"
        f"{repeat['rounds']}轮，每轮{main['rep_ms']}ms。全部路径100ms预热。"
        "预分配CUDA graph设备序列，不含编译、分配或CPU launch；未主动清cache/锁频。",
        "",
        f"主测独占GPU{main['environment']['device']}，独立复测独占GPU{repeat['environment']['device']}。"
        "只在每次会话内比较baseline/候选比值，不跨GPU比较绝对时延。",
        "",
        "## 各版本相对原始baseline",
        "",
        "| 版本 | 改动 | 主测几何加速 | 独立复测 | 复测最差点 |",
        "|---|---|---:|---:|---:|",
    ]
    summary = dict(
        evidence_grade="A",
        sources={str(p.relative_to(op)): sha(p) for p in paths},
        manifest_sha256=sha(folder / "manifest.json"),
        shape_count=30,
        versions={},
        additions=[],
        removals=[],
    )
    for name in manifest["names"]:
        values = []
        for result in (main, repeat):
            ratios = [r["timing"]["baseline_over"][name] for r in result["records"]]
            values.append(
                dict(geomean=geo(ratios), worst=min(ratios), best=max(ratios))
            )
        summary["versions"][name] = dict(main=values[0], repeat=values[1])
        lines.append(
            f"| `{name}` | {descriptions[name]} | {values[0]['geomean']:.6f}× | "
            f"{values[1]['geomean']:.6f}× | {values[1]['worst']:.6f}× |"
        )
    for heading, pairs, key in (
        ("逐项加入的条件收益", additions, "additions"),
        ("从最终版移除改动", removals, "removals"),
    ):
        lines.extend(
            [
                "",
                f"## {heading}",
                "",
                "| 对照（前者耗时/后者耗时） | 主测 | 独立复测 |",
                "|---|---:|---:|",
            ]
        )
        for before, after, label in pairs:
            a, b = compare(main, before, after), compare(repeat, before, after)
            lines.append(f"| {label}：`{before}` / `{after}` | {a:.6f}× | {b:.6f}× |")
            summary[key].append(
                dict(before=before, after=after, description=label, main=a, repeat=b)
            )
    lines.extend(
        [
            "",
            "比值逐shape取同轮时间比中位数后求几何均值。各版本在同一轮随机交错，"
            "不是相邻版本直接连续计时；这和将两次独立实验的均值相除不同。"
            "大于1表示后者更快。存在交互，不能相加为总收益；未计算置信区间，"
            "不足约1%或两次方向不同的变化不据此宣称稳定有效。",
            "",
            explanation,
            "",
            "## 正确性、编译和资源",
            "",
        ]
    )
    for label, result in (("主测", main), ("复测", repeat)):
        count = sum(len(c) for r in result["records"] for c in r["validation"].values())
        aa = [r["timing"]["aa_median"] for r in result["records"]]
        lines.append(
            f"{label}：{len(result['pipeline'])}项源码/文件baseline的LLVM/PTX/cubin和options一致性核验；"
            f"{count}项版本×分布数值检查；A/A中位数范围[{min(aa):.6f}, {max(aa):.6f}]，全通过。"
        )
    lines.extend(
        [
            "",
            "normal使用完整M；其他压力分布及CPU FP64参考按原验证器缩减行数。"
            "保留canary、输入不修改、输入seed/哈希及全部IR/下游哈希。"
            "最终版sanitizer沿用r6已存证据；本轮没有对全部中间消融重新做sanitizer。",
            "",
            "| 版本 | kernel/stage | registers范围 | spills范围 | shared bytes范围 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for name in manifest["names"]:
        for stage in main["records"][0]["kernels"][name]:
            kernels = [r["kernels"][name][stage] for r in main["records"]]
            ranges = []
            for getter in (
                lambda k: k["n_regs"],
                lambda k: k["n_spills"],
                lambda k: k["metadata"]["shared"],
            ):
                vs = [getter(k) for k in kernels]
                ranges.append(f"{min(vs)}–{max(vs)}")
            lines.append(f"| `{name}` | {stage} | " + " | ".join(ranges) + " |")
    lines.extend(
        [
            "",
            "资源为C级静态证据，范围覆盖不同N/配置；不能由寄存器或静态归约数证明性能因果。",
            "",
            "## 全部shape",
            "",
            "| shape | 主测final | 复测final |",
            "|---|---:|---:|",
        ]
    )
    with (folder / "all_shapes.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["phase", "shape", "version", "baseline_us", "version_us", "paired_speedup"]
        )
        for phase, result in (("main", main), ("repeat", repeat)):
            for r in result["records"]:
                for n in manifest["names"]:
                    writer.writerow(
                        [
                            phase,
                            "x".join(map(str, r["shape"])),
                            n,
                            r["timing"]["median_us"]["baseline"],
                            r["timing"]["median_us"][n],
                            r["timing"]["baseline_over"][n],
                        ]
                    )
    for a, b in zip(main["records"], repeat["records"]):
        lines.append(
            f"| {'×'.join(map(str, a['shape']))} | {a['timing']['baseline_over']['final']:.6f}× | "
            f"{b['timing']['baseline_over']['final']:.6f}× |"
        )
    lines.extend(
        [
            "",
            f"证据：[主测]({paths[0].relative_to(op)}) · [独立复测]({paths[1].relative_to(op)}) · "
            "[全部CSV](ablation_v1/all_shapes.csv) · [核验汇总](ablation_v1/summary.json) · "
            "[预声明计划](ablation_plan.md)。",
            "",
            "复现：在仓库根设置独占`CUDA_VISIBLE_DEVICES`，执行本算子的`run_experiment.sh`；"
            "使用新的`RUN_ID`，结果目录存在会拒绝覆盖。报告生成器重算原始样本比值，"
            "核查验证项数、A/A、最终选择和主/复测各阶段哈希，再生成本文。",
            "",
        ]
    )
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (op / "report_ablation.md").write_text("\n".join(lines))
