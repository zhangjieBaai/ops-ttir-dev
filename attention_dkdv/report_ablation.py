"""Validate raw paired evidence and render the independent Attention ablation."""

import argparse
import csv
import hashlib
import io
import json
import os
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS = {
    "baseline": "原始128-key基线",
    "dv_only": "原始128-key + dV后移",
    "rows64": "两个64-key子块",
    "structure": "64-key + 指针重算/静态展开",
    "rows64_dv": "64-key + dV后移（移除结构调整）",
    "final": "冻结最终版",
}
STEPS = [
    ("baseline", "rows64", "加入64-key子块"),
    ("rows64", "structure", "再加入指针重算/静态展开"),
    ("structure", "final", "再后移dV；亦为最终版移除dV的对照"),
    ("rows64_dv", "final", "最终版移除指针重算/展开，保留dV"),
    ("dv_only", "final", "最终版联合移除64-key及其结构调整，保留dV"),
    ("rows64", "rows64_dv", "仅在64-key结构上加入dV后移"),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ratio(row, before, after):
    return statistics.median(s["us"][before] / s["us"][after] for s in row["samples"])


def mean(rows, before, after):
    return statistics.geometric_mean(ratio(row, before, after) for row in rows)


def load(path):
    data = json.loads((path / "benchmark.json").read_text())
    accuracy = json.loads((path / "accuracy.json").read_text())
    assert data["passed"] and len(data["points"]) == 32
    assert len({tuple(row["shape"]) for row in data["points"]}) == 32
    assert accuracy["passed"] and accuracy["kernel_checks"] == 176
    assert len(accuracy["cases"]) == 44
    assert data["accuracy_sha256"] == digest(path / "accuracy.json")
    assert data["resources_sha256"] == digest(path / "resources.json")
    assert data["pipeline_sha256"] == digest(path.parent / "pipeline/pipeline.json")
    for row in data["points"]:
        assert len(row["samples"]) == data["rounds"]
        aa = ratio(row, "baseline", "baseline_copy")
        assert 0.97 <= aa <= 1.03
    return data


def render(main_path, repeat_path, summary_dir=None):
    runs = [load(main_path), load(repeat_path)]
    assert [run["seed"] for run in runs] == [20, 123]
    assert [run["rounds"] for run in runs] == [7, 5]
    resources = [
        json.loads((path / "resources.json").read_text())
        for path in (main_path, repeat_path)
    ]
    for a, b in zip(runs[0]["points"], runs[1]["points"]):
        assert a["shape"] == b["shape"] and a["ttir_sha256"] == b["ttir_sha256"]
    for group in resources[0]:
        for name in resources[0][group]:
            assert (
                resources[0][group][name]["hashes"]
                == resources[1][group][name]["hashes"]
            )
    lines = [
        "# Attention dK/dV 最终版逐项消融",
        "",
        "固定第六轮有效选择，原始基线、optimized与冻结源码均不修改。官方Triton3.7.0 / H100 / SM1200MHz，"
        "FP16、4warps/5stages/1CTA；D64/128及因果/非因果共32点。只计预分配CUDA graph中的dK/dV kernel，"
        "不含forward、K预缩放、delta准备、分配、编译或Python launch。",
        "",
        "本轮为A 级性能证据：主测seed20/7轮、独立进程与设备seed123/5轮，rep20ms；"
        "每路初始实际预热≥500ms，每次测量前≥100ms。每个shape/每轮随机交错候选与baseline副本。"
        "数值先于全部性能测量：各自32点源码/文件入口LLVM/PTX/cubin一致，44输入/176项数值检查全部通过；"
        "含逐位输出、FP64数学参考、有限性/完整覆盖和输入不变。",
        "",
        "## 各版本相对原始baseline",
        "",
        "D128每组8点。以下每点先取同轮baseline/版本配对比值中位数，再取组内几何均值，"
        "没有相乘历史搜索轮的平均值。D64每组8点仅有dV后移，单独列出。",
        "",
        "| 版本 | D128非因果主测 | 独立复测 | D128因果主测 | 独立复测 |",
        "|---|---:|---:|---:|---:|",
    ]
    stage_summary = {}
    for name in ("dv_only", "rows64", "structure", "rows64_dv", "final"):
        values = []
        for causal in (False, True):
            for run in runs:
                rows = [p for p in run["points"] if p["shape"][3:] == [128, causal]]
                values.append(mean(rows, "baseline", name))
        stage_summary[name] = values
        lines.append(
            "| " + LABELS[name] + " | " + " | ".join(f"{v:.4f}×" for v in values) + " |"
        )
    for d in (64, 128):
        values = [
            mean([p for p in r["points"] if p["shape"][3] == d], "baseline", "final")
            for r in runs
        ]
        lines += [
            "",
            f"D{d}全部16点最终版：主测 **{values[0]:.4f}×**，独立复测 **{values[1]:.4f}×**。",
        ]
    final = [mean(r["points"], "baseline", "final") for r in runs]
    lines += [
        "",
        f"全部32点最终版：主测 **{final[0]:.4f}×**，独立复测 **{final[1]:.4f}×**。",
        "",
        "## 逐项加入与从最终版移除",
        "",
        "表内为前一版耗时/后一版耗时的几何均值，均来自同场、同轮配对。大于1表示后一版更快。",
        "",
        "| 对照 | 非因果主测 | 复测 | 因果主测 | 复测 |",
        "|---|---:|---:|---:|---:|",
    ]
    effects = {}
    for before, after, label in STEPS:
        values = [
            mean(
                [p for p in run["points"] if p["shape"][3:] == [128, causal]],
                before,
                after,
            )
            for causal in (False, True)
            for run in runs
        ]
        effects[label] = values
        lines.append(
            "| " + label + " | " + " | ".join(f"{v:.4f}×" for v in values) + " |"
        )
    lines += [
        "",
        "移除64-key后指针重算/子块展开的原定义不再成立，因此只能联合移除这两项，不能把该行称作纯64-key贡献。"
        "rows64→rows64_dv与structure→final是dV重排在不同结构上的条件效应；rows64_dv→final的对照保留dV、只去掉指针重算/展开，"
        "补齐了旧版逐项加入实验缺少的最终移除对照。各项有依赖和交互，百分比不能相加；小差异不据此证明可靠收益。",
        "",
        "## 静态资源与因果边界（C 级）",
        "",
        "| 组 | 版本 | registers | spills | shared B |",
        "|---|---|---:|---:|---:|",
    ]
    for group, versions in resources[0].items():
        for name, info in versions.items():
            lines.append(
                f"| {group} | {name} | {info['registers']} | {info['spills']} | {info['shared_bytes']} |"
            )
    lines += [
        "",
        "资源和机器码哈希只提供静态线索，不能单独证明时延变化由spill、寄存器或某条调度引起。"
        "所有候选保留每个accumulator的dot遍历次序；D128缩小子块也增加读取Q/dO的次数。",
        "",
        "## 公开Triton表达与负结果",
        "",
        "dV后移可用源码语句次序表达，串行子块可用循环表达，指针重算用索引表达式，子块展开可用tl.static_range。"
        "已有[source_port](source_port/report.md)只回写首轮实现；本补测没有重新实现或计时r6源码版，"
        "不能用首轮源码结果代替当前最终版的源码/TTIR性能对照。",
        "",
        "原128-key基线上仅后移dV在D128存在负收益，见本表dv_only；"
        "[r6分离dV/dK遍历](multiround/r6/report.md)初筛为0.7290×/0.6604×（B级），未采用。"
        "所有旧实验和失败候选均保留，本补测不重新选择optimized。",
        "",
        "## 全部shape与证据",
        "",
        "| (B,H,N,D,causal) | baseline/final主测 | 独立复测 |",
        "|---|---:|---:|",
    ]
    csv_stream = io.StringIO()
    writer = csv.writer(csv_stream)
    writer.writerow(
        [
            "run",
            "B",
            "H",
            "N",
            "D",
            "causal",
            "variant",
            "baseline_over_variant",
            "median_us",
        ]
    )
    for a, b in zip(runs[0]["points"], runs[1]["points"]):
        lines.append(
            f"| {tuple(a['shape'])} | {ratio(a, 'baseline', 'final'):.4f}× | {ratio(b, 'baseline', 'final'):.4f}× |"
        )
    for run_name, run in zip(("main", "repeat"), runs):
        for row in run["points"]:
            for name in row["ttir_sha256"]:
                writer.writerow(
                    [
                        run_name,
                        *row["shape"],
                        name,
                        ratio(row, "baseline", name),
                        statistics.median(s["us"][name] for s in row["samples"]),
                    ]
                )
    aa = [
        [min(p["aa"] for p in r["points"]), max(p["aa"] for p in r["points"])]
        for r in runs
    ]
    lines += [
        "",
        f"A/A各点中位比范围：主测{aa[0][0]:.4f}–{aa[0][1]:.4f}，复测{aa[1][0]:.4f}–{aa[1][1]:.4f}。",
        "",
        "[实验契约及复现命令](ablation_plan.md)；主测与复测各自保存benchmark、accuracy、resources及pipeline JSON。"
        "新消融候选rows64_dv没有另跑sanitizer；冻结最终路线的memcheck/racecheck通过记录见[r6报告](multiround/r6/report.md)。"
        "旧D128逐项加入记录在[ablation](ablation/)保留。",
    ]
    lines += ["", "原始证据："]
    for label, path in zip(("主测", "独立复测"), (main_path, repeat_path)):
        rel = Path(os.path.relpath(path, ROOT))
        lines += [
            "",
            f"- {label}：[全部配对样本]({rel}/benchmark.json)、"
            f"[数值门槛]({rel}/accuracy.json)、[资源与哈希]({rel}/resources.json)、"
            f"[编译路径]({rel.parent}/pipeline/pipeline.json)。",
        ]
    summary_rel = Path(
        os.path.relpath(summary_dir or ROOT / "results/ablation_v2/summary", ROOT)
    )
    lines += [
        "",
        f"[全部shape和版本CSV]({summary_rel}/all_shapes.csv) · "
        f"[机器可读汇总与来源哈希]({summary_rel}/summary.json)",
    ]
    sources = {}
    for path in (main_path, repeat_path):
        for name in ("benchmark.json", "accuracy.json", "resources.json"):
            item = path / name
            sources[os.path.relpath(item, ROOT)] = digest(item)
    summary = dict(
        evidence_level="A",
        matrix_points=32,
        final_geomean=final,
        aa=aa,
        stages=stage_summary,
        conditional_effects=effects,
        sources=sources,
    )
    return "\n".join(lines) + "\n", csv_stream.getvalue(), summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main", type=Path, default=ROOT / "results/ablation_v2/main/measurement"
    )
    parser.add_argument(
        "--repeat", type=Path, default=ROOT / "results/ablation_v2/repeat/measurement"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--report", type=Path, help="Optional fresh public Markdown path"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.report is not None and args.report.exists():
        raise FileExistsError(args.report)
    markdown, csv_text, summary = render(
        args.main.resolve(), args.repeat.resolve(), args.output.resolve()
    )
    args.output.mkdir(parents=True)

    def linked_markdown(folder):
        return re.sub(
            r"\]\(([^)]+)\)",
            lambda m: "](" + os.path.relpath(ROOT / m[1], folder.resolve()) + ")",
            markdown,
        )

    (args.output / "report.md").write_text(linked_markdown(args.output))
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(linked_markdown(args.report.parent))
    (args.output / "all_shapes.csv").write_text(csv_text)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
