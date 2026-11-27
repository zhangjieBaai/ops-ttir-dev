"""Qualify each round and expose the actual accepted parent for the next round."""

import json
import math
import statistics

from common_mr import ROOT, digest, save
from generate import immutable


def group_summary(rows, name):
    return {
        metric: {
            "geo": math.exp(
                statistics.mean(math.log(r["ratios"][metric][name]) for r in rows)
            ),
            "min": min(r["ratios"][metric][name] for r in rows),
            "max": max(r["ratios"][metric][name] for r in rows),
        }
        for metric in ("parent_over", "anchor_over", "baseline_over")
    }


def main():
    main = json.loads((ROOT / "results/main/benchmark.json").read_text())
    repeat = json.loads((ROOT / "results/repeat/benchmark.json").read_text())
    chosen = json.loads((ROOT / "selection_pre_main.json").read_text())
    for data in (main, repeat):
        assert data["passed"] and data["aa_passed"] and len(data["points"]) == 32
    for tool, marker in (
        ("memcheck", "ERROR SUMMARY: 0 errors"),
        ("racecheck", "0 errors, 0 warnings"),
    ):
        assert marker in (ROOT / (tool + ".log")).read_text()
        acc = json.loads((ROOT / f"results/{tool}/accuracy.json").read_text())
        assert acc["passed"] and all(acc["valid"].values())
    result = {
        "round": ROOT.name,
        "matrix_points": 32,
        "groups": {},
        "clock_mhz": 1200,
        "evidence_level": "A for full32 paired main/repeat; static resources C",
        "aa": {
            label: [
                min(r["aa_median"] for r in d["points"]),
                max(r["aa_median"] for r in d["points"]),
            ]
            for label, d in (("main", main), ("repeat", repeat))
        },
        "sanitizer": {"memcheck": "0 errors", "racecheck": "0 errors,0 warnings"},
    }
    lines = [
        f"# Attention {ROOT.name}：顺序TTIR搜索",
        "",
        "| 组 | 实测候选 | 主测增量 | 复测增量 | 复测相对R2锚点 | 复测累计 | 是否采用 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for g, name in chosen["tested_candidate"].items():
        m = group_summary([r for r in main["points"] if r["group"] == g], name)
        r = group_summary([r for r in repeat["points"] if r["group"] == g], name)
        accepted = all(
            s["parent_over"]["geo"] >= 1.02 and s["parent_over"]["min"] >= 0.95
            for s in (m, r)
        )
        selected = name if accepted else "parent"
        folder = ROOT / "ir" / g
        immutable(folder / "optimized.ttir", (folder / f"{selected}.ttir").read_text())
        result["groups"][g] = {
            "tested": name,
            "main": m,
            "repeat": r,
            "accepted": accepted,
            "selected": selected,
            "selected_sha256": digest(folder / "optimized.ttir"),
        }
        lines.append(
            f"| {g} | {name} | {m['parent_over']['geo']:.6f} | {r['parent_over']['geo']:.6f} | "
            f"{r['anchor_over']['geo']:.6f} | {r['baseline_over']['geo']:.6f} | {accepted} |"
        )
    result["sources"] = {
        rel: digest(ROOT / rel)
        for rel in (
            "results/main/benchmark.json",
            "results/repeat/benchmark.json",
            "selection_pre_main.json",
        )
    }
    save(ROOT / "selection.json", result)
    save(ROOT / "summary.json", result)
    descriptions = {
        "r6": "D128接受的64-key结构上后移dV，或在同一CTA内分为先dV后dK的独立query遍历以减少共同存活累加器；D64使用原parent控制。",
        "r3": "结构组合：D128合并64-row外层展开与指针重算；D64将dP提前与既有dV后移结合。另一候选将相邻两次查询迭代合并，保持原累加顺序。",
        "r4": "只对循环Q/dO load使用cg，以及在当前parent上提前delta读取；这两个实验区别于在原始baseline上统一移动/改写所有load。",
        "r5": "输出DK/DV的store使用cs，以及Q/dO load使用ca+evict_last；两者从r4已验收parent出发检验缓存交互。",
    }
    lines += [
        "",
        descriptions[ROOT.name],
        "",
        "| shape | baseline us | anchor us | parent us | candidate us | 主测增量 | 独立复测增量 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for a, b in zip(main["points"], repeat["points"]):
        assert a["shape"] == b["shape"]
        name = chosen["tested_candidate"][a["group"]]
        us = a["median_us"]
        lines.append(
            f"| {tuple(a['shape'])} | {us['baseline']:.3f} | {us['anchor']:.3f} | "
            f"{us['parent']:.3f} | {us[name]:.3f} | {a['ratios']['parent_over'][name]:.6f} | "
            f"{b['ratios']['parent_over'][name]:.6f} |"
        )
    lines += [
        "",
        "全部比值来自本轮同场配对；未接受候选的负结果保留，不代表最终选择的性能。",
        "官方Triton3.7.0、GPU4固定SM1200MHz，原32点、ABI、grid、options与逐位输出合同不变。",
        "主测7轮rep20，独立seed123复测5轮rep20，A/A门槛[0.97,1.03]，没有重新autotune。",
        "计时前核验完整原source/file baseline pipeline及44输入精度。最终候选通过两种sanitizer。",
        "实际输入/IR/产物哈希、全部外层配对样本和资源在results/；库rep参数不是实际总GPU时长。",
        "本轮只改TTIR，变换也有公开Triton表达方式；没有测试本轮源码回写性能。",
        "[完整主测](results/main/benchmark.json) · [独立复测](results/repeat/benchmark.json) · "
        "[协议](optimization_plan.md) · [生成器](../generate.py)",
    ]
    (ROOT / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
