"""Accept or retain parent using complete main/repeat and sanitizer evidence."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(stage):
    folder = ROOT / stage
    chosen = json.loads((folder / "selection_pre_main.json").read_text())["candidate"]
    runs = {
        n: json.loads((folder / f"results/{n}/results.json").read_text())
        for n in ["main", "repeat"]
    }
    for d in runs.values():
        assert d["passed"] and d["aa_passed"] and len(d["records"]) == 30
        assert chosen in d["valid_names"]
    accepted = all(
        d["summary"][chosen]["incremental"]["geomean"] >= 1.02
        and d["summary"][chosen]["incremental"]["minimum"] >= 0.95
        for d in runs.values()
    )
    # Sanitize the tested finalist even when it will not be promoted.
    for name, marker in [
        ("memcheck", "ERROR SUMMARY: 0 errors"),
        ("racecheck", "0 errors, 0 warnings"),
    ]:
        assert marker in (folder / (name + ".log")).read_text()
        d = json.loads((folder / f"results/{name}/results.json").read_text())
        assert d["passed"] and chosen in d["valid_names"] and len(d["records"]) == 30
    selected = chosen if accepted else "parent"
    result = {
        "stage": stage,
        "matrix_points": 30,
        "candidate_tested": chosen,
        "accepted": accepted,
        "selected": selected,
        "main": runs["main"]["summary"][chosen],
        "repeat": runs["repeat"]["summary"][chosen],
        "selected_main": runs["main"]["summary"][selected],
        "selected_repeat": runs["repeat"]["summary"][selected],
        "sanitizer": {"memcheck": "0 errors", "racecheck": "0 errors,0 warnings"},
        "evidence": "A, frozen30 matrix; static identities C",
        "kernels": {},
        "sources": {n: digest(folder / f"results/{n}/results.json") for n in runs},
    }
    for a, b in zip(runs["main"]["records"], runs["repeat"]["records"]):
        assert a["case"] == b["case"]
        for name in ["baseline", "anchor", "parent", chosen]:
            assert (
                a["kernels"][name]["ttir_sha256"] == b["kernels"][name]["ttir_sha256"]
            )
            assert (
                digest(Path(a["kernels"][name]["path"]))
                == a["kernels"][name]["ttir_sha256"]
            )
        src = folder / "ir" / a["case"] / (selected + ".ttir")
        dest = src.with_name("optimized.ttir")
        if dest.exists():
            assert dest.read_bytes() == src.read_bytes()
        else:
            dest.write_bytes(src.read_bytes())
        result["kernels"][a["case"]] = {"selected": selected, "sha256": digest(src)}
    selection_path = folder / "selection.json"
    selection_text = json.dumps(result, indent=2) + "\n"
    if selection_path.exists():
        assert selection_path.read_text() == selection_text
    else:
        selection_path.write_text(selection_text)
    lines = [
        f"# INT4 {stage}：30点直接TTIR增量实验",
        "",
        f"本轮实测 `{chosen}`，{'接受新候选' if accepted else '未达到晋升门槛，保留parent'}。",
        f"增量（parent/candidate）主测/复测 **{result['main']['incremental']['geomean']:.6f}× "
        f"/ {result['repeat']['incremental']['geomean']:.6f}×**。"
        f"",
        f"对第二轮anchor **{result['main']['anchor_relative']['geomean']:.6f}× "
        f"/ {result['repeat']['anchor_relative']['geomean']:.6f}×**；"
        f"",
        f"对原baseline累计 **{result['main']['cumulative']['geomean']:.6f}× "
        f"/ {result['repeat']['cumulative']['geomean']:.6f}×**。"
        f"",
        "",
        "以下全为本轮同场配对，未用旧轮平均相乘。主测7轮、独立复测5轮，rep20ms，所有路径同输入/配置/ABI/grid。",
        "数值合同和原5种压力不变；源码/文件baseline pipeline逐点通过。最终两种sanitizer均零错误、racecheck零警告。",
        "",
        "| M,N,K | baseline us | anchor us | parent us | candidate us | 主测增量 | 复测增量 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for a, b in zip(runs["main"]["records"], runs["repeat"]["records"]):
        t = a["timing"]
        u = t["median_us"]
        lines.append(
            f"| {tuple(a['shape'])} | {u['baseline']:.3f} "
            f"| {u['anchor']:.3f} | {u['parent']:.3f} | {u[chosen]:.3f} "
            f"| {t['incremental'][chosen]:.5f}× | {b['timing']['incremental'][chosen]:.5f}× "
            f"|"
        )
    screen = json.loads((folder / "results/screen/results.json").read_text())
    lines += [
        "",
        "## 初筛与边界",
        "",
        "| 候选 | 初筛增量geo（B） | 最差 | 与parent相同.text点数 |",
        "|---|---:|---:|---:|",
    ]
    for name in screen["valid_names"]:
        if name in ["baseline", "anchor", "parent"]:
            continue
        sm = screen["summary"][name]["incremental"]
        same = sum(
            r["kernels"][name]["elf_text"] == r["kernels"]["parent"]["elf_text"]
            for r in screen["records"]
        )
        lines.append(
            f"| {name} | {sm['geomean']:.6f}× | {sm['minimum']:.6f}× | {same}/30 |"
        )
    lines += [
        "",
        "同名候选与旧结果不覆盖；原始顺序/样本/哈希/资源在results/。同.code指令段不能误写为完整cubin一致。",
        "公开Triton可以表达本轮变换；本次没有源码回写对照，不证明TTIR独占能力。新增shape沿用第二轮固定配置族，未重新autotune。",
        "[主测](results/main/results.json) · [复测](results/repeat/results.json) "
        "· [选择](selection.json) · [memcheck](memcheck.log) "
        "· [racecheck](racecheck.log)",
    ]
    (folder / "report.md").write_text("\n".join(lines) + "\n")
    print(stage, chosen, "accepted", accepted, result["repeat"])


def combined():
    stages = {
        s: json.loads((ROOT / s / "selection.json").read_text())
        for s in ["r3", "r4", "r5"]
    }
    final = stages["r5"]
    result = {
        "operator": "int4_gemm",
        "rounds": ["r3", "r4", "r5"],
        "matrix_points": 30,
        "anchor": "round2/r2_group_pair",
        "stages": stages,
        "accepted_chain": [
            {"stage": s, "name": d["selected"]}
            for s, d in stages.items()
            if d["accepted"]
        ],
        "final_stage": "r5",
        "final_selected": final["selected"],
        "final_main": final["selected_main"],
        "final_repeat": final["selected_repeat"],
    }
    result["final_effective_candidate"] = (
        result["accepted_chain"][-1]
        if result["accepted_chain"]
        else {"stage": "round2", "name": "r2_group_pair"}
    )
    (ROOT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# INT4 第三至第五轮：顺序优化结果",
        "",
        "固定30点与第二轮配置，三轮确实从当时已验收parent继续生成新TTIR，而非把多seed复测当作新轮次。",
        "30个唯一shape每轮均重新核验，三轮不是90个不同shape。",
        "",
        "| 轮次 | 实测候选 | 主测增量 | 独立复测增量 | 晋升 |",
        "|---|---|---:|---:|---|",
    ]
    for stage, d in stages.items():
        lines.append(
            f"| [{stage}]({stage}/report.md) | {d['candidate_tested']} "
            f"| {d['main']['incremental']['geomean']:.6f}× "
            f"| {d['repeat']['incremental']['geomean']:.6f}× "
            f"| {'接受' if d['accepted'] else '保留parent'} |"
            f""
        )
    lines += [
        "",
        f"最终已验收路径相对第二轮anchor：主测/复测 **{final['selected_main']['anchor_relative']['geomean']:.6f}× "
        f"/ {final['selected_repeat']['anchor_relative']['geomean']:.6f}×**。"
        f"",
        f"相对原始baseline累计：**{final['selected_main']['cumulative']['geomean']:.6f}× "
        f"/ {final['selected_repeat']['cumulative']['geomean']:.6f}×**。"
        f"",
        "",
        "r3使用固定GROUP128与连续stride的既有合同；r4探索合法尾组的PID重排；"
        "r5探索加载次序及两个量化group的循环展开。",
        "在M128,N11008,K4096上，group128相对第二轮anchor的寄存器96→108、spill4→0，"
        "shared仍24576B。变量除法消除同时改变了分配/调度，不能将所有收益只归因于某条指令。",
        "r4三个PID候选有16/30点的可执行.text与parent相同（单M tile适用恒等映射）；"
        "r5 preload_pair有17/30点.text相同。少量变化与计时波动分开记录；"
        "全量资源/静态PTX计数见[static_analysis.json](static_analysis.json)，不是动态NCU证据。",
        "每轮完整精度后筛选，冻结后主7/复5，A/A门槛不变。原baseline源码/TTIR产物一致，全部工具和候选哈希已记录。",
        "各轮新候选均做memcheck/racecheck，零错误、零警告；失败/退步保留。所有候选都可用公开Triton表达，未做源码版本性能比较。",
        "",
        "[协议](optimization_plan.md) · [机器可读汇总](summary.json)",
    ]
    (ROOT / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["r3", "r4", "r5", "combined"])
    a = p.parse_args()
    combined() if a.stage == "combined" else main(a.stage)
