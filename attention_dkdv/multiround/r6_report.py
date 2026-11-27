"""Audit and report r6 final effective routes using directly paired measurements."""

import ast
import hashlib
import json
import math
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE / "r6"
GROUPS = ["d64_c0", "d64_c1", "d128_c0", "d128_c1"]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    return {
        "geomean": math.exp(statistics.mean(math.log(x) for x in values)),
        "worst": min(values),
        "best": max(values),
    }


def main():
    selection = read(ROOT / "selection.json")
    screening = read(ROOT / "selection_pre_main.json")
    clock = read(ROOT / "clock_run.json")
    assert clock["passed"] and clock["restored"] and clock["restore"]["exit_code"] == 0
    assert all(x["exit_code"] == 0 for x in clock["commands"])
    sources = {}
    final = {}
    final_rows = {}
    subsets = {g: {} for g in ["d64", "d128", *GROUPS]}
    counts = {}
    for group in GROUPS:
        folder = ROOT / "ir" / group
        assert (folder / "parent.ttir").read_bytes() == (
            BASE / "r5/ir" / group / "optimized.ttir"
        ).read_bytes()
        assert (folder / "anchor.ttir").read_bytes() == (
            BASE.parent / "round2/ir" / group / "optimized.ttir"
        ).read_bytes()
        assert (
            digest(folder / "optimized.ttir")
            == selection["groups"][group]["selected_sha256"]
        )
    for label in ["screen", "main", "repeat", "memcheck", "racecheck"]:
        rel = (
            f"results/accuracy_{label}/accuracy.json"
            if label in ["screen", "main", "repeat"]
            else f"results/{label}/accuracy.json"
        )
        acc = read(ROOT / rel)
        assert acc["passed"] and all(acc["valid"].values())
        assert all(
            c["status"] == "PASS" and c["bitwise_baseline"] for c in acc["checks"]
        )
        counts[label] = len(acc["checks"])
        sources[rel] = digest(ROOT / rel)
        for key, rec in acc["compile"].items():
            name, dim, causal = ast.literal_eval(key)
            assert rec["status"] == "PASS"
            assert rec["metadata"]["ttir_sha256"] == digest(
                ROOT / "ir" / f"d{dim}_c{int(causal)}" / (name + ".ttir")
            )
    for phase in ["main", "repeat"]:
        path = ROOT / f"results/{phase}/benchmark.json"
        data = read(path)
        assert data["passed"] and data["aa_passed"] and len(data["points"]) == 32
        assert (
            data["environment"]["clocks_locked"]
            and data["environment"]["requested_sm_clock_mhz"] == 1200
        )
        for group, info in selection["groups"].items():
            ratios = [
                statistics.median(
                    s["times_us"]["parent"] / s["times_us"][info["tested"]]
                    for s in p["samples"]
                )
                for p in data["points"]
                if p["group"] == group
            ]
            assert len(ratios) == 8
            actual = stats(ratios)
            assert math.isclose(actual["geomean"], info[phase]["parent_over"]["geo"])
            assert math.isclose(actual["worst"], info[phase]["parent_over"]["min"])
        sources[str(path.relative_to(ROOT))] = digest(path)
        rows = []
        for point in data["points"]:
            group = point["group"]
            route = selection["groups"][group]["selected"]
            assert (
                point["metadata"][route]["ttir_sha256"]
                == selection["groups"][group]["selected_sha256"]
            )
            aa = statistics.median(
                x["times_us"][0] / x["times_us"][1] for x in point["aa"]
            )
            assert 0.97 <= aa <= 1.03
            ratios = {
                f"{num}_over_final": statistics.median(
                    s["times_us"][num] / s["times_us"][route] for s in point["samples"]
                )
                for num in ["parent", "anchor", "baseline"]
            }
            rows.append(
                {
                    "shape": point["shape"],
                    "group": group,
                    "selected_route": route,
                    "ratios": ratios,
                    "median_us": point["median_us"],
                    "selected_sha256": point["metadata"][route]["ttir_sha256"],
                }
            )
        final_rows[phase] = rows
        final[phase] = {
            key: stats([r["ratios"][key] for r in rows]) for key in rows[0]["ratios"]
        }
        for group in subsets:
            selected = [
                r
                for r in rows
                if r["group"] == group or r["group"].startswith(group + "_")
            ]
            subsets[group][phase] = {
                key: stats([r["ratios"][key] for r in selected])
                for key in rows[0]["ratios"]
            }
    screen = read(ROOT / "results/accuracy_screen/accuracy.json")
    resources = {}
    for group in GROUPS:
        dim, causal = int(group.split("_")[0][1:]), group.endswith("c1")
        resources[group] = {}
        parent = screen["compile"][str(("parent", dim, causal))]["metadata"]
        for name in ["parent", "r6_dv_late", "r6_fission_v_then_k_v2"]:
            md = screen["compile"][str((name, dim, causal))]["metadata"]
            resources[group][name] = {
                k: md[k]
                for k in ["n_regs", "n_spills", "shared", "ttir_sha256", "elf_text"]
            }
            resources[group][name]["same_text_as_parent"] = (
                md["elf_text"] == parent["elf_text"]
            )
            ttir = (ROOT / "ir" / group / (name + ".ttir")).read_text()
            resources[group][name]["static_ttir_counts"] = {
                "dot": ttir.count(" = tt.dot "),
                "query_loop": ttir.count("iter_args("),
                "exp2": ttir.count("math.exp2"),
            }
    for tool in ["memcheck", "racecheck"]:
        path = ROOT / (tool + ".log")
        assert (
            "ERROR SUMMARY: 0 errors" if tool == "memcheck" else "0 errors, 0 warnings"
        ) in path.read_text()
        sources[path.name] = digest(path)
    sources["clock_run.json"] = digest(ROOT / "clock_run.json")
    for rel in [
        "optimization_plan.md",
        "clock_restored_verification.json",
        "manifest.json",
        "selection_pre_main.json",
        "../r6_generate.py",
        "../r6_run_locked.py",
        "../r6_finalize.py",
        "../r6_report.py",
    ]:
        sources[rel] = digest(ROOT / rel)
    output = dict(
        selection,
        operator="attention_dkdv",
        parent_round="r5",
        anchor_round="r2",
        final=final,
        final_subsets=subsets,
        final_rows=final_rows,
        resources=resources,
        correctness_checks=counts,
        screening=screening["screen_summary"],
        sanitizer_coverage={
            "routes": "baseline, anchor, parent, and frozen per-group finalist",
            "negative_not_sanitized": "D128 r6_fission_v_then_k_v2",
            "d64_note": "Both candidate labels are byte-identical parent controls",
        },
        failures=read(ROOT / "failures.json"),
        source_port_measured=False,
        source_expressibility=(
            "Public Triton expression order and serial output-specific loops; "
            "no source-port measurement."
        ),
        sources=sources,
        clocks={"locked_sm_mhz": 1200, "automatic_restored": True},
    )
    (ROOT / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    lines = [
        "# Attention dK/dV 第六轮 TTIR 优化",
        "",
        "沿用32个冻结shape、4warps/5stages、ABI/grid/workspace和逐位输出合同。parent为r5有效版本，anchor为第二轮有效版本。只修改TTIR，未重新调参或回写源码。",
        "",
        "## 本轮新改动",
        "",
        "`r6_dv_late`在D128的已验收64-key结构中，把概率FP16转换和dV dot移至dK dot后；第一轮相关D128负结果使用的是128-key原基线，本轮是不同结构上的新组合。",
        "",
        "`r6_fission_v_then_k_v2`在同一CTA内将每个64-key块分为dV-only遍历和dK-only遍历。"
        "先存dV，再重新计算QK/概率并计算dK。每个accumulator原有masked/unmasked顺序及dot次序不变，"
        "无额外全局工作区；无别名合同允许该store顺序。代价是重复QK、概率和部分load。"
        "SSA依赖切片删除各pass不需要的运算和循环携带状态。",
        "",
        "D64两种候选均为逐字节parent fallback，是完整矩阵控制，不计为新算法收益。"
        "初版fission因数字SSA重命名和subtile常量作用域缺陷解析失败，保留v1文件、生成器快照和pilot日志；v2修复后独立验证。",
        "",
        "## 完整矩阵初筛（B级）",
        "",
        "| 组 | 候选 | parent/候选几何均值 | 最差点 |",
        "|---|---|---:|---:|",
    ]
    for group, candidates in screening["screen_summary"].items():
        for name, metrics in candidates.items():
            lines.append(
                f"| {group} | {name} | {metrics['geo']:.6f}× | {metrics['min']:.6f}× |"
            )
    lines += [
        "",
        "## 各组正式结果（A级性能证据）",
        "",
        "| 组 | 冻结候选 | 主测parent/候选 | 复测parent/候选 | 复测最差 | 决策 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for group, rec in selection["groups"].items():
        lines.append(
            f"| {group} | {rec['tested']} | {rec['main']['parent_over']['geo']:.6f}× | "
            f"{rec['repeat']['parent_over']['geo']:.6f}× | {rec['repeat']['parent_over']['min']:.6f}× | "
            f"{'接受' if rec['accepted'] else '保留parent'} |"
        )
    lines += [
        "",
        "晋升门槛为主测和独立复测组内几何均值均≥1.02，最差点均≥0.95，并通过数值、A/A和两种sanitizer。独立复测之后不重新挑选候选。",
        "",
        "## 最终实际路线",
        "",
        "未晋升组使用本轮实际parent耗时；以下全32点均来自本轮同场配对，未相乘历史平均收益。",
        "",
        "| 比值 | 主测 | 独立复测 |",
        "|---|---:|---:|",
    ]
    for key in ["parent_over_final", "anchor_over_final", "baseline_over_final"]:
        lines.append(
            f"| {key} | {final['main'][key]['geomean']:.6f}× | {final['repeat'][key]['geomean']:.6f}× |"
        )
    lines += [
        "",
        "## 静态资源（C级）",
        "",
        "| 组 | 版本 | registers | spills | shared B | .text与parent相同 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for group in ["d128_c0", "d128_c1"]:
        for name, md in resources[group].items():
            lines.append(
                f"| {group} | {name} | {md['n_regs']} | {md['n_spills']} | "
                f"{md['shared']} | {md['same_text_as_parent']} |"
            )
    lines += [
        "",
        "资源变化不单独证明性能原因或实际occupancy。fission多做的QK/exp/load与减少共同存活状态需要一起评估。",
        "",
        "## 逐shape最终路线",
        "",
        "| (B,H,N,D,causal) | 路线 | 主测最终 μs | 主测parent/最终 | 复测parent/最终 | 复测baseline/最终 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for m, r in zip(final_rows["main"], final_rows["repeat"]):
        assert m["shape"] == r["shape"]
        lines.append(
            f"| {tuple(m['shape'])} | {m['selected_route']} | {m['median_us'][m['selected_route']]:.3f} | "
            f"{m['ratios']['parent_over_final']:.6f}× | {r['ratios']['parent_over_final']:.6f}× | "
            f"{r['ratios']['baseline_over_final']:.6f}× |"
        )
    lines += [
        "",
        "## 验证与复现",
        "",
        "官方Triton3.7.0/H100，GPU4独占且正式screen/main/repeat统一锁SM1200MHz；"
        "每路初始实际500ms、每次测量前实际100ms等条件预热，计时本体仍为原do_bench_cudagraph。"
        "主测7轮、独立seed123复测5轮，rep20为库参数，全部随机顺序/配对样本保留。"
        "预分配设备时间不含Python、分配、参考或编译，未主动清cache。",
        "",
        "全32点原source/file baseline的LLVM/PTX/cubin gate通过；"
        f"screen/main/repeat/memcheck/racecheck的检查次数为{counts}。"
        "44输入覆盖32正常输入及四组各3压力输入；N128 normal另外CPU FP64参考。"
        "所有输出按int16原始位型与baseline一致、有限性/覆盖及输入不变检查通过。"
        "memcheck0 errors，racecheck0 errors/0 warnings。clock_run.json记录finally恢复自动频率成功。",
        "sanitizer覆盖各组冻结的正式候选及baseline/anchor/parent。未选中的D128 fission完成数值与初筛，"
        "未额外运行sanitizer；D64同名fission标签只是未改parent。",
        "",
        "两个改动分别可用公开Triton表达式顺序和串行循环/多输出分开计算表达；本轮只说明结构对应关系，未做源码JIT性能或机器码一致性实测。",
        "",
        "[协议](optimization_plan.md) · [汇总](summary.json) · [冻结选择](selection.json) · "
        "[筛选](results/screen/benchmark.json) · [主测](results/main/benchmark.json) · "
        "[独立复测](results/repeat/benchmark.json) · [时钟](clock_run.json)",
        "",
        "汇总复现：`/root/venvs/ir-ops-dev/bin/python attention_dkdv/multiround/r6_report.py`。"
        "GPU完整复现由`r6_run_locked.py`执行，已有结果目录不可原地覆盖，须使用独立新实验目录。",
        "",
    ]
    (ROOT / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
