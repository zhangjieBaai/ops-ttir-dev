"""Audit retained multi-round evidence and render the Norm/Top-K reports."""

import argparse
import hashlib
import json
import math
import statistics
import struct
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PHASES = ("screen", "main", "repeat", "memcheck", "racecheck")
METRICS = ("parent_over", "anchor_over", "baseline_over")
DESCRIPTIONS = {
    "norm_quant": {
        3: "合并标量 gain；对量化路径消去公共正 rho，保留 scale 的原统计公式与零行保护。",
        4: "以 abs(u) 的非负有限浮点位型执行 uint32 max，并比较联合与分离的 sum/max 归约。",
        5: "量化输出阶段重载 u；比较全行重载、1024 元素流式输出及预声明的大行策略。",
    },
    "partial_topk": {
        3: "先在 4/8 个逻辑组中做 uint32 max/min-index，再对小组胜者做 uint64 选择。",
        4: "累积 2/4/8 项结果后批量写值和索引，保持稳定选择顺序。",
        5: "利用固定 grid、有限值和完整 tile 合同删除恒真 mask 与不可能触发的耗尽判断；另测零符号归一化后移及组合。",
    },
}


def read(path):
    return json.loads(path.read_text())


@lru_cache(None)
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (a, b)


def aggregate(values):
    return {
        "geomean": math.exp(statistics.mean(math.log(v) for v in values)),
        "worst": min(values),
        "best": max(values),
    }


@lru_cache(None)
def text_hashes(path):
    blob = path.read_bytes()
    assert blob[:6] == b"\x7fELF\x02\x01"
    offset = struct.unpack_from("<Q", blob, 40)[0]
    size, count, names_index = struct.unpack_from("<HHH", blob, 58)
    headers = [
        struct.unpack_from("<IIQQQQIIQQ", blob, offset + i * size) for i in range(count)
    ]
    names_header = headers[names_index]
    names = blob[names_header[4] : names_header[4] + names_header[5]]
    result = {}
    for header in headers:
        name = names[header[0] :].split(b"\0", 1)[0].decode()
        if name.startswith(".text"):
            contents = blob[header[4] : header[4] + header[5]]
            assert contents
            result[name] = hashlib.sha256(contents).hexdigest()
    return result


def audit_timing(data):
    names = data["validated_names"]
    for record in data["records"]:
        timing = record["timing"]
        samples = timing["samples"]
        assert len(samples) == len(timing["aa"]) == data["rounds"]
        for sample in samples:
            assert set(sample["order"]) == set(names)
            times = sample["times_us"]
            for metric in METRICS:
                numerator = metric.removesuffix("_over")
                for name in names:
                    same(sample[metric][name], times[numerator] / times[name])
        for name in names:
            same(
                timing["median_us"][name],
                statistics.median(s["times_us"][name] for s in samples),
            )
            for metric in METRICS:
                same(
                    timing[metric][name],
                    statistics.median(s[metric][name] for s in samples),
                )
        for control in timing["aa"]:
            same(control["ratio"], control["times_us"][0] / control["times_us"][1])
        same(timing["aa_median"], statistics.median(s["ratio"] for s in timing["aa"]))
        assert 0.97 <= timing["aa_median"] <= 1.03
    for name in names:
        for metric in METRICS:
            values = [r["timing"][metric][name] for r in data["records"]]
            for key, value in aggregate(values).items():
                same(value, data["summary"][name][metric][key])
    assert data["aa_passed"]


def audit(op):
    folder = ROOT / op / "multiround"
    manifest_path = ROOT / op / "round2/expanded/manifest.json"
    manifest = read(manifest_path)
    expected = [(r["group"], r["shape"]) for r in manifest["records"]]
    assert len(expected) == len({tuple(s) for _, s in expected}) == 30
    workflow = read(folder / "workflow.json")
    assert workflow["passed"] and all(c["exit_code"] == 0 for c in workflow["commands"])
    norm = op == "norm_quant"
    stages = ("kernel",) if norm else ("local", "merge")
    rounds, total_checks, total_stress = [], 0, 0
    final_data = None
    for rid in (3, 4, 5):
        rr = folder / f"r{rid}"
        selection = read(rr / "selection.json")
        pre = read(rr / "selection_pre_main.json")
        candidate = selection["candidate_tested"]
        assert pre["candidate"] == candidate
        assert pre["screen_sha256"] == sha(rr / "results/screen/results.json")
        data = {phase: read(rr / f"results/{phase}/results.json") for phase in PHASES}
        counts, aa = {}, {}
        for phase, run in data.items():
            assert run["passed"] and not run["failures"]
            assert run["manifest_sha256"] == sha(manifest_path)
            assert run["environment"]["triton"] == "3.7.0"
            assert run["environment"]["device"] == ("6" if norm else "7")
            assert [(r["group"], r["shape"]) for r in run["records"]] == expected
            assert selection["evidence"][phase] == sha(
                rr / f"results/{phase}/results.json"
            )
            assert len(run["pipeline"]) == 30 * len(stages)
            assert all(
                p["gate"]["llir_ptx_cubin_identical"]
                and p["gate"]["full_options_identical"]
                for p in run["pipeline"]
            )
            for record_index, record in enumerate(run["records"]):
                for name, kernels in record["kernels"].items():
                    for stage, meta in kernels.items():
                        path = Path(meta["ir_path"])
                        assert sha(path) == meta["ir_sha256"]
                        artifact = rr / "artifacts" / record["group"] / stage / name
                        for suffix, value in meta["stage_sha256"].items():
                            assert sha(artifact / f"kernel.{suffix}") == value
                        assert (
                            text_hashes(artifact / "kernel.cubin")
                            == meta["elf_text_sha256"]
                        )
                        assert meta["elf_text_sha256"]
                        if name in ("parent", candidate):
                            original_meta = data["main"]["records"][record_index]
                            assert (
                                meta["ir_sha256"]
                                == original_meta["kernels"][name][stage]["ir_sha256"]
                            )
                    checks = record["validation"][name]
                    assert len(checks) == (7 if norm else 8)
                    for check in checks:
                        assert check["canary_input_unchanged"]
                        if norm:
                            assert check["max_code_error"] <= 1
                        else:
                            assert check["indices_exact"] and check["values_exact"]
                        assert check["shape"][1:] == record["shape"][1:]
                        reduced = (
                            phase in ("memcheck", "racecheck")
                            or check["distribution"] != "normal"
                        )
                        rows = (
                            min(record["shape"][0], 3 if norm else 2)
                            if reduced
                            else record["shape"][0]
                        )
                        assert check["shape"][0] == rows
            counts[phase] = sum(
                len(c) for r in run["records"] for c in r["validation"].values()
            )
            total_checks += counts[phase]
            if phase in ("screen", "main", "repeat"):
                audit_timing(run)
                aa[phase] = [
                    min(r["timing"]["aa_median"] for r in run["records"]),
                    max(r["timing"]["aa_median"] for r in run["records"]),
                ]
        for phase, nrounds, seed in (("main", 7, 42), ("repeat", 5, 123)):
            assert data[phase]["rounds"] == nrounds and data[phase]["seed"] == seed
            assert data[phase]["rep_ms"] == 20
            assert selection[phase] == data[phase]["summary"][candidate]
            assert (
                selection[f"selected_{phase}"]
                == data[phase]["summary"][selection["selected"]]
            )
        accepted = all(
            data[p]["summary"][candidate]["parent_over"]["geomean"] >= 1.02
            and data[p]["summary"][candidate]["parent_over"]["worst"] >= 0.95
            for p in ("main", "repeat")
        )
        assert selection["accepted"] == accepted
        assert selection["selected"] == (candidate if accepted else "parent")
        for tool, marker in (
            ("memcheck", "ERROR SUMMARY: 0 errors"),
            ("racecheck", "0 errors, 0 warnings"),
        ):
            assert marker in (rr / f"{tool}.log").read_text()
        for item in read(rr / "candidates.json"):
            base = rr / "ir" / item["group"]
            if not norm:
                base /= item["stage"]
            assert sha(base / f"{item['name']}.ttir") == item["sha256"]
            assert sha(base / "parent.ttir") == item["parent_sha256"]
        for group, _ in expected:
            for stage in stages:
                base = rr / "ir" / group
                old = (
                    (
                        ROOT / op / "round2/expanded"
                        if rid == 3
                        else folder / f"r{rid-1}"
                    )
                    / "ir"
                    / group
                )
                if not norm:
                    base /= stage
                    old /= stage
                assert sha(base / "parent.ttir") == sha(old / "optimized.ttir")
                assert sha(base / "optimized.ttir") == sha(
                    base / f"{selection['selected']}.ttir"
                )
                matching = [
                    item
                    for item in selection["mapping"]
                    if item["group"] == group and item["stage"] == stage
                ]
                assert len(matching) == 1
                assert matching[0]["selected_sha256"] == sha(base / "optimized.ttir")
        stress_count = 0
        if norm:
            stress = read(rr / "results/extreme_stress.json")
            assert stress["passed"]
            stress_count = len(stress["records"])
            total_stress += stress_count
            for record in data["main"]["records"]:
                for name in ("baseline", "anchor", "parent", candidate):
                    matching = [
                        s
                        for s in stress["records"]
                        if s["N"] == record["shape"][1]
                        and s["name"] == name
                        and s["ir_sha256"]
                        == record["kernels"][name]["kernel"]["ir_sha256"]
                    ]
                    assert {s["mode"] for s in matching} == {
                        "max_amplitude",
                        "zero_weight",
                        "min_subnormal",
                        "cancellation",
                        "spike",
                    }
                    assert all(s["max_code_error"] <= 1 for s in matching)
        rounds.append(
            {
                **selection,
                "screen": data["screen"]["summary"],
                "checks": counts,
                "extreme_checks": stress_count,
                "aa_ranges": aa,
                "description": DESCRIPTIONS[op][rid],
            }
        )
        final_data = data
    selected = rounds[-1]["selected"]
    per_shape = []
    for main, repeat in zip(
        final_data["main"]["records"], final_data["repeat"]["records"]
    ):
        per_shape.append(
            {
                "shape": main["shape"],
                "group": main["group"],
                "main": {
                    "median_us": main["timing"]["median_us"],
                    **{m: main["timing"][m][selected] for m in METRICS},
                },
                "repeat": {
                    "median_us": repeat["timing"]["median_us"],
                    **{m: repeat["timing"][m][selected] for m in METRICS},
                },
                "kernels": main["kernels"][selected],
            }
        )
    effective = "round2/reciprocal_early" if norm else "r5/contract"
    return {
        "operator": op,
        "date": "2026-11-24",
        "shape_count": 30,
        "rounds": rounds,
        "final_effective": {
            "selected": effective,
            "timed_route": selected,
            "main": rounds[-1]["selected_main"],
            "repeat": rounds[-1]["selected_repeat"],
        },
        "per_shape": per_shape,
        "validation": {
            "version_distribution_checks": total_checks,
            "extreme_checks": total_stress,
            "total_retained_checks": total_checks + total_stress,
            "sanitizer_runs": 6,
            "sanitizers": "0 errors; racecheck 0 warnings",
            "reduced_rows": 3 if norm else 2,
        },
        "audit": {
            "passed": True,
            "raw_paired_ratios_recomputed": True,
            "all_artifact_and_nonempty_text_hashes_match": True,
            "parent_chain_matches": True,
            "manifest_sha256": sha(manifest_path),
            "workflow_sha256": sha(folder / "workflow.json"),
        },
        "source_port": "Public Triton expressible; r3-r5 source port NOT_RUN.",
    }


def render(summary):
    op = summary["operator"]
    norm = op == "norm_quant"
    title = "Norm＋动态 INT8" if norm else "大 N partial Top-K"
    final = summary["final_effective"]
    rows = [
        f"# {title}：r3–r5 顺序优化报告（2026-11-24）",
        "",
        f"固定第二轮的 30 个 shape，完成三轮不同结构的直接 TTIR 搜索。最终版本为 `{final['selected']}`。",
        (
            "三轮均未达到预先声明的晋升门槛，保留第二轮版本。"
            if norm
            else "r3、r4 未晋升；r5 的固定合同简化通过完整主测、独立复测与 sanitizer，成为最终版本。"
        ),
        "",
        "| 最终有效路径的直接配对比值 | 主测 | 独立复测 |",
        "|---|---:|---:|",
    ]
    for metric, label in (
        ("anchor_over", "第二轮 anchor / 最终版"),
        ("baseline_over", "原始 baseline / 最终版"),
    ):
        rows.append(
            f"| {label} | {final['main'][metric]['geomean']:.6f}× | {final['repeat'][metric]['geomean']:.6f}× |"
        )
    rows += [
        "",
        "这里使用 r5 同轮实际计时的最终路径，未将历史轮次的加速比相乘。保留 parent 时，anchor/parent 也来自两个实际配对计时；约 1 的波动不算新收益。",
        "",
        "## 三轮搜索与晋升",
        "",
        "| 轮次 | 确认候选 | 主测增量 | 复测增量 | 复测最差点 | 结果 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for rnd in summary["rounds"]:
        rows.append(
            f"| r{rnd['round']} | `{rnd['candidate_tested']}` | {rnd['main']['parent_over']['geomean']:.6f}× | "
            f"{rnd['repeat']['parent_over']['geomean']:.6f}× | {rnd['repeat']['parent_over']['worst']:.6f}× | "
            f"{'接受' if rnd['accepted'] else '保留 parent'} |"
        )
    rows += [
        "",
        "晋升须主测与复测都满足：30 点增量几何均值 ≥1.02，最差点 ≥0.95，数值、A/A 与两种 sanitizer 全通过。候选在完整初筛后冻结，独立复测后不重选。",
        "",
    ]
    for rnd in summary["rounds"]:
        rows += [
            f"### r{rnd['round']}",
            "",
            rnd["description"],
            "",
            "完整 30 点初筛如下；只有进入正式确认的候选有独立复测，其他候选仅为 B 级筛选证据。",
            "",
            "| 初筛候选 | parent/候选几何均值 | 最差点 |",
            "|---|---:|---:|",
        ]
        for name, metrics in rnd["screen"].items():
            if name not in ("baseline", "anchor", "parent"):
                rows.append(
                    f"| `{name}` | {metrics['parent_over']['geomean']:.6f}× | {metrics['parent_over']['worst']:.6f}× |"
                )
        rid = rnd["round"]
        rows += [
            "",
            f"证据：[冻结选择](r{rid}/selection.json)、[主测](r{rid}/results/main/results.json)、"
            f"[独立复测](r{rid}/results/repeat/results.json)。",
            "",
        ]
    if norm:
        rows += [
            "量化 gain 合并的微小收益不足以晋升；uint max、重载及缩窄输出 tile 也没有获得整体提速。"
            "大行流式策略复测最差约 0.647×，因此没有只保留获益点、隐藏退化点。",
            "",
            "原融合边界仍为 residual 加和、RMS 统计、权重乘法、动态 scale 与 INT8 量化，输出 Z/Q/S。"
            "允许 Q 相对参考 ±1 code；不能将通过本合同写成 Q 的逐位等价。",
            "",
        ]
    else:
        rows += [
            "r3 计划原文将第二轮 parent 误称为全向量 packed64，已在计划末尾保留回顾性更正。"
            "实际 parent 使用全向量 uint32 max 加 i32 min-index；r3 再将这两次归约分组，"
            "只对组胜者引入 uint64 packed 归约。实验代码和数值不受这处文字更正影响。",
            "",
            "r5 证明条件来自本次冻结矩阵：完整 local tile 含 1024 个有限值，K≤16；merge 有至少 K 个有效有限候选。"
            "仅完整且对应 merge 长度为二次幂的映射移除 load mask；N=65537 的尾块保留保护和耗尽哨兵。"
            "merge 的第二维 grid 恒为 1，可把该维 program_id 替换为 0。",
            "",
            "r5 采用的简化是对现有合同特化，不适用于任意新 N、NaN/Inf 或每块有效值不足 K 的输入。"
            "输出按数值精确比较，稳定索引精确；+0/-0 等价，不承诺保留输入零的位型。",
            "",
        ]
    rows += ["## 能否用公开 Triton 表达", ""]
    if norm:
        rows += [
            "可以。gain 合并可写成标量 `tl.div_rn`/乘法，整数 max 可用 `to(tl.uint32, bitcast=True)` 与 `tl.max`，"
            "联合统计可用 `tl.reduce` 的 tuple combiner；重载和流式量化可用 `tl.load`、循环与 `tl.store`。"
            "这些结构具有源码表达方式，但 r3–r5 没有执行源码回写性能实验，不能声称源码已经复现本轮机器码。",
            "",
        ]
    else:
        rows += [
            "可以。分组归约可用 `reshape`、`tl.max`/`tl.min` 和 uint64 key；批量输出可用 `join`/`trans`/`reshape` 后 `tl.store`。"
            "r5 可用 `tl.constexpr` 分支只在已证明完整的映射省略 mask，按固定 grid 使用 chunk=0，并删除不可能触发的哨兵分支。"
            "零规范化可用整数 bitcast、按位运算和 `tl.where`。本轮只实测 TTIR，尚未实测 r3–r5 源码回写的性能或 `.text` 一致性。",
            "",
        ]
    v = summary["validation"]
    rows += [
        "## 验证、计时与范围",
        "",
        f"环境为官方 Triton 3.7.0、PyTorch 2.12.0+cu130、H100 80GB，独占 GPU{'6' if norm else '7'}，自动频率。"
        "保持原 grid、编译配置、阶段数和工作区，未重新 autotune。正式主测与复测使用 baseline/第二轮 anchor/当前 parent/候选四路同输入随机交错计时。"
        "主测 7 轮 seed42、独立复测 5 轮 seed123，每轮 `do_bench_cudagraph(rep=20ms)`；初筛 3 轮 rep10ms。"
        "输入分配前设置 shape 派生 seed 并记录输入哈希。",
        "",
        "每轮完整矩阵的源码 baseline→TTIR 文件入口 LLVM/PTX/cubin 一致；这项一致性属于 baseline pipeline，"
        "不等于新候选已有源码回写验证。所有阶段产物、非空 ELF `.text` 和最终选择哈希均已重新核对。",
        "",
        f"累计保留 {v['version_distribution_checks']} 项版本×输入分布检查"
        f"{'，另有 ' + str(v['extreme_checks']) + ' 项 CPU FP64 极端检查' if norm else ''}；"
        "三轮各一次 memcheck 与 racecheck，共六次，均 0 errors、racecheck 0 warnings。"
        "这些是重复执行的检查记录，不是同等数量的独立 shape。",
        "",
        f"数值覆盖需区分行数：normal 分布在初筛/主测/复测使用原完整 M；其他压力分布使用 min(M,{v['reduced_rows']}) 行。"
        f"sanitizer 的所有分布使用 min(M,{v['reduced_rows']}) 行，覆盖全部 30 个冻结 kernel 映射及其 N"
        f"{'/K' if not norm else ''}，没有在 sanitizer 下运行每个原始完整 M。",
        "",
    ]
    if norm:
        rows += [
            "核对 Z 精确输出、scale 容差、Q∈[-127,127]、code 差≤1、重构误差、canary 与输入未改写。"
            "额外极端检查按 N 与候选哈希去重，每种实现使用 2 行，覆盖大幅值、零权重、最小 subnormal、抵消和尖峰。",
            "",
        ]
    else:
        rows += [
            "覆盖 normal、全相等、整数 ties、±0、单调输入、正/负 subnormal、最低有限值；"
            "稳定索引、数值输出、canary 与输入未改写全部通过。小行输入另对 CPU FP64 stable sort 检查。",
            "",
        ]
    rows += ["| 轮次 | 主测 A/A 中位比范围 | 复测 A/A 中位比范围 |", "|---|---|---|"]
    for rnd in summary["rounds"]:
        a, b = rnd["aa_ranges"]["main"], rnd["aa_ranges"]["repeat"]
        rows.append(
            f"| r{rnd['round']} | {a[0]:.6f}–{a[1]:.6f} | {b[0]:.6f}–{b[1]:.6f} |"
        )
    rows += [
        "",
        "全部点在预设 [0.97,1.03] 门槛内。以上为固定矩阵 A 级实测结论，不能外推到其他硬件或形状，也不能由资源变化独立证明性能因果。",
        "",
        "## 最终 30 点独立复测",
        "",
        "耗时是各路径的样本中位数；加速比是同轮配对比值的中位数，因此不必等于表中两个耗时中位数直接相除。",
        "",
        "| shape | baseline µs | anchor µs | 最终 µs | anchor/最终 | baseline/最终 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["per_shape"]:
        values = row["repeat"]
        times = values["median_us"]
        shape = "×".join(str(n) for n in row["shape"])
        rows.append(
            f"| {shape} | {times['baseline']:.4f} | {times['anchor']:.4f} | "
            f"{times[final['timed_route']]:.4f} | {values['anchor_over']:.6f}× | {values['baseline_over']:.6f}× |"
        )
    rows += [
        "",
        "## 文件与复现",
        "",
        "- [机器可读汇总与核查结果](summary.json)",
        "- [预声明计划](plan.md)、[候选生成器](generate.py)、[顺序执行记录](workflow.json)",
        "- [计时入口](../../complex_suite/multiround_runner.py)、[顺序控制器](../../complex_suite/multiround_driver.py)",
        "- 每轮 `ir/<group>/optimized.ttir` 保存该轮接受的有效版本；失败晋升不覆盖旧 parent。",
        "",
        "重算报告（只读核查既有证据，不重新计时）：",
        "",
        "```bash",
        "cd /root/ops-ttir-dev",
        f"/root/venvs/ir-ops-dev/bin/python norm_quant/multiround/report.py --op {op}",
        "```",
        "",
        "GPU 原始命令与完整参数保存在 workflow.json；已有结果目录不可覆盖，重跑需使用新结果 ID。",
        "",
    ]
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", choices=("norm_quant", "partial_topk"), required=True)
    args = parser.parse_args()
    summary = audit(args.op)
    folder = ROOT / args.op / "multiround"
    (folder / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    (folder / "report.md").write_text(render(summary))
    print(args.op, "audit PASS", summary["final_effective"], summary["validation"])


if __name__ == "__main__":
    main()
