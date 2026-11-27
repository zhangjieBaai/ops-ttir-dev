"""Audit r6 Norm/Top-K raw evidence and generate reproducible reports."""

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "prior_audit", ROOT / "norm_quant/multiround/report.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def main(op):
    norm = op == "norm_quant"
    folder = ROOT / op / "multiround/r6"
    selection = audit.read(folder / "selection.json")
    pre = audit.read(folder / "selection_pre_main.json")
    manifest_path = ROOT / op / "round2/expanded/manifest.json"
    manifest = audit.read(manifest_path)
    expected = [(x["group"], x["shape"]) for x in manifest["records"]]
    assert len(expected) == len({tuple(s) for _, s in expected}) == 30
    candidate, selected = selection["candidate_tested"], selection["selected"]
    assert pre["candidate"] == candidate
    screen_run = selection["screen_run"]
    assert pre["screen_sha256"] == audit.sha(
        folder / f"results/{screen_run}/results.json"
    )
    phases = [screen_run, "main", "repeat", "memcheck", "racecheck"]
    data = {p: audit.read(folder / f"results/{p}/results.json") for p in phases}
    assert audit.read(folder / "workflow.json")["passed"]
    counts, aa = {}, {}
    for phase, run in data.items():
        assert run["passed"] and not run["failures"]
        assert run["manifest_sha256"] == audit.sha(manifest_path)
        assert [(r["group"], r["shape"]) for r in run["records"]] == expected
        assert len(run["pipeline"]) == (30 if norm else 60)
        assert all(
            p["gate"]["llir_ptx_cubin_identical"]
            and p["gate"]["full_options_identical"]
            for p in run["pipeline"]
        )
        if phase != screen_run:
            assert selection["evidence"][phase] == audit.sha(
                folder / f"results/{phase}/results.json"
            )
        for record in run["records"]:
            for name, kernels in record["kernels"].items():
                for stage, meta in kernels.items():
                    assert audit.sha(Path(meta["ir_path"])) == meta["ir_sha256"]
                    artifact = folder / "artifacts" / record["group"] / stage / name
                    assert (
                        audit.text_hashes(artifact / "kernel.cubin")
                        == meta["elf_text_sha256"]
                    )
                    assert meta["elf_text_sha256"]
                    for suffix, digest in meta["stage_sha256"].items():
                        assert audit.sha(artifact / f"kernel.{suffix}") == digest
                    if name == "parent":
                        old = folder.parent / "r5/ir" / record["group"]
                        if not norm:
                            old /= stage
                        assert audit.sha(old / "optimized.ttir") == meta["ir_sha256"]
                checks = record["validation"][name]
                assert len(checks) == (7 if norm else 8)
                for check in checks:
                    assert check["canary_input_unchanged"]
                    if norm:
                        assert check["max_code_error"] <= 1
                    else:
                        assert check["indices_exact"] and check["values_exact"]
                    reduced = (
                        phase in ("memcheck", "racecheck")
                        or check["distribution"] != "normal"
                    )
                    rows = (
                        min(record["shape"][0], 3 if norm else 2)
                        if reduced
                        else record["shape"][0]
                    )
                    assert check["shape"] == [rows, *record["shape"][1:]]
        counts[phase] = sum(
            len(c) for r in run["records"] for c in r["validation"].values()
        )
        if phase not in ("memcheck", "racecheck"):
            audit.audit_timing(run)
            aa[phase] = [
                min(r["timing"]["aa_median"] for r in run["records"]),
                max(r["timing"]["aa_median"] for r in run["records"]),
            ]
    for phase, rounds, seed in (("main", 7, 42), ("repeat", 5, 123)):
        run = data[phase]
        assert (run["rounds"], run["rep_ms"], run["seed"]) == (rounds, 20, seed)
        assert selection[phase] == run["summary"][candidate]
        assert selection[f"selected_{phase}"] == run["summary"][selected]
    passed_gate = all(
        data[p]["summary"][candidate]["parent_over"]["geomean"] >= 1.02
        and data[p]["summary"][candidate]["parent_over"]["worst"] >= 0.95
        for p in ("main", "repeat")
    )
    assert selection["accepted"] == passed_gate
    assert selected == (candidate if passed_gate else "parent")
    for item in selection["mapping"]:
        directory = folder / "ir" / item["group"]
        if not norm:
            directory /= item["stage"]
        assert (
            audit.sha(directory / "optimized.ttir")
            == audit.sha(directory / f"{selected}.ttir")
            == item["selected_sha256"]
        )
    assert "ERROR SUMMARY: 0 errors" in (folder / "memcheck.log").read_text()
    assert "0 errors, 0 warnings" in (folder / "racecheck.log").read_text()
    extreme = 0
    if norm:
        stress = audit.read(folder / "results/extreme_stress.json")
        assert stress["passed"]
        extreme = len(stress["records"])
        for record in data["main"]["records"]:
            for name in data["main"]["validated_names"]:
                matches = [
                    s
                    for s in stress["records"]
                    if s["N"] == record["shape"][1]
                    and s["name"] == name
                    and s["ir_sha256"] == record["kernels"][name]["kernel"]["ir_sha256"]
                ]
                assert len(matches) == 5 and all(
                    s["max_code_error"] <= 1 for s in matches
                )
    static = {}
    screen = data[screen_run]
    for name in screen["validated_names"]:
        if name in ("baseline", "anchor", "parent"):
            continue
        static[name] = (
            {
                stage: sum(
                    r["kernels"][name][stage]["elf_text_sha256"]
                    == r["kernels"]["parent"][stage]["elf_text_sha256"]
                    for r in screen["records"]
                )
                for stage in ("kernel",)
                if norm
            }
            if norm
            else {
                stage: sum(
                    r["kernels"][name][stage]["elf_text_sha256"]
                    == r["kernels"]["parent"][stage]["elf_text_sha256"]
                    for r in screen["records"]
                )
                for stage in ("local", "merge")
            }
        )
    result = {
        "operator": op,
        "round": 6,
        "shape_count": 30,
        **selection,
        "final_effective": {
            "timed_route": selected,
            "main": selection["selected_main"],
            "repeat": selection["selected_repeat"],
        },
        "screen": screen["summary"],
        "static_identity_counts": static,
        "validation": {
            "checks_by_phase": counts,
            "extreme_checks": extreme,
            "sanitizer_reduced_rows": 3 if norm else 2,
        },
        "aa_ranges": aa,
        "audit_passed": True,
        "source_port_measured": False,
        "sources": {p: audit.sha(folder / f"results/{p}/results.json") for p in phases},
    }
    (folder / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    title = "Norm＋动态INT8" if norm else "大N partial Top-K"
    rows = [
        f"# {title}：第六轮TTIR优化",
        "",
        f"冻结30点、原配置与数值合同；从r5最终有效版本出发。正式候选`{candidate}`，"
        + ("通过门槛并晋升。" if passed_gate else "未达到晋升门槛，保留parent。"),
        "",
        "| 指标 | 主测 | 独立复测 |",
        "|---|---:|---:|",
    ]
    for label, field in (
        ("新候选相对r5 parent", "parent_over"),
        ("最终版相对第二轮anchor", "anchor_over"),
        ("最终版相对原始baseline", "baseline_over"),
    ):
        target = candidate if field == "parent_over" else selected
        rows.append(
            f"| {label} | {data['main']['summary'][target][field]['geomean']:.6f}× | "
            f"{data['repeat']['summary'][target][field]['geomean']:.6f}× |"
        )
    rows += [
        "",
        "全部比值由本轮同输入随机交错的实际路径计时直接得到；最终保留parent时使用实测parent耗时，没有相乘历史比值。",
        "",
        "## 候选与完整矩阵初筛",
        "",
        "| 候选 | parent/候选几何均值 | 最差点 | 与parent相同的可执行.text映射数 |",
        "|---|---:|---:|---|",
    ]
    for name, identities in static.items():
        stats = screen["summary"][name]["parent_over"]
        identity_text = ", ".join(
            f"{stage}: {count}/30" for stage, count in identities.items()
        )
        rows.append(
            f"| `{name}` | {stats['geomean']:.6f}× | {stats['worst']:.6f}× | {identity_text} |"
        )
    if norm:
        rows += [
            "",
            "`convert_rn`保留浮点clamp，将nearbyintf后转i8改为公开libdevice float2int_rn后截断i8；"
            "`integer_clamp`进一步在舍入成i32后clamp到[-127,127]。"
            "沿用有限FP16输入、abs(X/R)≤10000、abs(W)≤8、N≤16384的合同，量化值约为[-127,127]，远离i32溢出。"
            "`fold4`将sum/max联合统计分成4组，先组内再组间归约，改变sum顺序但沿用原误差合同。",
            "",
            "三种结构都可用公开Triton表达：`tl.extra.cuda.libdevice.float2int_rn`、整数min/max、reshape与tuple reduce。"
            "使用libdevice有CUDA后端依赖；本轮没有源码回写性能或指令等价测试。",
        ]
    else:
        rows += [
            "",
            "local保持r5。merge利用每个chunk的K项已经排序这一前提，只持有每chunk当前头部key/id/cursor。"
            "每次选出全局最大头部、输出后推进其所属chunk，再加载该chunk下一项；比较uint32 max+min-id与packed uint64稳定max两条路线。"
            "减少merge逻辑宽度，同时引入依赖链和小规模后续load；整体收益必须由完整两kernel序列确认。",
            "",
            "正确性依赖局部列表有序：每个列表未消费元素不大于当前头部，因此最大头部就是全局下一项；"
            "相同值按原始id升序，局部次序与全局次序一致。全局前K项不可能需要任何单列表第K+1项。"
            "N=65537对应65列表，逻辑128头部中的padding仍mask，局部不足K项由-inf/INT_MAX填充；"
            "全局至少K个有限值，所以这些padding不会被输出。±0仍按数值等价合同处理。",
            "",
            "packed v1因向量pack与标量解码重名而解析失败，保留原文件和screen记录；v2只修复SSA名。"
            "最终冻结选择来自完整screen_v2，旧screen计时不用于选拔或收益声明。",
            "",
            "头部归并可用公开Triton的arange、带mask的load、where、uint32/uint64归约及static_range表达；"
            "没有源码回写实测。支持的是本实验生成的有序中间列表，不能用于任意未排序workspace。",
        ]
    rows += [
        "",
        "## 验证与范围",
        "",
        "官方Triton3.7.0/H100；保持原ABI、grid、配置、kernel数和workspace，没有autotune。"
        "源码baseline与文件入口LLVM/PTX/cubin逐30映射一致。全部候选先精度后性能；初筛3轮rep10、主测7轮rep20、"
        "独立seed123复测5轮rep20。每点A/A中位必须在[0.97,1.03]；主/复增量geo≥1.02且最差≥0.95才能晋升。",
        "",
        f"合格初筛、主测、复测、两种sanitizer共保留{sum(counts.values())}项版本×分布检查，额外极端检查{extreme}项。"
        "memcheck零错误，racecheck零错误/零警告。",
        f"normal在性能实验前验证完整M，其他压力分布及sanitizer使用min(M,{3 if norm else 2})行，覆盖全部30映射及N/K；"
        "不将缩减行数描述为全部原始M都已在sanitizer下执行。哈希、非空ELF.text、父版本链和原始配对比值均经报告生成器核对。",
        "",
        "| 运行 | A/A中位最小 | A/A中位最大 |",
        "|---|---:|---:|",
    ]
    for phase, limits in aa.items():
        rows.append(f"| {phase} | {limits[0]:.6f} | {limits[1]:.6f} |")
    rows += [
        "",
        "## 30点最终路线独立复测",
        "",
        "耗时为样本中位数；加速比为配对比值中位数，不必等于两个耗时中位数相除。",
        "",
        "| shape | baseline µs | 最终 µs | 相对第二轮 | 相对原始baseline |",
        "|---|---:|---:|---:|---:|",
    ]
    for record in data["repeat"]["records"]:
        t = record["timing"]
        shape = "×".join(map(str, record["shape"]))
        rows.append(
            f"| {shape} | {t['median_us']['baseline']:.4f} | {t['median_us'][selected]:.4f} | "
            f"{t['anchor_over'][selected]:.6f}× | {t['baseline_over'][selected]:.6f}× |"
        )
    rows += [
        "",
        "[全局预声明计划](../../../complex_suite/round6_plan.md) · [冻结选择](selection.json) · [机器可读汇总](summary.json)",
        "",
        f"生成报告：`/root/venvs/ir-ops-dev/bin/python complex_suite/round6_norm_topk_report.py --op {op}`。"
        "复现GPU命令及参数在本目录workflow.json；必须使用新的结果ID并隔离指定GPU。",
        "",
    ]
    (folder / "report.md").write_text("\n".join(rows))
    print(op, "AUDIT PASS", selection["accepted"], selection["selected_repeat"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", choices=("norm_quant", "partial_topk"), required=True)
    main(parser.parse_args().op)
