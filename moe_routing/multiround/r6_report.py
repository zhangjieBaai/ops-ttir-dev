"""Audit immutable r6 evidence and add mechanism/scope to the qualified report."""

import hashlib
import json
import math
import statistics
import struct
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE / "r6"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def executable_hashes(blob):
    assert blob[:6] == b"\x7fELF\x02\x01"
    off = struct.unpack_from("<Q", blob, 40)[0]
    size, count, ni = struct.unpack_from("<HHH", blob, 58)
    headers = [
        struct.unpack_from("<IIQQQQIIQQ", blob, off + i * size) for i in range(count)
    ]
    n = headers[ni]
    names = blob[n[4] : n[4] + n[5]]
    results = {}
    for item in headers:
        name = names[item[0] :].split(b"\0", 1)[0].decode()
        if name.startswith(".text"):
            code = blob[item[4] : item[4] + item[5]]
            assert code
            results[name] = hashlib.sha256(code).hexdigest()
    assert results
    return results


def main():
    selected = read(ROOT / "selection.json")
    assert selected["qualified"]
    workflow = read(ROOT / "workflow.json")
    assert workflow["completed"] and all(
        x["exit_code"] == 0 for x in workflow["commands"]
    )
    names = read(ROOT / "lineage.json")["candidates"]
    phases = ["screen", "main", "repeat"]
    if selected["accepted"]:
        phases += ["memcheck", "racecheck"]
    checks = 0
    mapping = read(ROOT / "mapping_proof.json")
    assert mapping["passed"] and len(mapping["records"]) == 90
    hashes = {"mapping_proof.json": digest(ROOT / "mapping_proof.json")}
    for phase in phases:
        folder = ROOT / "results" / phase
        accuracy = read(folder / "accuracy.json")
        assert not accuracy["failed"]
        assert all(row["status"] == "PASS" for row in accuracy["checks"])
        assert all(
            all(row["float32_bits_equal"].values()) for row in accuracy["checks"]
        )
        checks += len(accuracy["checks"])
        pipeline = read(folder / "pipeline.json")
        assert len(pipeline) == 30
        assert all(
            row["outputs_equal"]
            and all(
                s["llir_ptx_cubin_identical"] and s["full_options_identical"]
                for s in row["stages"]
            )
            for row in pipeline
        )
        compiled = read(folder / "compile.json")
        assert not compiled["errors"]
        for key, stages in compiled["metadata"].items():
            e, name = key.split("/")
            for stage, info in enumerate(stages, 1):
                assert digest(Path(info["ir_path"])) == info["ir_sha256"]
                artifact = ROOT / "artifacts" / f"{e}_w4" / name / f"s{stage}"
                for level, expected in info["stage_sha256"].items():
                    assert digest(artifact / f"kernel.{level}") == expected
                assert (
                    executable_hashes((artifact / "kernel.cubin").read_bytes())
                    == info["text_sha256"]
                )
                if stage == 2:
                    parent = compiled["metadata"][f"{e}/parent"][1]
                    assert info["ir_sha256"] == parent["ir_sha256"] or name in (
                        "baseline",
                        "anchor",
                    )
        for name in ("accuracy", "compile", "pipeline"):
            hashes[f"{phase}/{name}.json"] = digest(folder / f"{name}.json")
        if phase in ("main", "repeat"):
            performance = read(folder / "performance.json")
            assert performance["aa_passed"] and len(performance["records"]) == 30
            for label, numerator in [
                ("incremental", "parent"),
                ("anchor_relative", "anchor"),
                ("cumulative", "baseline"),
            ]:
                for output, route in [
                    (phase, selected["candidate_tested"]),
                    (f"selected_{phase}", selected["selected"]),
                ]:
                    vals = [
                        statistics.median(
                            s["us"][numerator] / s["us"][route]
                            for s in row["timing"]["samples"]
                        )
                        for row in performance["records"]
                    ]
                    geo = math.exp(statistics.mean(math.log(v) for v in vals))
                    assert abs(geo - selected[output][label]["geomean"]) < 1e-12
            for row in performance["records"]:
                ratios = [x["us"][0] / x["us"][1] for x in row["timing"]["aa"]]
                assert 0.97 <= statistics.median(ratios) <= 1.03
            hashes[f"{phase}/performance.json"] = digest(folder / "performance.json")
    for e in (256, 512):
        for stage in (1, 2):
            prior = BASE / f"r5/ir/e{e}_w4/optimized_s{stage}.ttir"
            current = ROOT / f"ir/e{e}_w4/parent_s{stage}.ttir"
            assert prior.read_bytes() == current.read_bytes()
            optimized = current.with_name(f"optimized_s{stage}.ttir")
            assert digest(optimized) == selected["selected_hashes"][f"e{e}/s{stage}"]
    selected.update(
        parent="r5/pair_integer_constants",
        anchor="round2/r2_rank_ordered_gather",
        source_port_measured=False,
        validation_checks=checks,
        audit={"passed": True, "pipeline_shapes_per_run": 30, "hashes": hashes},
    )
    (ROOT / "summary.json").write_text(json.dumps(selected, indent=2) + "\n")
    core = ROOT / "report_core.md"
    if not core.exists():
        core.write_text((ROOT / "report.md").read_text())
    screen = read(ROOT / "results/screen/performance.json")
    compiled = read(ROOT / "results/screen/compile.json")["metadata"]
    lines = [
        "# MoE r6：跨组批量执行stage1 top2",
        "",
        "继续固定30点，parent为r5已验收的`pair_integer_constants`。本轮只改stage1的组分配和二维数据流，stage2与parent逐字节一致。",
        "",
        f"最终{'接受' if selected['accepted'] else '未晋升'} `{selected['candidate_tested']}`。",
        "",
        "| 最终实际路线 | 主测 | 独立复测 |",
        "|---|---:|---:|",
    ]
    for key, label in [
        ("incremental", "相对r5 parent"),
        ("anchor_relative", "相对r2 anchor"),
        ("cumulative", "相对原始baseline"),
    ]:
        lines.append(
            f"| {label} | {selected['selected_main'][key]['geomean']:.6f}× | "
            f"{selected['selected_repeat'][key]['geomean']:.6f}× |"
        )
    lines += [
        "",
        "所有比值是本轮同场直接计时，不与历史平均相乘。晋升要求两次增量几何均值≥1.02、最差点≥0.95并通过数值/A-A/sanitizer。",
        "",
        "## 三个预声明候选",
        "",
        "| 候选 | 每个活动CTA计算组数 | 初筛增量几何均值 | 最差点 |",
        "|---|---:|---:|---:|",
    ]
    for batch, name in zip((2, 4, 8), names):
        values = screen["summary"][name]["parent_over_candidate"]
        lines.append(
            f"| {name} | {batch} | {values['geomean']:.6f}× | {values['worst']:.6f}× |"
        )
    lines += [
        "",
        "初筛为B级；冻结正式候选后才进行完整主测及独立复测。保留原stage1的`M*8` grid，"
        "group id不能被batch整除的CTA直接返回；活动CTA处理相邻batch组。每个输出组恰好写一次，"
        "没有改变host launch或增加kernel/workspace。batch4时每token仍调度8个CTA，其中2个计算、6个返回。",
        "",
        "活动CTA使用`[batch, experts_per_group]`张量，sigmoid、bias加法、top2组合器和最终加法均保持原表达；"
        "归约从一维改为沿axis1分别计算各组。这探索warp利用率与空CTA调度的取舍，"
        "不将逻辑工作分配直接等同实际occupancy或动态性能原因。",
        "",
        "## 静态资源",
        "",
        "| E | stage1 | regs | spills | shared B |",
        "|---:|---|---:|---:|---:|",
    ]
    for e in (256, 512):
        for name in ("parent", *names):
            meta = compiled[f"e{e}/{name}"][0]
            lines.append(
                f"| {e} | {name} | {meta['n_regs']} | {meta['n_spills']} | {meta['metadata']['shared']} |"
            )
    lines += [
        "",
        "以上为C级编译产物事实，未采集NCU动态计数，不以资源值单独证明加速原因。",
        "",
        "## 验证与源码表达",
        "",
        f"累计{checks}项版本×分布数值检查均通过，八类分布使用完整冻结M，"
        "包括normal/ties/group_ties/bias_dominant/saturated/negative/negative_bias/wide_bias。"
        "group scores与weights按FP32原始位型检查，indices精确，另有CPU参考、canary和输入未改检查。",
        "",
        "每次正式运行逐30点检查原源码baseline与TTIR文件入口LLVM/PTX/cubin及完整编译options一致；"
        "所有实际IR、各阶段产物、选择链与配对比值已重新核对。GPU2独占、自动频率；"
        "其余环境/预热/rep预算与r3–r5相同。",
        "",
        f"Sanitizer：{selected['sanitizers']}。完整原始记录见results各运行目录和对应日志。",
        "",
        "公开Triton可用program_id、统一分支、二维arange/broadcast和tuple tl.reduce表达这一结构。"
        "本轮没有源码回写计时或源码/候选`.text`等价实验。",
        "",
        "[预声明协议](optimization_plan.md) · [机器可读汇总](summary.json) · [候选谱系](lineage.json)",
        "[输出组覆盖枚举](mapping_proof.json) · [静态资源与指令计数](static_analysis.json)",
        "",
        "复现：在空闲GPU2使用官方venv，运行`r6_generate.py`再运行`r6_run.py --round r6 --run NEW_ID`。"
        "已有结果目录不可覆盖；正式预算、选拔/验收命令详见`workflow.json`及`r6_workflow.py`。",
        "",
        core.read_text().replace("# MoE r6: 顺序TTIR优化", "## 完整30点正式结果"),
    ]
    (ROOT / "report.md").write_text("\n".join(lines) + "\n")
    print("AUDIT PASS", checks, selected["selected"], flush=True)


if __name__ == "__main__":
    main()
