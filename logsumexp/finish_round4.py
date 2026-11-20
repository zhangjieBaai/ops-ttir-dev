"""Summarize direct parent/baseline pairs and promote only an accepted finalist."""

import argparse
import json

from audit_round3 import arithmetic_interval
from audit_round4 import FLAT_SHAPES, subset_summary
from benchmark import bootstrap, summary
from common import ROOT, ir_path, manifest, save_json, sha256


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    candidate = args.candidate
    round_root = ROOT / "results/runs/r4_final"
    result = dict(
        candidate=candidate,
        round_protocol_sha256=sha256(ROOT / "optimization_plan_round4.md"),
        evidence_level="A: full fixed matrix, paired main and independent process repeat",
        runs={},
    )
    points = {}
    device_uuids = set()
    for name in ("main_parent", "repeat_parent", "main_baseline", "repeat_baseline"):
        path = round_root / (name + ".json")
        data = read(path)
        rows = [r for r in data["records"] if r["name"] == candidate]
        assert len(rows) == 30 and data["aa_passed"]
        assert data["round_protocol_sha256"] == result["round_protocol_sha256"]
        assert data["performance_reference"] == (
            "r3_mix" if name.endswith("parent") else "baseline"
        )
        device_uuids.add(data["physical_uuid"])
        score = dict(
            flat_subset=subset_summary(rows),
            **summary(rows),
            bootstrap_95=bootstrap(rows),
            arithmetic_bootstrap_95=arithmetic_interval(rows),
        )
        result["runs"][name] = dict(
            path=str(path.relative_to(ROOT)), sha256=sha256(path), summary=score
        )
        points[name] = {tuple(row["shape"]): row for row in rows}
    assert len(device_uuids) == 1
    result["physical_uuid"] = device_uuids.pop()
    result["accepted"] = all(
        v["summary"]["geometric"] > 1.01 and v["summary"]["worst"] >= 0.90
        for v in result["runs"].values()
    )
    for name in ("main_baseline", "repeat_baseline"):
        rows = list(points[name].values())
        subset = [
            r
            for r in rows
            if manifest()["mapping"]["x".join(map(str, r["shape"]))]["partition"]
            == "validation"
        ]
        assert summary(subset)["geometric"] > 1.01
    result["target"] = dict(
        reference="original baseline",
        geometric_gt=1.10,
        arithmetic_gt=1.10,
        flat_geometric_vs_parent_gt=1.01,
        flat_shapes=FLAT_SHAPES,
    )
    result["target_met"] = all(
        result["runs"][name]["summary"][metric] > 1.10
        for name in ("main_baseline", "repeat_baseline")
        for metric in ("geometric", "arithmetic")
    )
    result["flat_target_met"] = all(
        result["runs"][name]["summary"]["flat_subset"]["geometric"] > 1.01
        for name in ("main_parent", "repeat_parent")
    )
    assert (
        result["accepted"] and result["target_met"] and result["flat_target_met"]
    ), "Round4 target not met"
    pipelines = [
        read(round_root / (name + ".json")) for name in ("pipeline", "pipeline_repeat")
    ]
    accuracies = [
        read(round_root / (name + ".json"))
        for name in ("accuracy", "accuracy_repeat", "memcheck", "racecheck")
    ]
    assert all(data["passed"] for data in [*pipelines, *accuracies])
    for data in [*pipelines, *accuracies]:
        for group, kernels in data["kernels"].items():
            assert kernels[candidate]["input_sha256"] == sha256(
                ir_path(group, candidate)
            )
    for tool in ("memcheck", "racecheck"):
        log = (round_root / (tool + ".log")).read_text()
        assert (
            "ERROR SUMMARY: 0 errors" if tool == "memcheck" else "0 errors, 0 warnings"
        ) in log
    result["checks"] = dict(
        pipeline_points=[len(d["points"]) for d in pipelines],
        accuracy_checks=[len(d["records"]) for d in accuracies],
        memcheck="0 errors",
        racecheck="0 errors, 0 warnings",
    )
    result["evidence_sha256"] = {
        str(p.relative_to(ROOT)): sha256(p) for p in round_root.glob("*.json")
    }
    save_json(ROOT / "results/acceptance_round4.json", result)
    selection = dict(
        candidate=candidate,
        kernel=candidate,
        reason="Fixed finalist met direct baseline AND parent paired criteria in both processes",
        parent="r3_mix",
        acceptance="results/acceptance_round4.json",
        groups={},
    )
    for group in manifest()["groups"]:
        path = ir_path(group, candidate)
        ir_path(group, "optimized").write_bytes(path.read_bytes())
        selection["groups"][group] = dict(
            candidate=str(path.relative_to(ROOT)),
            optimized=str(ir_path(group, "optimized").relative_to(ROOT)),
            sha256=sha256(path),
        )
    save_json(ROOT / "results/selection_round4.json", selection)
    # Current selection is an intentionally mutable alias; historical parent
    # identity remains in r4_setup/parent_selection.json and old run receipts.
    (ROOT / "results/selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    lines = [
        "# logsumexp TTIR 第四轮报告",
        "",
        f"选中 `{candidate}`，以第三轮 `r3_mix` 为 parent（第三轮对照版本），重点检查原先持平的15点。",
        "",
        "| 同进程配对参照 | 主测几何加速（95%轮次bootstrap） | 独立复测 | 主测/复测最差 |",
        "|---|---|---|---|",
    ]
    for reference in ("baseline", "parent"):
        first = result["runs"]["main_" + reference]["summary"]
        second = result["runs"]["repeat_" + reference]["summary"]
        lines.append(
            f'| {reference} | {first["geometric"]:.5f}× {first["bootstrap_95"]} | '
            f'{second["geometric"]:.5f}× {second["bootstrap_95"]} | {first["worst"]:.5f}× / {second["worst"]:.5f}× |'
        )
    flat_main = result["runs"]["main_parent"]["summary"]["flat_subset"]
    flat_repeat = result["runs"]["repeat_parent"]["summary"]["flat_subset"]
    lines += [
        "",
        f"预先冻结的15个原先持平点：相对第三轮几何平均 {flat_main['geometric']:.5f}× / "
        f"{flat_repeat['geometric']:.5f}×；该子集不代替完整30点汇总。",
    ]
    main_score = result["runs"]["main_baseline"]["summary"]
    repeat_score = result["runs"]["repeat_baseline"]["summary"]
    lines += [
        "",
        f"相对baseline（冻结的原始FlagGems实现），30点算术平均 {main_score['arithmetic']:.5f}× / "
        f"{repeat_score['arithmetic']:.5f}×；几何平均 {main_score['geometric']:.5f}× / "
        f"{repeat_score['geometric']:.5f}×，两轮点估计均达到目标。",
        f"算术平均的95%配对轮次bootstrap区间：{main_score['arithmetic_bootstrap_95']} / "
        f"{repeat_score['arithmetic_bootstrap_95']}。几何平均区间见上表；"
        "这些区间仅描述冻结shape上的轮次波动，不代表所有输入分布。",
        "",
        "## 30个冻结shape逐点结果",
        "",
        "| M,N,K | 原baseline µs | 候选 µs | 对baseline 主测/复测 | 对parent 主测/复测 |",
        "|---|---:|---:|---|---|",
    ]
    for shape, row in points["main_baseline"].items():
        ratios = [
            points[name][shape]["median_speedup"]
            for name in (
                "main_baseline",
                "repeat_baseline",
                "main_parent",
                "repeat_parent",
            )
        ]
        lines.append(
            f'| {shape} | {row["baseline_us"]:.3f} | {row["candidate_us"]:.3f} | '
            f"{ratios[0]:.4f}× / {ratios[1]:.4f}× | {ratios[2]:.4f}× / {ratios[3]:.4f}× |"
        )
    lines += [
        "",
        "## 实验边界与验证",
        "",
        "保持第一轮FP16连续布局、有限值/-inf语义、51点映射、grid、签名与完整编译选项。"
        "唯一计算修改在TTIR；基线（原始对照实现）/parent/候选均经官方Triton3.7.0原生文件编译入口与同一CompiledKernel启动包装。"
        "原始FlagGems源码未改，pipeline逐点核对源码TTIR、LLVM/PTX/cubin字节一致与归一化TTGIR一致。",
        "",
        "7轮主测、5轮独立进程/不同种子复测；每轮随机对内顺序，rep=20ms CUDA graph replay，"
        "预分配输出，计时前编译加载和预热。同一物理GPU，最终保留的主测/复测全部A/A通过，保存原始样本和遥测。"
        "不清缓存、不主动锁频，锁频配置未知；不含Python提交、分配、编译、加载或应用端到端。",
        "",
        "每版重置输出sentinel（用于发现输出漏写的预填哨兵值）；FP64 CUDA oracle舍入FP16并对照native FP32 ATen，容差rtol=1e-3/atol=2e-3，"
        "包含8类压力分布、guard和输入保留检查。第二种子以及最终memcheck/racecheck覆盖全部51shape；"
        "sanitizer仅过滤logsumexp kernel，不将oracle kernel算作已检查。",
        "",
        "第一轮全部shape已经暴露，本轮20/10划分仅作调优/回归集，不是新holdout。"
        "bootstrap区间只描述冻结shape上配对轮次波动，不代表其他shape或应用总体。"
        "限定FP16、contiguous、axis=1；NaN/+inf、空输入、任意stride、其他dtype、完整FlagGems API均未验证。"
        "本地RECORD与动态组件路径/哈希（从内容计算的校验值）核对不等于远程wheel独立来源认证。",
        "",
        "负结果、静态产物和候选选择见 results/round4_disposition.json；原始结果见 results/runs/r4_*。"
        "旧baseline和候选不覆盖；当前optimized按results/selection_round4.json绑定不可变候选，"
        "第三轮selection/acceptance归档于results/runs/r4_setup/parent_*.json。",
    ]
    wrapped = []
    for line in lines:
        while len(line) > 125 and not line.startswith(("|", "#")):
            quoted = False
            positions = []
            for index, char in enumerate(line[:126]):
                if char == "`":
                    quoted = not quoted
                if not quoted and char in "，；。 ":
                    positions.append(index + 1)
            end = next((i for i in reversed(positions) if i >= 45), 125)
            wrapped.append(line[:end].rstrip())
            line = line[end:].lstrip()
        wrapped.append(line)
    with (ROOT / "report_round4.md").open("x") as stream:
        stream.write("\n".join(wrapped) + "\n")
    print(json.dumps(result["runs"], indent=2))


if __name__ == "__main__":
    main()
