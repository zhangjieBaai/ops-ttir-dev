"""Summarize direct parent/baseline pairs and promote only an accepted finalist."""

import argparse
import json

from benchmark import bootstrap, summary
from common import ROOT, ir_path, manifest, save_json, sha256


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    candidate = args.candidate
    round_root = ROOT / "results/runs/r2_final"
    result = dict(
        candidate=candidate,
        round_protocol_sha256=sha256(ROOT / "optimization_plan_round2.md"),
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
            "r1_trim" if name.endswith("parent") else "baseline"
        )
        device_uuids.add(data["physical_uuid"])
        score = dict(**summary(rows), bootstrap_95=bootstrap(rows))
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
    assert result["accepted"], "Frozen retention criteria not met"
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
    save_json(ROOT / "results/acceptance_round2.json", result)
    selection = dict(
        candidate=candidate,
        kernel=candidate,
        reason="Fixed finalist met direct baseline AND parent paired criteria in both processes",
        parent="r1_trim",
        acceptance="results/acceptance_round2.json",
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
    save_json(ROOT / "results/selection_round2.json", selection)
    # Current selection is an intentionally mutable alias; historical round1
    # identity remains in r2_setup/parent_selection.json and old run receipts.
    (ROOT / "results/selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    lines = [
        "# logsumexp TTIR 第二轮报告",
        "",
        f"选中 `{candidate}`，以第一轮 `r1_trim` 为 parent。",
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
    lines += [
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
        "唯一计算修改在TTIR；基线/parent/候选均经官方Triton3.7.0原生文件编译入口与同一CompiledKernel启动包装。"
        "原始FlagGems源码未改，pipeline逐点核对源码TTIR、LLVM/PTX/cubin字节一致与归一化TTGIR一致。",
        "",
        "7轮主测、5轮独立进程/不同种子复测；每轮随机对内顺序，rep=20ms CUDA graph replay，"
        "预分配输出，计时前编译加载和预热。同一物理GPU，全部A/A通过，保存原始replay样本和遥测。"
        "不清缓存、不主动锁频，锁频配置未知；不含Python提交、分配、编译、加载或应用端到端。",
        "",
        "每版重置输出sentinel；FP64 CUDA oracle舍入FP16并对照native FP32 ATen，容差rtol=1e-3/atol=2e-3，"
        "包含8类压力分布、guard和输入保留检查。第二种子以及最终memcheck/racecheck覆盖全部51shape；"
        "sanitizer仅过滤logsumexp kernel，不将oracle kernel算作已检查。",
        "",
        "第一轮全部shape已经暴露，本轮20/10划分仅作调优/回归集，不是新holdout。"
        "bootstrap区间只描述冻结shape上配对轮次波动，不代表其他shape或应用总体。"
        "限定FP16、contiguous、axis=1；NaN/+inf、空输入、任意stride、其他dtype、完整FlagGems API均未验证。"
        "本地RECORD与动态组件路径/哈希核对不等于远程wheel独立来源认证。",
        "",
        "负结果、静态产物和候选选择见 results/round2_disposition.json；原始结果见 results/runs/r2_*。"
        "旧baseline和候选不覆盖；当前optimized按results/selection_round2.json绑定不可变候选，"
        "第一轮selection/acceptance归档于results/runs/r2_setup/parent_*.json。",
    ]
    with (ROOT / "report_round2.md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    print(json.dumps(result["runs"], indent=2))


if __name__ == "__main__":
    main()
