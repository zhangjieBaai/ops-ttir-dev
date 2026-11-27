"""Freeze finalists and qualify each sequential round from direct paired times."""

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parents[1]))
from complex_suite.support import digest, freeze, save  # noqa: E402


def measured(data, name):
    result = {}
    rng = random.Random(802)
    for label, numerator in [
        ("incremental", "parent"),
        ("anchor_relative", "anchor"),
        ("cumulative", "baseline"),
    ]:
        values = [
            statistics.median(
                s["us"][numerator] / s["us"][name] for s in row["timing"]["samples"]
            )
            for row in data["records"]
        ]
        boots = []
        for _ in range(2000):
            medians = []
            for row in data["records"]:
                samples = row["timing"]["samples"]
                medians.append(
                    statistics.median(
                        (s := rng.choice(samples))["us"][numerator] / s["us"][name]
                        for _ in samples
                    )
                )
            boots.append(math.exp(statistics.mean(math.log(v) for v in medians)))
        boots.sort()
        result[label] = {
            "geomean": math.exp(statistics.mean(math.log(v) for v in values)),
            "worst": min(values),
            "best": max(values),
            "bootstrap95": [boots[50], boots[1950]],
        }
    return result


def select(root, screen, selection_name):
    path = root / "results" / screen / "performance.json"
    data = json.loads(path.read_text())
    assert data["aa_passed"] and len(data["records"]) == 30
    candidates = data["summary"]
    eligible = [
        n for n, v in candidates.items() if v["parent_over_candidate"]["worst"] >= 0.95
    ]
    name = max(
        eligible or candidates,
        key=lambda n: candidates[n]["parent_over_candidate"]["geomean"],
    )
    record = {
        "candidate": name,
        "screen": screen,
        "screen_sha256": digest(path),
        "rule": "highest full30 incremental geo among worst>=.95; negative confirmation if no2% gain",
        "screen_metrics": candidates[name],
    }
    freeze(root / selection_name, json.dumps(record, indent=2) + "\n")
    print(name, flush=True)


def finish(root, selection_name, main_id, repeat_id, suffix):
    prior = json.loads((root / selection_name).read_text())
    run_ids = {"main": main_id, "repeat": repeat_id}
    name = prior["candidate"]
    runs = {
        label: json.loads(
            (root / f"results/{run_ids[label]}/performance.json").read_text()
        )
        for label in ("main", "repeat")
    }
    for label, d in runs.items():
        assert d["aa_passed"] and len(d["records"]) == 30
        accuracy = json.loads(
            (root / f"results/{run_ids[label]}/accuracy.json").read_text()
        )
        assert not accuracy["failed"] and name in accuracy["passed_variants"]
        assert all(x["status"] == "PASS" for x in accuracy["checks"])
    metrics = {label: measured(d, name) for label, d in runs.items()}
    accepted = all(
        m["incremental"]["geomean"] >= 1.02 and m["incremental"]["worst"] >= 0.95
        for m in metrics.values()
    )
    sanitizers = {}
    if accepted:
        for tool, marker in [
            ("memcheck", "ERROR SUMMARY: 0 errors"),
            ("racecheck", "0 errors, 0 warnings"),
        ]:
            assert marker in (root / f"{tool}{suffix}.log").read_text()
            acc = json.loads(
                (root / f"results/{tool}{suffix}/accuracy.json").read_text()
            )
            assert not acc["failed"] and name in acc["passed_variants"]
            sanitizers[tool] = "PASS"
    else:
        sanitizers = {
            "status": "not required for promotion: parent retained; candidate numerical and timing checks complete"
        }
    selected = name if accepted else "parent"
    hashes = {}
    for e in (256, 512):
        for stage in (1, 2):
            p = root / f"ir/e{e}_w4/{selected}_s{stage}.ttir"
            freeze(p.with_name(f"optimized_s{stage}.ttir"), p.read_text())
            hashes[f"e{e}/s{stage}"] = digest(p)
            for d in runs.values():
                meta = d["kernels"][f"e{e}/{selected}"][stage - 1]
                assert meta["ir_sha256"] == digest(p)
    result = {
        "round": root.name,
        "qualified": True,
        "accepted": accepted,
        "candidate_tested": name,
        "selected": selected,
        "shape_count": 30,
        "runs": run_ids,
        "main": metrics["main"],
        "repeat": metrics["repeat"],
        "selected_main": measured(runs["main"], selected),
        "selected_repeat": measured(runs["repeat"], selected),
        "selected_hashes": hashes,
        "sanitizers": sanitizers,
        "aa": {
            label: [
                min(x["timing"]["aa_median"] for x in d["records"]),
                max(x["timing"]["aa_median"] for x in d["records"]),
            ]
            for label, d in runs.items()
        },
        "evidence": {
            label: digest(root / f"results/{run_ids[label]}/performance.json")
            for label in runs
        },
    }
    save(root / "selection.json", result)
    save(root / "summary.json", result)
    report_round(root, result, runs)
    print(root.name, "accepted", accepted, metrics, flush=True)


def report_round(root, result, runs):
    name = result["candidate_tested"]
    lines = [
        f"# MoE {root.name}: 顺序TTIR优化",
        "",
        f"测试候选 `{name}`，最终{'接受候选' if result['accepted'] else '保留parent'}。",
        "所有数字来自本轮完整30点、baseline/round2 anchor/parent/candidate同轮随机交错计时。",
        "",
        "| 指标 | 主测7轮 | 独立复测5轮 |",
        "|---|---:|---:|",
    ]
    for key, label in [
        ("incremental", "parent/candidate增量"),
        ("anchor_relative", "round2 anchor/candidate"),
        ("cumulative", "原baseline/candidate累计"),
    ]:
        lines.append(
            f"| {label} | {result['main'][key]['geomean']:.6f}× | {result['repeat'][key]['geomean']:.6f}× |"
        )
    lines += [
        "",
        "不以旧轮加速比相乘代替直接测量。未接受时不把候选回退抹成1×；下一轮parent使用本轮optimized别名的实际字节。",
        "",
        "| (M,E) | baseline μs | anchor μs | parent μs | candidate μs | 主测增量 | 复测增量 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m, r in zip(runs["main"]["records"], runs["repeat"]["records"]):
        assert m["shape"] == r["shape"]
        t = m["timing"]
        lines.append(
            f"| {tuple(m['shape'])} | {t['median_us']['baseline']:.3f} | {t['median_us']['anchor']:.3f} | "
            f"{t['median_us']['parent']:.3f} | {t['median_us'][name]:.3f} | "
            f"{t['parent_over_candidate'][name]:.5f}× | {r['timing']['parent_over_candidate'][name]:.5f}× |"
        )
    lines += [
        "",
        "保持原30点、FP32/G8/top4/top8、两kernel与工作区、4warps/3stages和严格输出合同。每次主实验逐30点验证原源码baseline与文件入口LLVM/PTX/cubin一致。",
        "原六压力分布、CPU稳定索引与权重、输出覆盖/canary、输入不改均检查；本轮另外用int32视图核查FP32的原始位型，不放宽舍入。",
        "从r3独立复测起增加negative_bias和wide_bias两类有限输入，检验有序key的负值分支和大幅度ties；各次实际检查数量见accuracy.json。",
        "主测7轮、独立seed123复测5轮，rep=20参数、默认函数内10次重放均值；保存全部外层配对样本和输入/IR/产物哈希。",
        f"A/A范围：{result['aa']}。预分配完整两kernel CUDA graph设备时间，未锁频、不清cache；不含Python/分配/编译。",
        f"Sanitizer状态：{result['sanitizers']}。",
        "只有完整数值、主/复测增量geo>=1.02且最差>=.95并通过sanitizer的候选晋升；未采用的候选保持负结果。",
        f"[主测](results/{result['runs']['main']}/performance.json) · "
        f"[复测](results/{result['runs']['repeat']}/performance.json) · "
        "[选择](selection.json) · [候选](candidates.json)",
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["select", "finish"])
    p.add_argument("--round", required=True, choices=["r3", "r4", "r5"])
    p.add_argument("--screen", default="screen")
    p.add_argument("--selection", default="selection_pre_main.json")
    p.add_argument("--main", default="main")
    p.add_argument("--repeat", default="repeat")
    p.add_argument("--suffix", default="")
    a = p.parse_args()
    (
        select(BASE / a.round, a.screen, a.selection)
        if a.mode == "select"
        else finish(BASE / a.round, a.selection, a.main, a.repeat, a.suffix)
    )
