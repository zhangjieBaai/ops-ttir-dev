"""Recompute all paired evidence, freeze primary-only finalists, and report results."""

# flake8: noqa: E402 -- shared accuracy/benchmark conventions live in parent MM.
import argparse
import copy
import json
import math
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from common import frozen, save_json, sha256
from experiment_round1 import confidence

R = ROOT / "results"


def read(name):
    return json.loads((R / name).read_text())


def persist(path, value):
    if path.exists():
        assert json.loads(path.read_text()) == value, path
    else:
        save_json(path, value)


def filtered(data, family, name):
    return [r for r in data["records"] if r["family"] == family and r["name"] == name]


def valid(rows, controls):
    return all(r["valid"] for r in rows + controls)


def passes(stats):
    return (
        stats["geometric_mean"] >= 1.02
        and stats["ci95"][0] > 1
        and stats["min"] >= 0.95
    )


def freeze():
    main = read("main.json")
    result = dict(
        main_sha256=sha256(R / "main.json"),
        rule="Freeze highest full-family primary observed geomean meeting numerical performance thresholds; validity remains pending fixed retest; otherwise baseline",
        families={},
    )
    for family, config in frozen()["families"].items():
        eligible = []
        for name in ["host_tma", "host_pid"]:
            rows = filtered(main, family, name)
            if len(rows) == len(config["square_sizes"]):
                stats = confidence(rows)
                if passes(stats):
                    eligible.append((stats["geometric_mean"], name))
        result["families"][family] = max(eligible)[1] if eligible else "baseline"
    persist(R / "primary_finalists.json", result)
    print(json.dumps(result, indent=2))


def merge_repairs(data, label, repair_sizes):
    path = R / ("repair_" + label + ".json")
    if not path.exists():
        return data
    patch = json.loads(path.read_text())
    assert patch["fingerprints"] == data["fingerprints"]
    result = copy.deepcopy(data)
    replace = {(r["family"], r["name"], tuple(r["shape"])): r for r in patch["records"]}
    assert {r["shape"][0] for r in patch["records"]} == set(repair_sizes)
    for i, row in enumerate(result["records"]):
        key = (row["family"], row["name"], tuple(row["shape"]))
        if row["shape"][0] in repair_sizes:
            result["records"][i] = dict(replace[key], repair_source=path.name)
    return result


def failed_sizes():
    a, b = read("main.json"), read("repeat.json")
    sizes = sorted(
        {r["shape"][0] for data in [a, b] for r in data["records"] if not r["valid"]}
    )
    result = dict(
        main_sha256=sha256(R / "main.json"),
        repeat_sha256=sha256(R / "repeat.json"),
        sizes=sizes,
        rule="One fixed seven-round retest of union in both processes; no timing-based selection",
    )
    persist(R / "repair_plan.json", result)
    print(" ".join(map(str, sizes)))


def analyze():
    manifest = frozen()
    originals = [read("main.json"), read("repeat.json")]
    sizes = (
        read("repair_plan.json")["sizes"] if (R / "repair_plan.json").exists() else []
    )
    a, b = [
        merge_repairs(data, label, sizes)
        for data, label in zip(originals, ["main", "repeat"])
    ]
    selection = dict(
        families={},
        source_sha256={
            name: sha256(R / name)
            for name in [
                "main.json",
                "repeat.json",
                "check_v4.json",
                "repeat_check.json",
                "pipeline.json",
            ]
        },
        sync_qualified=False,
    )
    # A primary finalist is never changed to the other candidate based on repeat results.
    proposed = read("primary_finalists.json")["families"]
    tools = {}
    for tool in ["memcheck", "racecheck"]:
        log = (R / f"{tool}.log").read_text()
        summary = re.findall(r"(?:ERROR|RACECHECK) SUMMARY:.*", log)
        tools[tool] = dict(sha256=sha256(R / f"{tool}.log"), summary=summary)
    selection["tools"] = tools
    selection["tool_exit_note"] = (
        "Initial race shell wrapper swallowed tool exit; 0 is not PASS; use actual log summary of warnings. See execution_notes.md."
    )
    mem_ok = bool(
        re.search(r"ERROR SUMMARY: 0 errors", (R / "memcheck.log").read_text())
    )
    race_errors = re.search(
        r"RACECHECK SUMMARY:.*?([1-9][0-9]*) errors", (R / "racecheck.log").read_text()
    )
    numerical = [
        read(n)
        for n in [
            "check_v4.json",
            "repeat_check.json",
            "memcheck_check.json",
            "racecheck_check.json",
        ]
    ]
    for data in [a, b]:
        for row in data["records"]:
            assert math.isclose(
                row["speedup"],
                statistics.median(
                    p["baseline_us"] / p["candidate_us"] for p in row["pairs"]
                ),
                rel_tol=1e-12,
            )
    for family, config in manifest["families"].items():
        decisions = {}
        for name in ["host_tma", "host_pid"]:
            stats = {}
            accepted = mem_ok and not race_errors
            for label, data in [("main", a), ("repeat", b)]:
                rows = filtered(data, family, name)
                assert [r["shape"][0] for r in rows] == config["square_sizes"]
                values = confidence(rows)
                values["valid"] = valid(rows, filtered(data, family, "control"))
                stats[label] = values
                accepted &= passes(values) and values["valid"]
            accepted &= all(
                data["families"][family]["validation"][name]["passed"]
                for data in numerical
            )
            decisions[name] = dict(
                accepted_performance_under_warning_context=bool(accepted), stats=stats
            )
        candidate = proposed[family]
        chosen = (
            candidate
            if candidate != "baseline"
            and decisions[candidate]["accepted_performance_under_warning_context"]
            else "baseline"
        )
        path = ROOT / "ir" / family / (chosen + ".ttir")
        selection["families"][family] = dict(
            selected=chosen,
            proposed=candidate,
            decisions=decisions,
            path=str(path.relative_to(ROOT)),
            sha256=sha256(path),
        )
    for label, data in [("main", a), ("repeat", b)]:
        selection[label] = {}
        for name in ["host_tma", "host_pid"]:
            rows = [r for r in data["records"] if r["name"] == name]
            selection[label][name] = dict(
                confidence(rows), all_valid=all(r["valid"] for r in rows)
            )
        chosen_rows = []
        for family, config in manifest["families"].items():
            choice = selection["families"][family]["selected"]
            for n in config["square_sizes"]:
                if choice == "baseline":
                    chosen_rows.append(dict(speedup=1.0, pairs=[dict(speedup=1.0)]))
                else:
                    chosen_rows += [
                        r for r in filtered(data, family, choice) if r["shape"][0] == n
                    ]
        selection[label]["selected"] = dict(
            confidence(chosen_rows),
            scope="Mixed family selection; baseline fallback ratios are identity, not new paired measurements",
        )
    for name in [
        "repair_plan.json",
        "repair_main.json",
        "repair_repeat.json",
        "primary_finalists.json",
        "memcheck_check.json",
        "racecheck_check.json",
        "memcheck.log",
        "racecheck.log",
    ]:
        if (R / name).exists():
            selection["source_sha256"][name] = sha256(R / name)
    selection["analysis_sha256"] = sha256(Path(__file__))
    selection["repair_sizes"] = sizes
    selection["remaining_invalid"] = {
        label: sorted({r["shape"][0] for r in data["records"] if not r["valid"]})
        for label, data in [("main", a), ("repeat", b)]
    }
    persist(R / "selection.json", selection)
    write_report(manifest, selection, a, b)
    print(
        json.dumps(
            {k: selection[k] for k in ["main", "repeat", "remaining_invalid"]}, indent=2
        )
    )


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join(["---"] * len(headers)) + "|",
        ]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def write_report(manifest, selection, a, b):
    parts = [
        "# MM host TMA 全31点扩展",
        "本轮在同一H100 GPU7上重测全部31点，官方Triton3.7.0，FP16输入输出/FP32累加。沿用各组原BM/BN/BK/warps/stages/GROUP8/CTA1；对比具有同样host descriptor新接口的普通load基线。直接改TTIR，既有局部签名reader与原生GPU编译产物字节一致，scratch为0。旧基线/选择不覆盖。",
        "主测7轮、独立seed123复测5轮，rep20ms，配对随机顺序，预准备descriptor/output CUDA graph设备时间。无锁频/主动清cache。数值正确性先于计时；A/A偏离≤2%、baseline max/min≤1.10。同步告警保留，sync_qualified=false，结果是本限定下的B级性能观测，不表示同步资格或API端到端收益。",
    ]
    parts += [
        "## 配置组与方法消融",
        "host_tma使用原PID公式；host_pid将组内行号改为pid%size。两者均使用descriptor load/store，保持dot分块与累加顺序。组选择由主测完整组胜者确定，再要求独立复测geo≥1.02、CI下界>1、worst≥.95、所有控制有效；否则baseline回退。",
    ]
    rows = []
    for family, cfg in manifest["families"].items():
        rec = selection["families"][family]
        dims = "×".join(str(cfg["meta"]["BLOCK_SIZE_" + key]) for key in "MNK")
        stat = rec["decisions"]
        rows.append(
            [
                dims,
                len(cfg["square_sizes"]),
                f"{stat['host_tma']['stats']['main']['geometric_mean']:.4f}× / {stat['host_tma']['stats']['repeat']['geometric_mean']:.4f}×",
                f"{stat['host_pid']['stats']['main']['geometric_mean']:.4f}× / {stat['host_pid']['stats']['repeat']['geometric_mean']:.4f}×",
                rec["selected"],
            ]
        )
    parts += [
        table(
            ["BM×BN×BK", "shape数", "host TMA 主/复", "host TMA+PID 主/复", "最终选择"],
            rows,
        )
    ]
    parts += [
        "## 全31点候选时间与加速比",
        "所有时间µs；各时间列为中位数，加速比是配对比值中位数。下表保留host_pid的全部实测，即使该组被回退也不删除负值。其他候选逐点另列。",
    ]
    for name in ["host_pid", "host_tma"]:
        rows = []
        for n in range(256, 4097, 128):
            x, y = [
                next(
                    r
                    for r in data["records"]
                    if r["name"] == name and r["shape"][0] == n
                )
                for data in [a, b]
            ]
            rows.append(
                [
                    n,
                    f"{x['baseline_us']:.3f}",
                    f"{x['candidate_us']:.3f}",
                    f"{x['speedup']:.4f}×",
                    f"{y['baseline_us']:.3f}",
                    f"{y['candidate_us']:.3f}",
                    f"{y['speedup']:.4f}×",
                    str(x["valid"]) + "/" + str(y["valid"]),
                ]
            )
        parts += [
            "### " + name,
            table(
                [
                    "N=M=K",
                    "主测基线",
                    "主测候选",
                    "主测比值",
                    "复测基线",
                    "复测候选",
                    "复测比值",
                    "有效 主/复",
                ],
                rows,
            ),
        ]
    parts += [
        "## 汇总",
        "以下完整31点观测包含256的小尺寸回退，不能把只看收益子集当作全31点结果。",
    ]
    rows = []
    for name in ["host_tma", "host_pid", "selected"]:
        rows.append(
            [
                name,
                f"{selection['main'][name]['geometric_mean']:.4f}×",
                f"{selection['repeat'][name]['geometric_mean']:.4f}×",
            ]
        )
    parts += [
        table(["版本", "主测几何比", "复测几何比"], rows),
        "selected按配置组选择，回退基线的1×是身份相同，不是新增计时。无效点（主/复）："
        + str(selection["remaining_invalid"])
        + "。一次固定重测范围："
        + str(selection["repair_sizes"])
        + "。",
    ]
    parts += ["## 资源与正确性", "资源为C级事实，不单独解释性能。"]
    rows = []
    for family in manifest["families"]:
        for name, rec in a["families"][family]["resources"].items():
            rows.append(
                [
                    family,
                    name,
                    rec["n_regs"],
                    rec["n_spills"],
                    rec["metadata"]["shared"],
                ]
            )
    parts += [table(["family", "版本", "registers", "spill", "shared bytes"], rows)]
    for filename in [
        "check_v4.json",
        "repeat_check.json",
        "memcheck_check.json",
        "racecheck_check.json",
    ]:
        data = read(filename)
        count = sum(
            len(rec["cases"])
            for group in data["families"].values()
            for rec in group["validation"].values()
        )
        passed = all(
            rec["passed"]
            for group in data["families"].values()
            for rec in group["validation"].values()
        )
        parts.append(
            f"{filename}：{count}次用例（含各版本及重复），passed={passed}；参考native FP16/禁TF32 FP32及小尺寸CPU FP64，atol=.01/rtol=.001，finite、NaN sentinel、padding canary、输入不变检查。"
        )
    parts += [
        "工具实际摘要：",
        "```json\n" + json.dumps(selection["tools"], indent=2) + "\n```",
    ]
    parts += [
        "## CPU调用与原接口对照",
        "新ABI匹配基线不是原9参数接口。原接口设备计时以original_pointer标签与新ABI基线同次配对保存；walltime包含每次descriptor准备/编码/launch/同步，预分配输出，不等同于应用吞吐。以下为host_pid与原接口walltime中位数（µs），不从设备图加速推断端到端收益。",
    ]
    rows = []
    for n in range(256, 4097, 128):
        x, y = [
            data["wall_metrics"][str(n)]["median_us"] for data in originals_for_wall()
        ]
        rows.append(
            [
                n,
                f"{x['host_pid']:.2f}",
                f"{x['original_pointer']:.2f}",
                f"{y['host_pid']:.2f}",
                f"{y['original_pointer']:.2f}",
            ]
        )
    parts += [
        table(
            [
                "N",
                "主测host_pid调用",
                "主测原接口调用",
                "复测host_pid调用",
                "复测原接口调用",
            ],
            rows,
        )
    ]
    parts += [
        "## 证据",
        "[计划](plan.md)、[输入注册](inputs_v4.json)、[主测](results/main.json)、[独立复测](results/repeat.json)、[选择](results/selection.json)、[pipeline](results/pipeline.json)、[脚本](experiment.py)。旧六点结果不拼入本次统计；历史版本与失败样本保留。",
    ]
    (ROOT / "report.md").write_text("\n\n".join(parts) + "\n")


def originals_for_wall():
    return [read("main.json"), read("repeat.json")]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["freeze", "failed", "report"])
    args = parser.parse_args()
    {"freeze": freeze, "failed": failed_sizes, "report": analyze}[args.mode]()
