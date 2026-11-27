"""Audit both full runs and generate the final-route ablation report and CSV."""

import argparse
import csv
import hashlib
import io
import json
import os
import statistics as st
from pathlib import Path

OP = Path(__file__).resolve().parent
ROOT = OP / "ablation_v1"
CHAIN = [
    "baseline",
    "add_membership",
    "add_ordered",
    "add_gather",
    "add_rank",
    "add_pair",
    "add_constants",
    "final",
]
LABELS = {
    "baseline": "原始 baseline",
    "add_membership": "组选择广播/reshape",
    "add_ordered": "再加 uint32 有序 key",
    "add_gather": "再加循环后 gather",
    "add_rank": "再加 8×8 稳定选组排名",
    "add_pair": "再加 stage1 top2 双值归约",
    "add_constants": "再加两阶段固定整数特化",
    "final": "最终：再加 stage1 batch4",
    "drop_membership": "最终去掉 membership 广播",
    "drop_ordered": "最终去掉有序 key",
    "drop_gather": "最终去掉循环后 gather",
    "drop_rank": "最终去掉稳定排名",
    "drop_pair": "最终去掉双值归约（保留 batch4）",
    "drop_constants": "最终去掉整数特化（保留 tile/batch）",
    "stage1_only": "只替换最终 stage1",
    "stage2_only": "只替换最终 stage2",
}
REMOVALS = {
    n: n
    for n in (
        "drop_membership",
        "drop_ordered",
        "drop_gather",
        "drop_rank",
        "drop_pair",
        "drop_constants",
    )
}
REMOVALS["drop_batch"] = "add_constants"


def read(p):
    return json.loads(p.read_text())


def relative(p):
    return os.path.relpath(p, OP)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def audit(path, seed, rounds):
    p = read(path / "performance.json")
    checks = read(path / "accuracy.json")
    pipeline = read(path / "pipeline.json")
    meta = read(path / "compile.json")
    manifest = read(ROOT / "manifest.json")
    points = [
        (x["tokens"], x["experts"]) for x in read(ROOT / "manifest30.json")["points"]
    ]
    names = manifest["variants"]
    assert p["manifest_sha256"] == sha(ROOT / "manifest.json")
    assert p["shape_manifest_sha256"] == sha(ROOT / "manifest30.json")
    assert p["names"] == names and p["seed"] == seed and p["rounds"] == rounds
    assert p["rep_ms"] == 10
    assert [tuple(x["shape"]) for x in p["records"]] == points
    assert [tuple(x["shape"]) for x in pipeline] == points
    assert all(
        x["outputs_equal"]
        and all(
            s["llir_ptx_cubin_identical"] and s["full_options_identical"]
            for s in x["stages"]
        )
        for x in pipeline
    )
    source_options = pipeline[0]["stages"][0]["options"]
    for stages in meta.values():
        for stage in stages:
            assert {k: stage["metadata"][k] for k in source_options} == source_options
    original = read(path.parent / "pipeline_original.json")
    assert len(original["points"]) == 8
    assert all(
        p["outputs_bitwise"] and all(all(c.values()) for c in p["checks"])
        for p in original["points"]
    )
    assert not checks["failed"]
    rows = checks["checks"]
    assert len(rows) == 30 * len(names) * 8
    assert len({(*x["shape"], x["variant"], x["distribution"]) for x in rows}) == len(
        rows
    )
    assert all(
        x["status"] == "PASS" and all(x["float32_bits_equal"].values()) for x in rows
    )
    assert {tuple(x["shape"]) for x in rows} == set(points)
    assert {x["variant"] for x in rows} == set(names)
    assert len({x["distribution"] for x in rows}) == 8
    for e in manifest["entries"]:
        f = OP / e["ir"]
        assert sha(f) == e["sha256"]
        m = meta[f'{e["experts"]}/{e["variant"]}'][e["stage"] - 1]
        assert m["ir_sha256"] == e["sha256"]
    assert p["aa_passed"]
    for row in p["records"]:
        t = row["timing"]
        assert len(t["samples"]) == rounds and len(t["aa"]) == rounds
        assert 0.97 <= st.median(x["ratio"] for x in t["aa"]) <= 1.03
        for n in names[1:]:
            assert (
                st.median(s["us"]["baseline"] / s["us"][n] for s in t["samples"])
                == t["baseline_over_candidate"][n]
            )
    return p, meta


def ratio_rows(run, before, after):
    return [
        st.median(s["us"][before] / s["us"][after] for s in r["timing"]["samples"])
        for r in run["records"]
    ]


def ratio(run, before, after):
    return st.geometric_mean(ratio_rows(run, before, after))


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--main", type=Path, default=ROOT / "results/main/measurement")
    cli.add_argument("--repeat", type=Path, default=ROOT / "results/repeat/measurement")
    args = cli.parse_args()
    args.main = args.main.resolve()
    args.repeat = args.repeat.resolve()
    primary, meta = audit(args.main, 42, 5)
    repeat, repeat_meta = audit(args.repeat, 123, 3)
    assert {
        k: [(x["ir_sha256"], x["stage_sha256"]) for x in v] for k, v in meta.items()
    } == {
        k: [(x["ir_sha256"], x["stage_sha256"]) for x in v]
        for k, v in repeat_meta.items()
    }
    selection = OP / "multiround/r6/selection.json"
    assert sha(selection) == read(ROOT / "manifest.json")["final_selection_sha256"]
    for e in (256, 512):
        for stage in (1, 2):
            assert (
                sha(ROOT / "ir" / f"e{e}_w4" / f"final_s{stage}.ttir")
                == read(selection)["selected_hashes"][f"e{e}/s{stage}"]
            )
    names = primary["names"]
    summary = dict(
        evidence_grade="A",
        shapes=30,
        variants=names,
        checks_per_run=3840,
        main=relative(args.main),
        repeat=relative(args.repeat),
        source_sha256={
            relative(p): sha(p)
            for root in (args.main, args.repeat)
            for p in [*root.glob("*.json"), root.parent / "pipeline_original.json"]
        },
        relative_to_baseline={
            n: dict(
                main=ratio(primary, "baseline", n), repeat=ratio(repeat, "baseline", n)
            )
            for n in names
        },
        additions={
            f"{a}->{b}": dict(main=ratio(primary, a, b), repeat=ratio(repeat, a, b))
            for a, b in zip(CHAIN, CHAIN[1:])
        },
        removals={
            k: dict(
                control=v,
                main=ratio(primary, v, "final"),
                repeat=ratio(repeat, v, "final"),
            )
            for k, v in REMOVALS.items()
        },
        aa={
            label: [
                min(r["timing"]["aa_median"] for r in run["records"]),
                max(r["timing"]["aa_median"] for r in run["records"]),
            ]
            for label, run in (("main", primary), ("repeat", repeat))
        },
    )
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    csv_text = io.StringIO()
    out = csv.writer(csv_text)
    out.writerow(
        [
            "tokens",
            "experts",
            "variant",
            "main_baseline_over_variant",
            "repeat_baseline_over_variant",
        ]
    )
    for i, row in enumerate(primary["records"]):
        for n in names:
            out.writerow(
                [
                    *row["shape"],
                    n,
                    ratio_rows(primary, "baseline", n)[i],
                    ratio_rows(repeat, "baseline", n)[i],
                ]
            )
    (ROOT / "all_shapes.csv").write_text(csv_text.getvalue())
    lines = [
        "# MoE routing 最终版 TTIR 消融实验",
        "",
        (
            "研究已验收的 r6 `batch4`，不修改原始 baseline、冻结源码、最终选择或 optimized 别名。官方 Triton 3.7.0 / H100 / FP"
            "32；完整30点（E=256/512，各15个tokens），G=8、选4组和8专家、稳定小索引tie，4 warps / 3 stages / 1 CTA。所有路线计完"
            "整原两阶段设备序列。实验合同见[ablation_plan.md](ablation_plan.md)。"
        ),
        "",
        "## 各版本相对原始 baseline",
        "",
        "| 版本 | 主测几何加速 | 独立复测几何加速 |",
        "|---|---:|---:|",
    ]
    for n in names:
        v = summary["relative_to_baseline"][n]
        lines.append(f"| {LABELS[n]} | {v['main']:.4f}× | {v['repeat']:.4f}× |")
    lines += [
        "",
        (
            "主测GPU2、seed42、5轮；独立进程GPU3、seed123、3轮；rep10ms，每版本100ms稳定阶段，同会话随机交错。全部逐shape A/A中位数通过[0"
            ".97,1.03]，完整矩阵及独立复测证据为A级；不同GPU保留各自的配对baseline，不比较两张GPU的绝对耗时。仅计预分配CUDA graph两kernel设备时"
            "间，自动频率、不清cache；不含分配、编译和CPU launch，不代表端到端或冷cache。"
        ),
        "",
        "## 逐项加入的条件收益",
        "",
        "| 对照 | 改动 | 主测条件加速 | 独立复测条件加速 |",
        "|---|---|---:|---:|",
    ]
    for a, b in zip(CHAIN, CHAIN[1:]):
        v = summary["additions"][f"{a}->{b}"]
        lines.append(
            f"| {a} → {b} | {LABELS[b]} | {v['main']:.4f}× | {v['repeat']:.4f}× |"
        )
    lines += [
        "",
        "## 从最终版移除改动",
        "",
        "以下为缺少改动版本耗时 / 最终版耗时，大于1表示该改动在其他最终变化已存在时有正贡献。",
        "",
        "| 去掉的变化 | 主测条件加速 | 独立复测条件加速 |",
        "|---|---:|---:|",
    ]
    for k, v in summary["removals"].items():
        label = "stage1 batch4" if k == "drop_batch" else LABELS[k]
        lines.append(f"| {label} | {v['main']:.4f}× | {v['repeat']:.4f}× |")
    lines += [
        "",
        (
            "去掉pair reducer时保留batch4二维数据流，只恢复原始max1/计数/屏蔽唯一最大/max2归约。去掉整数特化时恢复运行时标量和地址stride，保留ten"
            "sor形状与batch4；不声称支持新的运行时E/G/stride。去掉batch4复用add_constants，其他最终变化保持不变。"
        ),
        "",
        (
            "条件比值直接用同一shape、同一外层轮次的前版/后版时间，再取中位数和跨shape几何平均；不相乘历史轮平均。存在编译交互，加入和移除的条件收益可以不同，不能相加为总收"
            "益；低于约1%的差异不据此宣称稳定优化，未提供置信区间。"
        ),
        "",
        "## 两阶段交叉消融",
        "",
        "| 完整两次launch的组合 | 主测几何加速 | 独立复测几何加速 |",
        "|---|---:|---:|",
    ]
    for n in ("baseline", "stage1_only", "stage2_only", "final"):
        v = summary["relative_to_baseline"][n]
        lines.append(f"| {LABELS[n]} | {v['main']:.4f}× | {v['repeat']:.4f}× |")
    lines += [
        "",
        "此处使用完整30点；历史stage_ablation/main和repeat的六点结果保留为B级，不将子集值混入本表。",
        "",
        "## 编译资源与验证",
        "",
        "| E | 路线 | stage1 regs/spill/shared B | stage2 regs/spill/shared B |",
        "|---:|---|---|---|",
    ]
    for e in (256, 512):
        for n in names:
            cells = [
                f"{x['n_regs']}/{x['n_spills']}/{x['metadata']['shared']}"
                for x in meta[f"{e}/{n}"]
            ]
            lines.append(f"| {e} | {n} | {' | '.join(cells)} |")
    lines += [
        "",
        "资源表为C级编译事实，没有采集NCU动态指标，寄存器/shared变化不能单独证明加速原因。",
        "",
        (
            "每进程先运行独立verify_pipeline，再在runner中逐30点确认冻结源码和baseline文件入口LLIR/PTX/cubin与编译options一致、输出"
            "一致。每次3840项（16路线×30点×8分布）全部通过：group scores/weights的FP32原始位型与baseline完全一致、indices精确、CPU"
            "参考权重容差rtol2e-5/atol2e-6、canary、输入不变。final两阶段的IR哈希逐项等于已验收r6选择。"
        ),
        "",
        "既有最终版[memcheck/racecheck](multiround/r6/report.md)已通过；本次新增消融路线没有额外运行sanitizer，不将最终版的结果外推为全部消融路线的sanitizer结论。",
        "",
        "## 能否用普通 Triton 实现",
        "",
        "| 改动 | 公开Triton表达 | 边界 |",
        "|---|---|---|",
        "| membership | expand_dims/broadcast/reshape | 固定连续等长分组 |",
        "| 有序key | bitcast、位运算、uint32 max | 有限分数和符号零正规化，仍需稳定tie |",
        "| 循环后gather | tl.gather | 布局与编译收益需实测 |",
        "| 8×8排名 | broadcast、比较、tl.sum | 用较小索引打破同分 |",
        "| 双值top2 | tuple tl.reduce | 保留原舍入/最高两值的语义 |",
        "| 整数特化 | constexpr | 同一固定合同内表达，JIT ABI可能不同 |",
        "| batch4 | program_id、统一条件、二维arange/reduce | 原grid中非活动CTA返回 |",
        "",
        "以上为源码可表达性说明，不是r6最终源码版的性能或机器码等价证据。已有[首轮源码实测](source_port/report.md)只覆盖当轮membership，不能代表本轮所有改动。",
        "",
        "## 全部 shape",
        "",
        "各单元格为主测 / 独立复测相对共同baseline的逐shape配对比中位数；全部16版本见[CSV](ablation_v1/all_shapes.csv)。",
        "",
        "| (tokens,E) | 最终版 | 仅stage1 | 仅stage2 |",
        "|---|---:|---:|---:|",
    ]
    for i, row in enumerate(primary["records"]):
        cells = [
            f"{ratio_rows(primary, 'baseline', n)[i]:.4f}× / {ratio_rows(repeat, 'baseline', n)[i]:.4f}×"
            for n in ("final", "stage1_only", "stage2_only")
        ]
        lines.append(f"| {tuple(row['shape'])} | {' | '.join(cells)} |")
    lines += [
        "",
        (
            f"证据：[主测]({relative(args.main)}/performance.json)、"
            f"[独立复测]({relative(args.repeat)}/performance.json)、"
            f"[主测数值]({relative(args.main)}/accuracy.json)、"
            f"[复测数值]({relative(args.repeat)}/accuracy.json)、"
            f"[编译资源和哈希]({relative(args.main)}/compile.json)、"
            "[冻结谱系](ablation_v1/manifest.json)、[审计汇总](ablation_v1/summary.json)。"
        ),
        "",
        (
            "复现：先保留CUDA_VISIBLE_DEVICES给独占空闲卡，分别运行`bash moe_routing/run_experiment.sh NEW_MAIN 42 "
            "5`和`bash moe_routing/run_experiment.sh NEW_REPEAT 123 3`；两次均完成后，`python moe_routing/r"
            "eport_ablation.py --main NEW_MAIN/measurement --repeat NEW_REPEAT/measurement`重新核对并生成"
            "报告。固定使用`/root/venvs/ir-ops-dev/bin/python`。新输出目录不可存在；既有候选与结果不覆盖。"
        ),
        "",
    ]
    (OP / "report_ablation.md").write_text("\n".join(lines))
    print(json.dumps(summary["relative_to_baseline"], indent=2))


if __name__ == "__main__":
    main()
