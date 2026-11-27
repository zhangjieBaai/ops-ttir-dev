"""Audit the accepted chain and report directly measured final effective routes."""

import hashlib
import json
import math
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parent
GROUPS = ["d64_c0", "d64_c1", "d128_c0", "d128_c1"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometric(xs):
    return math.exp(statistics.mean(math.log(x) for x in xs))


def main():
    clock = json.loads((BASE / "clock_run.json").read_text())
    assert clock["passed"] and clock["restored"] and clock["restore"]["exit_code"] == 0
    rounds = {
        rd: json.loads((BASE / rd / "summary.json").read_text())
        for rd in ("r3", "r4", "r5")
    }
    chain = {g: [] for g in GROUPS}
    for rd, summary in rounds.items():
        assert summary["matrix_points"] == 32
        previous = BASE.parent / "round2" if rd == "r3" else BASE / f"r{int(rd[1:])-1}"
        for g, info in summary["groups"].items():
            parent = BASE / rd / "ir" / g / "parent.ttir"
            assert (
                parent.read_bytes()
                == (previous / "ir" / g / "optimized.ttir").read_bytes()
            )
            selected = BASE / rd / "ir" / g / "optimized.ttir"
            assert digest(selected) == info["selected_sha256"]
            if info["accepted"]:
                chain[g].append(
                    {
                        "round": rd,
                        "candidate": info["selected"],
                        "sha256": info["selected_sha256"],
                    }
                )
    effective = {}
    final_samples = {}
    for phase in ("main", "repeat"):
        data = json.loads((BASE / f"r5/results/{phase}/benchmark.json").read_text())
        assert data["passed"] and data["aa_passed"]
        assert (
            data["environment"]["clocks_locked"]
            and data["environment"]["requested_sm_clock_mhz"] == 1200
        )
        rows = []
        for point in data["points"]:
            g = point["group"]
            info = rounds["r5"]["groups"][g]
            route = info["selected"]
            samples = [
                {
                    "anchor_over_final": s["times_us"]["anchor"] / s["times_us"][route],
                    "baseline_over_final": s["times_us"]["baseline"]
                    / s["times_us"][route],
                    "parent_over_final": s["times_us"]["parent"] / s["times_us"][route],
                    "times_us": s["times_us"],
                    "order": s["order"],
                }
                for s in point["samples"]
            ]
            row = {
                "shape": point["shape"],
                "group": g,
                "measured_route": route,
                "final_ttir_sha256": point["metadata"][route]["ttir_sha256"],
                "ratios": {
                    key: statistics.median(s[key] for s in samples)
                    for key in (
                        "anchor_over_final",
                        "baseline_over_final",
                        "parent_over_final",
                    )
                },
                "median_us": point["median_us"],
                "samples": samples,
            }
            assert row["final_ttir_sha256"] == digest(
                BASE / "r5/ir" / g / "optimized.ttir"
            )
            rows.append(row)
        final_samples[phase] = rows
        effective[phase] = {
            metric: {
                "geomean": geometric([r["ratios"][metric] for r in rows]),
                "worst": min(r["ratios"][metric] for r in rows),
                "best": max(r["ratios"][metric] for r in rows),
            }
            for metric in (
                "anchor_over_final",
                "baseline_over_final",
                "parent_over_final",
            )
        }
    subsets = {}
    for label in ("d64", "d128"):
        subsets[label] = {"points": 16}
        for phase, rows in final_samples.items():
            filtered = [r for r in rows if r["group"].startswith(label + "_")]
            assert len(filtered) == 16
            subsets[label][phase] = {
                metric: {
                    "geomean": geometric([r["ratios"][metric] for r in filtered]),
                    "worst": min(r["ratios"][metric] for r in filtered),
                    "best": max(r["ratios"][metric] for r in filtered),
                }
                for metric in ("anchor_over_final", "baseline_over_final")
            }
    static = {}
    for rd in rounds:
        accuracy = json.loads(
            (BASE / rd / "results/accuracy_screen/accuracy.json").read_text()
        )
        static[rd] = {}
        for g in GROUPS:
            dim = 64 if g.startswith("d64_") else 128
            causal = g.endswith("c1")
            parent = accuracy["compile"][str(("parent", dim, causal))]["metadata"]
            names = json.loads((BASE / rd / "manifest.json").read_text())[g][
                "candidates"
            ]
            static[rd][g] = {}
            for name in names:
                rec = accuracy["compile"].get(str((name, dim, causal)), {})
                if rec.get("status") != "PASS":
                    static[rd][g][name] = {"compile_status": rec.get("status")}
                    continue
                md = rec["metadata"]
                static[rd][g][name] = {
                    "same_text_as_parent": md["elf_text"] == parent["elf_text"],
                    "n_regs": md["n_regs"],
                    "n_spills": md["n_spills"],
                    "shared": md["shared"],
                    "parent_resources": {
                        k: parent[k] for k in ("n_regs", "n_spills", "shared")
                    },
                }
    summary = {
        "operator": "attention_dkdv",
        "round_count": 3,
        "rounds": rounds,
        "matrix_points_per_round": 32,
        "distinct_shapes": 32,
        "shape_rounds": 96,
        "accepted_chain": chain,
        "final": effective,
        "final_subsets": subsets,
        "final_metric_scope": (
            "All32 actual measured r5 effective routes; parent fallback uses measured parent times, "
            "never artificial1x"
        ),
        "final_samples": final_samples,
        "static": static,
        "clocks": {
            "locked_during_measurement": True,
            "requested_sm_mhz": 1200,
            "automatic_restored": True,
        },
        "clock_record_sha256": digest(BASE / "clock_run.json"),
    }
    (BASE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Attention dK/dV：第三至第五轮TTIR优化",
        "",
        "完成三个顺序搜索轮，每轮固定32个shape（四个D/causal组各8点），并保持原精度、ABI、grid、4warps/5stages。",
        "baseline为原官方提取版本，anchor为第二轮验收版本，parent只随已验收结果更新。没有把复测种子当成新优化轮次。",
        "",
        "| 轮次 | 新假设 | 完成情况 |",
        "|---|---|---|",
        "| r3 | D128组合外层展开与指针重算；D64将dP提前与既有dV后移组合；两次查询迭代合并 | 全32点筛选、主测、独立复测与sanitizer |",
        "| r4 | Q/dO专用cg；在当前parent上提前delta读取 | 全32点筛选、主测、独立复测与sanitizer |",
        "| r5 | 输出cs；Q/dO ca+evict_last | 全32点筛选、主测、独立复测与sanitizer |",
        "",
        "## 各组的逐轮决策",
        "",
        "表中数值是实际测试候选相对当轮parent的增量几何均值；未通过验收的数值保留，不冒充最终选择。",
        "",
        "| 轮次 | 组 | 候选 | 主测增量 | 独立复测增量 | 复测最差 | 决策 |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for rd, r in rounds.items():
        for g, info in r["groups"].items():
            lines.append(
                f"| {rd} | {g} | {info['tested']} | {info['main']['parent_over']['geo']:.4f}× | "
                f"{info['repeat']['parent_over']['geo']:.4f}× | {info['repeat']['parent_over']['min']:.4f}× | "
                f"{'接受' if info['accepted'] else '保留parent'} |"
            )
    lines += [
        "",
        "接受门槛事先固定：两次增量geo>=1.02、组内最差>=0.95、A/A在[0.97,1.03]，数值与sanitizer通过。",
        "没有按独立复测逐shape重新挑最有利版本。",
        "",
        "## 最终有效选择的同轮实测",
        "",
        "下面对全部32点使用r5同轮实际测得的最终路线：若某组不晋升，使用该组真实parent时间；没有填入人为1×，也没有乘历史几何均值。",
        "",
        "| 指标（完整32点） | 主测 | 独立复测 |",
        "|---|---:|---:|",
        f"| 相对第二轮anchor | {effective['main']['anchor_over_final']['geomean']:.6f}× | "
        f"{effective['repeat']['anchor_over_final']['geomean']:.6f}× |",
        f"| 相对原始baseline | {effective['main']['baseline_over_final']['geomean']:.6f}× | "
        f"{effective['repeat']['baseline_over_final']['geomean']:.6f}× |",
        "",
        "此处完整32点包含D64和D128，不能与第二轮仅D12816点的约2.74×相除。最终代码映射为r5/ir/<group>/optimized.ttir，",
        "每轮选择和全部实际路线时间见summary.json及各轮原始benchmark.json。",
        "",
        "| 最终子集 | 独立复测相对R2anchor | 独立复测相对原baseline |",
        "|---|---:|---:|",
        f"| D64，16点 | {subsets['d64']['repeat']['anchor_over_final']['geomean']:.6f}× | "
        f"{subsets['d64']['repeat']['baseline_over_final']['geomean']:.6f}× |",
        f"| D128，16点 | {subsets['d128']['repeat']['anchor_over_final']['geomean']:.6f}× | "
        f"{subsets['d128']['repeat']['baseline_over_final']['geomean']:.6f}× |",
        "",
        "## 机制与边界",
        "",
        "查询二合一保持原accumulator更新顺序，但扩大软件流水内活跃数据；例如D128因果parent98 spills升到query_pair340，",
        "shared由115712B升到198656B。这是静态资源事实，不能把全部性能退步单独归因于spill。",
        "展开与重算组合在D128因果降低部分spill，却没有自动带来稳定组级收益。r4的qdo_cg四组可执行.text与parent完全相同，",
        "说明改变TTIR cache表达未改变此实例的最终指令；相关控制时间仍保留。",
        "r5的ca+evict_last同样在四组得到与parent相同的.text；属性仍能在TTGIR async-copy中看到，",
        "但最终PTX仍为相同cp.async.cg路径。仅对本版本/配置成立，不推广成所有cache hint都无效。",
        "阶段片段和哈希见[cache lowering静态证据](cache_lowering_evidence.json)。",
        "本轮没有调整全局或loop-local pipeline depth；共享内存数接近资源边界只作为后续独立调参研究线索，未证明具体occupancy。",
        "",
        "这些结构有公开Triton的循环、表达式次序、load/store cache参数对应写法；本轮没有做源码回写性能对照。",
        "原第二轮及其更早文件全部保持不变，新候选和失败记录只在本目录。",
        "",
        "## 验证与计时",
        "",
        "官方Triton3.7.0，GPU4独占，整个三轮工作流固定SM1200MHz。每轮全32点原source/file LLVM/PTX/cubin一致性核验，",
        "44输入参考检查（32正常+四组各3压力），FP64门槛不变，候选输出按int16视图与baseline逐位比较，输入不修改、输出覆盖/有限性通过。",
        "每轮筛选3次rep10，主测7次rep20，独立seed123复测5次rep20；rep是库参数，默认内部10次重放均值构成一个外层样本。",
        "每路初始实际>=500ms、每次测量前实际>=100ms等条件预热；每轮随机四路线顺序，同输入预分配CUDA graph设备时间，",
        "不含分配、编译、参考和Python启动，未主动清cache。完整options/输入/IR/产物哈希、资源、所有外层配对样本保留。",
        "三轮最终候选均跑memcheck/racecheck，0 errors、0 warnings。时钟在finally恢复，详见clock_run.json。",
        "",
        "[r3报告](r3/report.md) · [r4报告](r4/report.md) · [r5报告](r5/report.md) · "
        "[完整汇总](summary.json) · [协议](optimization_plan.md)",
    ]
    (BASE / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
