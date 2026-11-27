"""Finalize group-qualified incremental/cumulative evidence from frozen runs."""

import json
import math
import statistics

from common_round2 import ROOT, digest, save


def immutable(path, content):
    if path.exists():
        assert path.read_text() == content
    else:
        path.write_text(content)


def summary(rows, name):
    result = {}
    for ratio in (
        "parent_over_candidate",
        "baseline_over_candidate",
        "baseline_over_parent",
    ):
        values = [r["candidates"][name]["ratios"][ratio] for r in rows]
        result[ratio] = {
            "geo": math.exp(statistics.mean(math.log(v) for v in values)),
            "min": min(values),
            "max": max(values),
        }
    return result


def main():
    main = json.loads((ROOT / "results/main32/benchmark.json").read_text())
    repeat = json.loads((ROOT / "results/repeat32/benchmark.json").read_text())
    chosen = json.loads((ROOT / "selection_pre_main.json").read_text())
    assert (
        main["passed"]
        and repeat["passed"]
        and main["aa_passed"]
        and repeat["aa_passed"]
    )
    assert len(main["points"]) == len(repeat["points"]) == 32
    clock = json.loads((ROOT / "clock_lock_fullrun.json").read_text())
    assert clock["passed"] and clock["restored"]
    for timed in (main, repeat):
        assert timed["environment"]["clocks_locked"]
        assert timed["environment"]["requested_sm_clock_mhz"] == 1200
        assert timed["warmup_ms"] == 500
        assert timed["per_measurement_warmup_ms"] == 100
        assert all(
            warm["actual_event_ms"] >= warm["target_ms"]
            for row in timed["points"]
            for warm in row["warmup_records"]
        )
    for tool, marker in [
        ("memcheck", "ERROR SUMMARY: 0 errors"),
        ("racecheck", "0 errors, 0 warnings"),
    ]:
        assert marker in (ROOT / f"{tool}.log").read_text()
        data = json.loads((ROOT / "results" / tool / "accuracy.json").read_text())
        assert data["passed"] and all(data["valid"].values())
    result = {
        "matrix_points": 32,
        "evidence_level": "A: full frozen32 matrix, 7 main rounds and independent5 repeat rounds; static claims C",
        "groups": {},
        "clocks": {
            "locked_during_measurement": True,
            "requested_sm_mhz": 1200,
            "automatic_restored": clock["restored"],
            "record": "clock_lock_fullrun.json",
        },
        "aa": {
            name: {
                "min": min(row["aa_median"] for row in timed["points"]),
                "max": max(row["aa_median"] for row in timed["points"]),
            }
            for name, timed in (("main", main), ("repeat", repeat))
        },
        "sanitizer": {"memcheck": "0 errors", "racecheck": "0 errors, 0 warnings"},
        "sources": {
            str(p.relative_to(ROOT)): digest(p)
            for p in [
                ROOT / "results/main32/benchmark.json",
                ROOT / "results/repeat32/benchmark.json",
                ROOT / "selection_pre_main.json",
                ROOT / "clock_lock_fullrun.json",
            ]
        },
    }
    lines = [
        "# Attention dK/dV 第二轮：输出子块与循环状态",
        "",
        "沿用原始baseline与本轮开始时parent（D64=r1_dv_late，D128=baseline），只直接修改TTIR。"
        "正式性能集扩展为32点，每个D/causal组8点；没有把旧12点样本混入新矩阵。"
        "正式数据全部在GPU4临时锁定SM1200MHz下采集，不能与其他未锁频算子的时间混算。",
        "",
        "| 组 | 实测候选 | 主测增量parent/candidate | 复测增量 | 主测累计baseline/candidate | 复测累计 | 最终选择 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for group, name in chosen["tested_candidate"].items():
        m = [x for x in main["points"] if x["group"] == group]
        r = [x for x in repeat["points"] if x["group"] == group]
        assert len(m) == len(r) == 8
        sm, sr = summary(m, name), summary(r, name)
        accepted = all(
            s["parent_over_candidate"]["geo"] >= 1.02
            and s["parent_over_candidate"]["min"] >= 0.95
            for s in [sm, sr]
        )
        selected = name if accepted else "parent"
        folder = ROOT / "ir" / group
        immutable(
            folder / "optimized.ttir", (folder / (selected + ".ttir")).read_text()
        )
        result["groups"][group] = {
            "candidate_tested": name,
            "main": sm,
            "repeat": sr,
            "accepted": accepted,
            "selected": selected,
            "selected_sha256": digest(folder / "optimized.ttir"),
        }
        lines.append(
            f"| {group} | {name} | {sm['parent_over_candidate']['geo']:.4f}× "
            f"| {sr['parent_over_candidate']['geo']:.4f}× | "
            f"{sm['baseline_over_candidate']['geo']:.4f}× | "
            f"{sr['baseline_over_candidate']['geo']:.4f}× | "
            f"{selected} |"
        )
    save(ROOT / "summary.json", result)
    lines += [
        "",
        "组级接受门槛为主/复测增量几何均值均至少1.02×、组内最差至少0.95×，数值和sanitizer通过；不按复测逐shape重新选择。表中未接受候选的回退也保留，不能以人为1×覆盖其真实测量。",
        "本轮只晋升D128的16个测试点；D64的16点保留原parent。不能把D128约2.6–2.9×写成全部32点的增量。"
        "D128的parent与baseline本来相同，两路单独计时及中位数汇总会产生细微数值差异。",
        "",
        "## 改了什么",
        "",
        "rows64/rows32在原每CTA负责128个key的网格内，串行计算两个64-key或四个32-key输出子块。"
        "输出总覆盖、ABI、grid和kernel数不变；减小常驻K/V以及dK/dV accumulator。"
        "为保留累加次序，causal子块仍使用原128-key块的8次16-query masked循环，"
        "再接32-query unmasked循环，不改每个输出的query遍历。其代价是重复读取Q/dO、"
        "增加子块循环。",
        "",
        "remat_ptrs把循环携带的两个指针张量换为由scalar curr_m重算，循环仅携带dK/dV/curr_m；"
        "两种组合检查它与输出分块的交互。第六候选rows64_unroll直接展开两次子块，排查嵌套循环的流水影响。"
        "所有候选都从冻结parent TTIR直接重写，没有优化Python kernel导出。",
        "",
        "这些输出子块、循环展开与指针重算可用公开Triton循环和地址表达式实现；本轮未做源码回写对照，不宣称TTIR独有能力。",
        "",
        "## 逐点正式结果",
        "",
        "| (B,H,N,D,causal) | baseline μs | parent μs | candidate μs | 主测增量 | 复测增量 | 主测累计 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m, r in zip(main["points"], repeat["points"]):
        assert m["shape"] == r["shape"]
        name = chosen["tested_candidate"][m["group"]]
        t = m["candidates"][name]
        other = r["candidates"][name]
        us = t["median_us"]
        lines.append(
            f"| {tuple(m['shape'])} | {us['baseline']:.3f} |"
            f" {us['parent']:.3f} | {us[name]:.3f} | {t['ratios']['parent_over_candidate']:.4f}× "
            f"| {other['ratios']['parent_over_candidate']:.4f}× "
            f"| {t['ratios']['baseline_over_candidate']:.4f}× "
            f"|"
        )
    lines += [
        "",
        "设备时间为预分配输出CUDA graph，原官方Triton3.7.0/H100 GPU4、"
        "4warps/5stages/1CTA，未重新autotune；每版本500ms对等初始预热、"
        "每次计时前100ms同路线预热，主测7轮20ms，独立seed123复测5轮20ms，同轮随机三版本顺序。"
        "比值先逐点取配对中位数再几何平均，SM固定1200MHz、未主动清cache；时间不包含forward、K预缩放、"
        "delta、参考、编译和Python。",
        "",
        "原32点逐点源码baseline/文件入口LLVM/PTX/cubin一致性通过，完整backend "
        "options一致。完整矩阵与压力合计44输入；每个输入检查FP64参考、baseline逐位相同、"
        "输入不变、NaN sentinel覆盖与有限性，小N另CPU FP64 autograd。候选筛选前五项共308检查通过，"
        "增加展开项另132检查；合并去重后共352项，在固定频率下又重新完整验证352项。"
        "主/复测和两sanitizer各132检查。",
        "",
        "memcheck为0 errors；racecheck为0 errors/0 warnings。"
        "原始证据：[主测](results/main32/benchmark.json)、[复测](results/repeat32/benchmark.json)、"
        "[选择](selection_pre_main.json)、[memcheck](memcheck.log)、"
        "[racecheck](racecheck.log)。",
        "",
        "## 静态证据与负结果",
        "",
        "D128的32-row候选把记录spill从178/182降到0，但它也让PTX从WGMMA变成mma.sync；"
        "这不是免费降低寄存器。64-row保留WGMMA，仍需检查重复读取和串行处理代价。静态指令计数不是动态次数，"
        "也不能单项解释全部收益。详见[资源与PTX记录](static_pilot.json)。",
        "原baseline是官方固定4warps/5stages实现，未逐shape全面调优；D128原有严重spill。"
        "本轮收益不能解释为超过其他调优后的attention库，也不是完整backward或训练step收益。",
        "",
        "v1子块候选缺少SCF for的显式i32类型，v2又混淆128-head和128-key的范围，"
        "均在解析/验证阶段失败，文件及日志保留；v3修正后完整数值通过，没有改原baseline或放宽数值合同。正式验证额外按FP16原始int16位型比较，覆盖符号零。"
        "展开v4的纯数字SSA名字加后缀也在解析阶段失败，v5修复名字后通过完整验证。所有筛选负结果保留在[稳定六项筛选](results/screen32_locked/benchmark.json)；"
        "旧五项筛选的一个A/A超限点和中止的部分展开筛选均保留，见[计时修订](screen_disposition.json)。"
        "",
        "自动频率下，GPU4/5的大D64非因果parent对照均出现个别样本漂移；"
        "固定图计时也未完全消除，期间观测到SM频率变化。临时锁定1200MHz后控制稳定，"
        "于是全部32点重新筛选、主测、复测，不混入自动频率样本。"
        "计时本体仍是do_bench_cudagraph；仅预热改为可复用32-launch图并保存实际CUDA-event时长。",
        "设置前后、请求值、观测频率、finally恢复返回码和恢复后快照见"
        "[频率记录](clock_lock_fullrun.json)。GPU4已恢复自动频率。",
        "",
        "## 复现",
        "",
        "使用/root/venvs/ir-ops-dev/bin/python，先隔离空闲GPU。结果目录必须是新目录，"
        "命令示例见README.md；所有生成器仅写本目录并对既有候选做字节一致断言。原始父目录的源码、"
        "baseline、selection和历史报告没有修改。",
    ]
    (ROOT / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
