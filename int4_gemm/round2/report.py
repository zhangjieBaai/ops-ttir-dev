"""Finalize expanded INT4 round2 only after measured incremental qualification."""

import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import common as c  # noqa: E402


def geo(values):
    return math.exp(statistics.mean(math.log(x) for x in values))


def bootstrap(data, candidate, metric):
    rng = random.Random(37)
    values = []
    for _ in range(3000):
        ratios = []
        for row in data["records"]:
            samples = row["timing"]["samples"]
            ratios.append(
                statistics.median(
                    rng.choice(samples)[metric][candidate] for _ in samples
                )
            )
        values.append(geo(ratios))
    values.sort()
    return [values[75], values[2925]]


def main():
    main_data = json.loads((ROOT / "results/main/results.json").read_text())
    repeat = json.loads((ROOT / "results/repeat/results.json").read_text())
    screen = json.loads((ROOT / "results/screen/results.json").read_text())
    frozen = json.loads((ROOT / "selection_pre_main.json").read_text())
    candidate = frozen["candidate"]
    assert frozen["screen_sha256"] == c.sha(ROOT / "results/screen/results.json")
    assert len(main_data["records"]) == len(repeat["records"]) == 30
    for data in (main_data, repeat):
        assert data["passed"] and data["aa_passed"]
        assert data["manifest_sha256"] == frozen["manifest_sha256"]
    assert "ERROR SUMMARY: 0 errors" in (ROOT / "results/memcheck.log").read_text()
    assert "0 errors, 0 warnings" in (ROOT / "results/racecheck.log").read_text()
    for name in ("memcheck", "racecheck"):
        check = json.loads((ROOT / f"results/{name}/results.json").read_text())
        assert check["passed"] and len(check["records"]) == 30
        assert check["manifest_sha256"] == frozen["manifest_sha256"]
    m, r = main_data["summary"][candidate], repeat["summary"][candidate]
    accepted = (
        min(m["incremental"]["geomean"], r["incremental"]["geomean"]) >= 1.02
        and min(m["incremental"]["minimum"], r["incremental"]["minimum"]) >= 0.95
    )
    rows = []
    for first, second in zip(main_data["records"], repeat["records"]):
        assert first["case"] == second["case"]
        for name in ("baseline", "parent", candidate):
            a, b = first["kernels"][name], second["kernels"][name]
            assert a["ttir_sha256"] == b["ttir_sha256"] == c.sha(a["path"])
            assert a["stages"] == b["stages"]
        picked = candidate if accepted else "parent"
        ir = ROOT / "expanded/ir" / first["case"]
        c.write_once(ir / "optimized.ttir", (ir / f"{picked}.ttir").read_text())
        rows.append(
            {
                "case": first["case"],
                "shape": first["shape"],
                "config": first["config"],
                "main_us": first["timing"]["median_us"],
                "repeat_us": second["timing"]["median_us"],
                "main_incremental": first["timing"]["incremental"][candidate],
                "repeat_incremental": second["timing"]["incremental"][candidate],
                "main_cumulative": first["timing"]["cumulative"][candidate],
                "repeat_cumulative": second["timing"]["cumulative"][candidate],
                "resources": {
                    n: {
                        "registers": v["n_regs"],
                        "spills": v["n_spills"],
                        "shared": v["metadata"]["shared"],
                    }
                    for n, v in first["kernels"].items()
                },
                "selected_sha256": c.sha(ir / "optimized.ttir"),
            }
        )
    summary = {
        "operator": "int4_gemm",
        "matrix_size": 30,
        "parent": "scalar_unsigned_v1",
        "candidate": candidate,
        "accepted": accepted,
        "selected": candidate if accepted else "parent",
        "evidence_level": "A, frozen30 matrix only",
        "main": m,
        "repeat": r,
        "parent_main": main_data["summary"]["parent"],
        "parent_repeat": repeat["summary"]["parent"],
        "bootstrap95_incremental": {
            "main": bootstrap(main_data, candidate, "incremental"),
            "repeat": bootstrap(repeat, candidate, "incremental"),
        },
        "bootstrap95_cumulative": {
            "main": bootstrap(main_data, candidate, "cumulative"),
            "repeat": bootstrap(repeat, candidate, "cumulative"),
        },
        "aa_ranges": {
            name: [
                min(x["timing"]["aa_median"] for x in d["records"]),
                max(x["timing"]["aa_median"] for x in d["records"]),
            ]
            for name, d in [("main", main_data), ("repeat", repeat)]
        },
        "accuracy_counts": {
            "screen": 1200,
            "main": 450,
            "repeat": 450,
            "memcheck": 450,
            "racecheck": 450,
        },
        "sanitizer": {
            "memcheck": "0 errors",
            "racecheck": "0 hazards, 0 errors, 0 warnings",
        },
        "source_expressibility": "Public Triton can express "
        "narrow load/broadcast and a128-step loop with "
        "two ordered64-K dot updates; no source-port "
        "benchmark in this round",
        "baseline_policy": "Original7 configurations "
        "retained; new23 fixed BM16 ifM<=16 elseBM32, "
        "BN64/BK64/4warps/3stages, no new autotuning",
        "sources": {
            name: c.sha(ROOT / name)
            for name in [
                "expanded/manifest.json",
                "selection_pre_main.json",
                "results/main/results.json",
                "results/repeat/results.json",
                "results/memcheck.log",
                "results/racecheck.log",
            ]
        },
        "records": rows,
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (ROOT / "selection.json").write_text(
        json.dumps(
            {
                "accepted": accepted,
                "candidate": candidate,
                "selected": summary["selected"],
                "matrix_sha256": frozen["manifest_sha256"],
                "records": [
                    {"case": x["case"], "sha256": x["selected_sha256"]} for x in rows
                ],
            },
            indent=2,
        )
        + "\n"
    )
    lines = [
        "# INT4 TTIR 第二轮：30点扩展矩阵上的窄加载与组内复用",
        "",
        f"最终选择 **{summary['selected']}**。相对上一轮已验收parent，"
        f"30点增量几何平均为 **{m['incremental']['geomean']:.4f}× "
        f"/ {r['incremental']['geomean']:.4f}×**（主测/独立复测）；"
        f"",
        f"相对原baseline的累计加速为 **{m['cumulative']['geomean']:.4f}× / {r['cumulative']['geomean']:.4f}×**。",
        f"最差点增量仍为 **{m['incremental']['minimum']:.4f}× "
        f"/ {r['incremental']['minimum']:.4f}×**。这是本固定30点的A级性能证据，"
        f"不是源代码/IR信息优势实验。",
        "",
        "## 范围与基线",
        "",
        "按用户追加要求，从原7点扩展为30点，覆盖M1..256、N64..11008、K128..8192以及M/N尾部。原7点保留其既有配置；新增23点使用预声明固定族：",
        "M<=16取BM16，否则BM32，BN64/BK64、4warps/3stages。新点没有逐shape autotune，不能把累计加速解释为胜过最强INT4实现。",
        "完整列表和配置映射在 [expanded_plan.md](expanded_plan.md) "
        "与 [manifest](expanded/manifest.json)。30点baseline均重新从原冻结tutorial导出；"
        "parent通过原scalar+unsigned变换生成，并在新增点重新验证。",
        "原7点目录仅保留为pilot生成记录，无计时；所有本报告性能来自expanded30。原baseline、parent与历史optimized均未改写。",
        "",
        "## 实际改动",
        "",
        "1. 将scale/zero的逻辑加载形状从 `[BK/2,BN]` 缩为 `[1,BN]`，先加载/扩展再广播。每个K tile内group已由父版本证明恒定，列stride为1，保留N mask及相同other值。",
        "2. 利用K为128倍数、BK64/128且k0对齐，删除activation/packed恒真的K边界判断；M/N尾部仍保留。",
        "3. BK64时将两个原64步迭代显式并入128步外循环，复用同一group的scale/zero；"
        "两个64-K dot按原先顺序依次更新同一accumulator，FP16舍入点与FP32累积次序不变。"
        "BK128点只执行前两项。",
        "",
        "只有TTIR内容改变；签名、grid、kernel数、workspace、完整options、编译/加载入口不变。公开Triton可用窄load+broadcast和128步循环内两个有序dot表达这些结构，",
        "不需要隐藏属性、inline PTX或手工改TTGIR/LLVM/PTX。本轮没有另做源码回写计时，不能声称源码已经复现本轮新收益。",
        "",
        "## 三路逐点时间",
        "",
        "单位μs；时间为各路径样本中位数，增量=parent/candidate，累计=baseline/candidate，均由同轮直接比值取中位数，不能用旧轮数字相除。",
        "",
        "| M,N,K | baseline | parent | new | 增量主测 | 增量复测 | 累计主测 | 累计复测 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        t = row["main_us"]
        lines.append(
            f"| {tuple(row['shape'])} | {t['baseline']:.3f} "
            f"| {t['parent']:.3f} | {t[candidate]:.3f} | {row['main_incremental']:.4f}× "
            f"| {row['repeat_incremental']:.4f}× | {row['main_cumulative']:.4f}× "
            f"| {row['repeat_cumulative']:.4f}× |"
        )
    lines += [
        "",
        "## 消融及负结果",
        "",
        "以下全部为30点、3轮10ms初筛（B级），只用于冻结唯一获选候选；不将筛选数字冒充最终结果。",
        "",
        "| 候选 | 相对parent几何平均 | 最差点 |",
        "|---|---:|---:|",
    ]
    for name, v in screen["summary"].items():
        if name not in ["baseline", "parent"]:
            lines.append(
                f"| {name} | {v['incremental']['geomean']:.4f}× | {v['incremental']['minimum']:.4f}× |"
            )
    lines += [
        "",
        "窄加载是主要单项收益；去掉K mask单独几乎没有平均收益且在M256,N4096,K4096回退到约0.80×，说明删mask不保证更快。",
        "A后移与scale.cg组合也没有超过group_pair；没有按每个shape挑最有利方案。parent在新增矩阵的部分点本身不如原baseline，这一负结果保留在原始时间和parent汇总。",
        "",
        "## 资源和机制证据（C级）",
        "",
        "在M128,N11008,K4096上，parent→narrow为138→96寄存器、shared20480→12288B、spill均0；group_pair为96寄存器、4spill、shared24576B。",
        "parent和narrow的TTIR都只有4个tt.load，但scale/zero加载形状不同，PTX静态ld.global计数为12→4；这证明不能只数TTIR load指令。",
        "pair版外循环包含6个load和2个dot，相当于两原迭代复用scale/zero；循环次数减半。"
        "静态计数、hash和资源在 [static_analysis.json](static_analysis.json) "
        "与原始结果里。",
        "未采集NCU动态DRAM字节/周期，不能把全部收益归给某个寄存器数、指令数或猜测带宽。",
        "",
        "## 验证、计时和复现",
        "",
        "原30点source/file LLVM/PTX/cubin逐字节一致；每次主/复测先运行verify_pipeline.py，"
        "run.py内也逐点重复门槛。完整backend options和target相同，官方Triton3.7.0/H100 "
        "GPU0独占。",
        "所有候选完整正确性先于计时。screen保留1200项检查，main/repeat/memcheck/racecheck各450项；"
        "全部按原随机/extreme nibble/zero scale/zero/constant五分布，"
        "要求与baseline逐位相同，",
        "对FP32 ATen参考使用原rtol0.003/atol0.035，小N<=256额外CPU FP64；有canary、输出覆盖、输入不修改和有限性检查。未放宽原精度合同。",
        "memcheck=0 errors，racecheck=0 hazards/0 errors/0 warnings，均覆盖完整30点、三路径和全部五分布。",
        "",
        "主测7轮/20ms，独立seed123复测5轮/20ms，三路随机交错，同等100ms预热，每轮parent A/A。输入seed/hash、IR及所有后端产物hash、资源和全部samples保留。",
        "预分配CUDA graph设备时间，不含输入构造/参考/分配/编译/Python启动；未锁频、未清cache。",
        f"A/A中位比范围：{summary['aa_ranges']}。增量固定矩阵bootstrap95：{summary['bootstrap95_incremental']}。区间只描述保存样本的波动。",
        "",
        "复现（从仓库根运行，结果目录必须新建）：",
        "",
        "```bash",
        "CUDA_VISIBLE_DEVICES=0 /root/venvs/ir-ops-dev/bin/python int4_gemm/round2/verify_pipeline.py \\",
        "  --output int4_gemm/round2/results/my_pipeline.json",
        "CUDA_VISIBLE_DEVICES=0 /root/venvs/ir-ops-dev/bin/python int4_gemm/round2/run.py \\",
        "  --run my_repeat --names baseline,parent,r2_group_pair --rounds 5 --rep 20 --seed 234",
        "```",
        "",
        "候选可由`make_candidates.py --expanded`逐字节重建，已有不同内容会拒绝覆盖。`--check-only`只验证，sanitizer可包裹同命令。",
        "[主测](results/main/results.json) · [独立复测](results/repeat/results.json) "
        "· [汇总](summary.json) · [冻结选择](selection.json) "
        "· [memcheck](results/memcheck.log) · [racecheck](results/racecheck.log)",
    ]
    (ROOT / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
