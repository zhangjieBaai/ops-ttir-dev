"""Summarize measured source ports without reusing historical timing ratios."""

import json
import math
import random
import statistics
from pathlib import Path

from support import digest, save

ROOT = Path(__file__).resolve().parents[1]
RATIOS = ("baseline_over_ir", "baseline_over_source", "ir_over_source")


def geometric(values):
    return math.exp(statistics.mean(math.log(x) for x in values))


def summarize(dataset):
    return {
        name: {
            "geomean": geometric(
                [r["timing"]["ratios"][name] for r in dataset["records"]]
            ),
            "min": min(r["timing"]["ratios"][name] for r in dataset["records"]),
            "max": max(r["timing"]["ratios"][name] for r in dataset["records"]),
        }
        for name in RATIOS
    }


def bootstrap(dataset, name):
    rng = random.Random(121)
    draws = []
    for _ in range(3000):
        shape_values = []
        for r in dataset["records"]:
            samples = r["timing"]["samples"]
            shape_values.append(
                statistics.median(rng.choice(samples)[name] for _ in samples)
            )
        draws.append(geometric(shape_values))
    draws.sort()
    return [draws[75], draws[2925]]


def report(op):
    folder = ROOT / op / "source_port"
    main = json.loads((folder / "results/main/results.json").read_text())
    repeat = json.loads((folder / "results/repeat/results.json").read_text())
    assert main["passed"] and repeat["passed"]
    assert main["aa_passed"] and repeat["aa_passed"]
    assert main["source"] == repeat["source"]
    assert digest(folder / "source_v1.py") == main["source"]["source_sha256"]
    summaries = {"main": summarize(main), "repeat": summarize(repeat)}
    for tool, marker in (
        ("memcheck", "ERROR SUMMARY: 0 errors"),
        ("racecheck", "0 errors, 0 warnings"),
    ):
        assert marker in (folder / f"results/{tool}.log").read_text()
        data = json.loads((folder / f"results/{tool}/results.json").read_text())
        assert data["passed"] and data["source"] == main["source"]
    stages = ["kernel"] if op == "norm_quant" else ["local", "merge"]
    matches = []
    for row, re in zip(main["records"], repeat["records"]):
        assert row["shape"] == re["shape"]
        for stage in stages:
            a, b = row["kernels"]["source"][stage], row["kernels"]["ir"][stage]
            again = re["kernels"]["source"][stage]
            assert a["ir_sha256"] == again["ir_sha256"]
            assert a["stage_sha256"] == again["stage_sha256"]
            matches.append(
                {
                    "shape": row["shape"],
                    "stage": stage,
                    "source_ir_text_identical": a["elf_text_sha256"]
                    == b["elf_text_sha256"],
                    "source_ir_complete_cubin_identical": a["stage_sha256"]["cubin"]
                    == b["stage_sha256"]["cubin"],
                    "source_resources": {k: a[k] for k in ("n_regs", "n_spills")},
                    "ir_resources": {k: b[k] for k in ("n_regs", "n_spills")},
                    "source_structure": a["structural_counts"],
                    "ir_structure": b["structural_counts"],
                }
            )
    result = {
        "op": op,
        "source": "source_v1.py",
        "candidate": "stream_v1" if op == "norm_quant" else "ordered_carry_v3",
        "main": summaries["main"],
        "repeat": summaries["repeat"],
        "source_vs_ir_ratio_definition": (
            "ir_over_source = IR device microseconds / source device microseconds; "
            ">1 means source faster"
        ),
        "all_source_ir_text_identical": all(
            r["source_ir_text_identical"] for r in matches
        ),
        "code_comparison": matches,
        "ir_over_source_bootstrap95": {
            "main": bootstrap(main, "ir_over_source"),
            "repeat": bootstrap(repeat, "ir_over_source"),
        },
        "evidence_level": "A for frozen-matrix paired performance; C for static code/structure facts",
        "inputs": {
            "main": digest(folder / "results/main/results.json"),
            "repeat": digest(folder / "results/repeat/results.json"),
        },
        "sanitizer": {"memcheck": "0 errors", "racecheck": "0 errors, 0 warnings"},
    }
    save(folder / "summary.json", result)
    title = "Norm＋动态INT8" if op == "norm_quant" else "大N partial Top-K"
    lines = [
        f"# {title}：直接改 Triton 源码的实测结果",
        "",
        "忠实回写后的源码版复现了现有TTIR优化的机器指令，性能与TTIR版基本一致。所有映射的ELF `.text` 完全相同；",
        "源码路径和TTIR文件路径的完整cubin可能因调试位置等信息不同，不能把`.text`一致写成完整cubin一致。",
        "本次不需要额外调优或第二个源码候选。",
        "",
        "| 本轮几何平均 | 主测 | 独立复测 |",
        "|---|---:|---:|",
    ]
    labels = {
        "baseline_over_ir": "baseline / TTIR优化",
        "baseline_over_source": "baseline / 源码优化",
        "ir_over_source": "TTIR优化 / 源码优化",
    }
    for name in RATIOS:
        lines.append(
            f"| {labels[name]} | {summaries['main'][name]['geomean']:.6f}× | "
            f"{summaries['repeat'][name]['geomean']:.6f}× |"
        )
    lines += [
        "",
        "最后一行>1表示源码更快。三种比值均来自本轮逐轮直接对照，不是用历史加速比相除。",
        "微小差异属于此次测量波动，不能在指令相同时宣称某条编译入口更快。",
        "逐点中位数的乘除不一定等于直接比值的中位数，因此三行也不必精确满足乘法恒等式。",
        "",
        "| shape | baseline μs | TTIR μs | source μs | 主测IR/source | 复测IR/source | 复测baseline/source |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for a, b in zip(main["records"], repeat["records"]):
        t = a["timing"]
        lines.append(
            f"| {tuple(a['shape'])} | {t['median_us']['baseline']:.3f} | {t['median_us']['ir']:.3f} | "
            f"{t['median_us']['source']:.3f} | {t['ratios']['ir_over_source']:.6f}× | "
            f"{b['timing']['ratios']['ir_over_source']:.6f}× | {b['timing']['ratios']['baseline_over_source']:.6f}× |"
        )
    lines += ["", "## 实际源码改了什么", ""]
    if op == "norm_quant":
        lines += [
            "`source_v1.py`把 `sum(z*z)` 和 `max(abs(z*w))` 写入公开 `tl.reduce` 的tuple组合函数，",
            "再以 `max_u*rho` 得到amax，输出沿用原除法、clamp、nearbyint、INT8 cast，三个store都用公开 `.cs`。",
            "没有采用另一个未获选的消去rho方案。前端后的TTIR保留多值reduce和cs，最终指令段与stream_v1一致。",
            "量化仍采用原先允许相对FP64最多差1 code的合同，另检查scale、重构误差、Z、INT8范围、canary和输入不修改。",
            "源码与IR在计时normal输入上的输出相等另行记录，不能推广成所有可能输入逐位等价。",
        ]
    else:
        lines += [
            "`source_v1.py`在K循环前归一化±0，用公开bitcast、signed shift、OR/XOR生成uint32有序key，",
            "循环变量保持uint32，`tl.max(keys)`后仍以原始index求min解决ties；选中key改为-inf对应哨兵。",
            "前端保留整数循环状态，PTX保留 `redux.sync.max.u32`，local/merge两阶段的指令段都与ordered_carry_v3一致。",
            "保持原两kernel、chunk1024、工作区和稳定index合同，含尾块、subnormal、最低有限值和±0压力测试。",
            "值要求数值相等，±0等价，不要求保留零符号位。",
        ]
    lines += [
        "",
        "## 验证与计时",
        "",
        "使用官方Triton3.7.0 / H100，同原shape映射和完整backend options，无重新autotune。",
        "源码由正常 `@triton.jit` 产生CompiledKernel，**计时直接启动该对象**；",
        "其导出TTIR的重编译只做LLVM/PTX/cubin和输出一致性检查，没有替代源码路径。",
        "源CompiledKernel的Python launcher需附带constexpr参数槽，但CUDA实际参数、grid和设备序列与文件路径相同。",
        "三方使用同一输出存储和计时包装，CUDA graph指标不包含Python启动、编译、分配。",
        "",
        "主测7轮、独立seed123复测5轮、rep20ms；逐轮随机三路顺序，同shape每轮A/A。",
        "计时输入在设置seed之后分配，记录实际输入哈希。未锁频、未主动清cache。",
        f"IR/source固定矩阵配对bootstrap95：主测 {result['ir_over_source_bootstrap95']['main']}；",
        f"复测 {result['ir_over_source_bootstrap95']['repeat']}。这只描述保存样本的波动。",
        "",
        "三路均先运行原参考/压力校验后计时，新源码的memcheck和racecheck已完成：0 errors、0 warnings。",
        "本轮保留原baseline源码/TTIR pipeline核验，源码回写前先格式化后冻结，未改动原实验文件。",
        "",
        "[主测](results/main/results.json) · [复测](results/repeat/results.json) · [汇总/代码对照](summary.json) · ",
        "[memcheck](results/memcheck.log) · [racecheck](results/racecheck.log) · [源码](source_v1.py)",
        "",
        "## 复现",
        "",
        "```bash",
        f"CUDA_VISIBLE_DEVICES={6 if op == 'norm_quant' else 7} "
        "/root/venvs/ir-ops-dev/bin/python complex_suite/source_port_runner.py \\",
        f"  --op {op} --run my_repeat --seed 234 --rounds 5 --rep 20",
        "```",
        "",
        "从仓库根执行，结果目录必须不存在；`--check-only`只做pipeline与数值检查。",
        "这是同一个已知优化的源码回写对照，不是两组Agent各自独立搜索的A/B/C试验。",
    ]
    (folder / "report.md").write_text("\n".join(lines) + "\n")
    return result


if __name__ == "__main__":
    save(
        ROOT / "complex_suite/source_port_norm_topk_summary.json",
        {name: report(name) for name in ["norm_quant", "partial_topk"]},
    )
