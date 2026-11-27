"""Derive the source-port report exclusively from immutable run evidence."""

import hashlib
import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KEYS = ("baseline_over_ir", "baseline_over_source", "ir_over_source")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(data):
    result = {}
    for key in KEYS:
        values = [r["timing"]["ratios"][key] for r in data["records"]]
        rng = random.Random(453)
        boot = []
        for _ in range(3000):
            ratios = []
            for row in data["records"]:
                samples = row["timing"]["samples"]
                ratios.append(
                    statistics.median(rng.choice(samples)[key] for _ in samples)
                )
            boot.append(math.exp(statistics.mean(math.log(v) for v in ratios)))
        boot.sort()
        result[key] = dict(
            geomean=math.exp(statistics.mean(math.log(v) for v in values)),
            minimum=min(values),
            maximum=max(values),
            bootstrap95_fixed_shapes=[boot[75], boot[2925]],
        )
    return result


def main():
    main_data = json.loads((ROOT / "results/main_v2/results.json").read_text())
    repeat = json.loads((ROOT / "results/repeat/results.json").read_text())
    for data in (main_data, repeat):
        assert data["passed"] and data["aa_passed"]
        assert data["source_sha256"] == digest(ROOT / "source_v1.py")
    mem = (ROOT / "memcheck.log").read_text()
    race = (ROOT / "racecheck.log").read_text()
    assert "ERROR SUMMARY: 0 errors" in mem
    assert "0 errors, 0 warnings" in race
    m, r = summarize(main_data), summarize(repeat)
    summary = dict(
        experiment="INT4 direct Triton source port of scalar_unsigned_v1",
        candidate="source_v1",
        main=m,
        repeat=r,
        evidence_level="A, fixed seven-point matrix; no optimization-search information ablation",
        source_vs_ir_executable_text_identical_all_cases=True,
        source_sha256=digest(ROOT / "source_v1.py"),
        sources={
            p: digest(ROOT / p)
            for p in [
                "results/main_v2/results.json",
                "results/repeat/results.json",
                "memcheck.log",
                "racecheck.log",
            ]
        },
        records=[],
    )
    for row, other in zip(main_data["records"], repeat["records"]):
        assert row["case"] == other["case"]
        for item in (row, other):
            assert item["kernels"]["source"]["compiler_input_class"] == "ASTSource"
            assert (
                item["kernels"]["ir"]["elf_text"]
                == item["kernels"]["source"]["elf_text"]
            )
            assert all(item["pipeline"].values())
            assert len(item["accuracy"]) == 15
        summary["records"].append(
            dict(
                shape=row["shape"],
                config=row["config"],
                main_us=row["timing"]["median_us"],
                repeat_us=other["timing"]["median_us"],
                main_ratios=row["timing"]["ratios"],
                repeat_ratios=other["timing"]["ratios"],
                text_sha256=row["kernels"]["source"]["elf_text"],
            )
        )
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "**INT4 的已知 TTIR 优化已成功回写成普通 Triton 源码，且7个测试点的源码版与TTIR优化版产生完全相同的可执行 `.text`。**",
        "",
        f"同轮实测源码相对原基线的几何平均加速为 **{m['baseline_over_source']['geomean']:.4f}×**，"
        f"独立进程复测 **{r['baseline_over_source']['geomean']:.4f}×**。"
        f"直接比较TTIR版时间/源码版时间为 **{m['ir_over_source']['geomean']:.4f}× / "
        f"{r['ir_over_source']['geomean']:.4f}×**，即性能实际相当。",
        "",
        "这次测量回答‘已知这一优化以后，直接写Triton能否复现’；没有比较只看源码与看TTIR的搜索能力。"
        "没有重新调优tile/warps/stages，也未修改原实验的任何冻结源码或IR。",
        "",
        "| 路径与指标 | 主测7轮 | 独立复测5轮 |",
        "|---|---:|---:|",
        f"| baseline / optimized TTIR | {m['baseline_over_ir']['geomean']:.6f}× | "
        f"{r['baseline_over_ir']['geomean']:.6f}× |",
        f"| baseline / optimized source | {m['baseline_over_source']['geomean']:.6f}× | "
        f"{r['baseline_over_source']['geomean']:.6f}× |",
        f"| optimized TTIR / optimized source | {m['ir_over_source']['geomean']:.6f}× | "
        f"{r['ir_over_source']['geomean']:.6f}× |",
        "",
        "下表为每个shape同轮中位时间（微秒），最后两列由每轮直接配对时间比取中位，"
        "不是用两个历史加速比相除。比值>1表示源码更快；微小偏离1不代表稳定的编译优势。",
        "",
        "| M,N,K | baseline μs | IR μs | source μs | 主测 IR/source | 复测 IR/source |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["records"]:
        t = row["main_us"]
        lines.append(
            f"| {tuple(row['shape'])} | {t['baseline']:.3f} | {t['ir']:.3f} | {t['source']:.3f} | "
            f"{row['main_ratios']['ir_over_source']:.6f}× | {row['repeat_ratios']['ir_over_source']:.6f}× |"
        )
    lines += [
        "",
        "**源码具体改动及编译结果。** [source_v1.py](source_v1.py) 从原冻结设备函数回写："
        "low/high nibble共享scale/zero，完整且对齐的K tile内group由k0/group标量计算再广播，"
        "已证明非负的除余经 `tl.uint32` 表达，必要时cast回int32保持后续索引语义。"
        "FP16反量化舍入、join/permute/reshape、dot及累积顺序保持原样。",
        "",
        "源码路径确实经过普通 `@triton.jit` 前端，三路计时中的source对象是 `ASTSource` 产生的原始CompiledKernel；"
        "它没有被导出TTIR的重编译对象替换。源码对象使用完整Python调用参数，文件对象使用已特化的IR参数；"
        "两者最终CUDA参数ABI、grid和设备计算相同，计时不含Python启动成本。",
        "",
        "前端保留共享与标量group、unsigned除余，正常执行了broadcast/CSE/canonicalize等前端处理。"
        "TTIR文本和完整PTX/cubin受源位置、调试信息等影响不逐字节相同；"
        "所有7点的ELF可执行 `.text` 字节哈希相同，寄存器/shared/spill也相同。"
        "这个结论比‘API能表达’更强：本轮固定环境下已经生成相同设备机器码。"
        "哈希与资源在 [static_analysis.json](static_analysis.json) 及两轮原始JSON中。",
        "",
        "**环境、验证与性能边界。** upstream Triton3.7.0、PyTorch2.12.0+cu130、H100物理GPU0独占。"
        "原7点和冻结配置不变；FP16 A/scale/zero/C、uint8 packed、group128。"
        "每次先证明原baseline source/file与新source/export-file的LLVM/PTX/cubin各自字节一致、"
        "三路全部解析后的backend options一致，之后才验证和计时。"
        "[toolchain.json](toolchain.json)保存包RECORD检查、backend/libtriton/ptxas指纹；RECORD并非远程来源认证。",
        "",
        "主/复测各105项版本/shape/分布检查通过：7点×random/extreme_nibble/zero_scale/zero/constant×3路。"
        "每个候选输出与baseline逐位一致，对FP32 ATen参考沿用rtol=.003、atol=.035，"
        "两个小shape另对CPU FP64，检查NaN sentinel覆盖、64元素尾canary及输入不修改。"
        "memcheck/racecheck各再跑105项数值检查，过滤本实验kernel，"
        "分别 **0 errors** / **0 errors、0 warnings**。",
        "",
        "每轮三路随机顺序测量，另有A/A；每次20ms预分配CUDA graph重放，保留全部样本、执行顺序、"
        "输入seed/SHA256。主/复测所有点A/A中位比都在[0.97,1.03]内。"
        "未锁频、未主动清cache；不包含分配、JIT、模块加载或Python启动，不外推调用端到端成本。",
        "",
        f"直接IR/source的固定shape配对轮次bootstrap95%区间：主测 `{m['ir_over_source']['bootstrap95_fixed_shapes']}`，"
        f"复测 `{r['ir_over_source']['bootstrap95_fixed_shapes']}`。"
        "相同机器码下仍有约千分之一的计时差，可能包括测量次序、模块地址等本轮未隔离因素；"
        "不能据此宣称源码特有优化。",
        "",
        "首个 `main` 尝试在pipeline门槛处退出：错误地把源码TTIR与文件解析后的打印文本逐字节比较。"
        "修正为源码TTIR与冻结输入文件比较，继续严格验证LLVM/PTX/cubin后，完整主测记录为 `main_v2`。"
        "失败日志保留；冻结source_v1和原IR没有修改。",
        "",
        "复现命令见 [README.md](README.md)。证据：[主测](results/main_v2/results.json)、"
        "[复测](results/repeat/results.json)、[memcheck](memcheck.log)、[racecheck](racecheck.log)、"
        "[机器可读汇总](summary.json)。",
        "",
    ]
    (ROOT / "report.md").write_text("\n".join(lines))
    print(json.dumps({"main": m, "repeat": r}, indent=2))


if __name__ == "__main__":
    main()
