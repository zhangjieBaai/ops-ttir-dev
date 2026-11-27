"""Qualify expanded30 TTIR winners against both parent and original baseline."""

import json
import math
import random
import statistics
from pathlib import Path

from support import digest, freeze, save

ROOT = Path(__file__).resolve().parents[1]


def boot(records, name, metric):
    rng = random.Random(907)
    draws = []
    for _ in range(2000):
        values = []
        for r in records:
            samples = r["timing"]["samples"]
            values.append(
                statistics.median(rng.choice(samples)[metric][name] for _ in samples)
            )
        draws.append(math.exp(statistics.mean(math.log(v) for v in values)))
    draws.sort()
    return [draws[50], draws[1950]]


def report(op):
    folder = ROOT / op / "round2/expanded"
    main_name, repeat_name = (
        ("main", "repeat")
        if op == "norm_quant"
        else ("main_isolated", "repeat_isolated")
    )
    main = json.loads((folder / f"results/{main_name}/results.json").read_text())
    repeat = json.loads((folder / f"results/{repeat_name}/results.json").read_text())
    selection = json.loads((folder / "selection_pre_main.json").read_text())
    chosen = selection["name"]
    assert len(main["records"]) == len(repeat["records"]) == 30
    assert (
        main["passed"]
        and repeat["passed"]
        and main["aa_passed"]
        and repeat["aa_passed"]
    )
    assert chosen in main["validated_names"] and chosen in repeat["validated_names"]
    assert not main["failures"] and not repeat["failures"]
    for dataset in (main, repeat):
        increment = dataset["summary"][chosen]["parent_over"]
        assert increment["geomean"] >= 1.02 and increment["worst"] >= 0.95
    for tool, marker in (
        ("memcheck", "ERROR SUMMARY: 0 errors"),
        ("racecheck", "0 errors, 0 warnings"),
    ):
        assert marker in (folder / f"results/{tool}.log").read_text()
        checked = json.loads((folder / f"results/{tool}/results.json").read_text())
        assert (
            checked["passed"]
            and chosen in checked["validated_names"]
            and not checked["failures"]
        )
    stages = ["kernel"] if op == "norm_quant" else ["local", "merge"]
    kernels = []
    for a, b in zip(main["records"], repeat["records"]):
        assert a["shape"] == b["shape"]
        for stage in stages:
            meta = a["kernels"][chosen][stage]
            assert (
                meta["ir_sha256"]
                == b["kernels"][chosen][stage]["ir_sha256"]
                == digest(meta["ir_path"])
            )
            path = Path(meta["ir_path"])
            freeze(path.parent / "optimized.ttir", path.read_text())
            kernels.append(
                {
                    "shape": a["shape"],
                    "stage": stage,
                    "path": meta["ir_path"],
                    "ir_sha256": meta["ir_sha256"],
                }
            )
    if op == "norm_quant":
        extra = json.loads((folder / "results/extreme_stress.json").read_text())
        assert extra["passed"] and len(extra["records"]) == 90
    result = {
        "candidate": chosen,
        "accepted": True,
        "shape_count": 30,
        "main": main["summary"][chosen],
        "repeat": repeat["summary"][chosen],
        "timing_runs": {"main": main_name, "repeat": repeat_name},
        "parent_definition": "stream_v1" if op == "norm_quant" else "ordered_carry_v3",
        "evidence_level": "A, expanded fixed30 matrix; C for resources; no source-port performance tested this round",
        "bootstrap95_incremental": {
            "main": boot(main["records"], chosen, "parent_over"),
            "repeat": boot(repeat["records"], chosen, "parent_over"),
        },
        "aa_ranges": {
            label: [
                min(r["timing"]["aa_median"] for r in d["records"]),
                max(r["timing"]["aa_median"] for r in d["records"]),
            ]
            for label, d in (("main", main), ("repeat", repeat))
        },
        "sanitizer": {"memcheck": "0 errors", "racecheck": "0 errors, 0 warnings"},
        "kernel_mapping": kernels,
        "evidence": {
            label: digest(folder / f"results/{name}/results.json")
            for label, name in (("main", main_name), ("repeat", repeat_name))
        },
    }
    save(folder / "selection.json", result)
    save(ROOT / op / "round2/summary.json", result)
    title = "Residual RMSNorm + 动态INT8" if op == "norm_quant" else "大N partial Top-K"
    lines = [
        f"# {title}：第二轮TTIR优化，扩展30点",
        "",
        f"最终采用 `{chosen}`，对上一轮parent的增量几何平均 **{result['main']['parent_over']['geomean']:.4f}× / "
        f"{result['repeat']['parent_over']['geomean']:.4f}×**；",
        f"对原始baseline的累计几何平均 **{result['main']['baseline_over']['geomean']:.4f}× / "
        f"{result['repeat']['baseline_over']['geomean']:.4f}×**。",
        "两个数依次为主测/独立复测，全部为本轮同场配对。A级仅限冻结30点和固定配置。",
        "",
        "本轮所有新候选都直接编辑TTIR；原始源码、baseline、上一轮optimized和source_port不变。",
        "新选中别名与哈希映射位于expanded/selection.json和各新IR目录，不覆盖父目录历史选择。",
        "",
        "## 30点、配置与验证范围",
        "",
    ]
    if op == "norm_quant":
        lines += [
            "N={257,1024,4096,4097,8192,16384}，基本rows={1,32,128,512,2048}；",
            "为保留原点，将(32,257)换成(64,257)、(512,4097)换成(513,4097)。原8点全部保留。",
            "沿用原width-family配置：N4097为4warps，其余8warps，3stages。新增点没有autotune。",
            "FP16 X/R/W、FP32 Z/scale、INT8输出，Q相对FP64最多差1 code，scale与重构门槛不变。",
            "额外90项CPU FP64极端检查覆盖最大合法幅度、零权重、最小FP16 subnormal、精确残差抵消、spike，全部通过。",
        ]
    else:
        lines += [
            "rows={1,4,16,64,128}，配(N,K)={(65536,8),(65537,8),(131072,8),(262144,8),(262144,16),(1048576,8)}，",
            "共30点，含原6点和尾块仅1个元素的边界。原两阶段chunk1024、4warps/3stages不变。",
            "固定shape族，新增点无autotune。有限FP32、stable原始index、values数值精确（±0等价）合同不变，含subnormal/最低finite压力。",
        ]
    lines += [
        "",
        "原小矩阵结果仅为pilot；最终候选是在30点完整初筛后冻结，独立复测不重新选择。",
        "主测7轮20ms、独立seed123复测5轮20ms；同输入预分配完整设备序列CUDA graph，逐轮随机顺序及parent A/A。",
        "环境为官方Triton3.7.0/H100；无锁频、无主动清cache。不计编译、加载、参考或分配。",
        "全部30映射重新核对原源码与baseline文件LLVM/PTX/cubin一致，三路完整backend options相同。",
        "",
        "| (shape) | baseline μs | parent μs | candidate μs | 主测增量 | 复测增量 | 复测累计 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for a, b in zip(main["records"], repeat["records"]):
        t, rt = a["timing"], b["timing"]
        lines.append(
            f"| {tuple(a['shape'])} | {t['median_us']['baseline']:.3f} | {t['median_us']['parent']:.3f} | "
            f"{t['median_us'][chosen]:.3f} | {t['parent_over'][chosen]:.4f}× | "
            f"{rt['parent_over'][chosen]:.4f}× | {rt['baseline_over'][chosen]:.4f}× |"
        )
    lines += [
        "",
        "时间列为各自中位数，比值列为同轮配对比的中位数，不能简单用两列时间相除替代。",
        f"最差增量：主测{result['main']['parent_over']['worst']:.4f}×，复测{result['repeat']['parent_over']['worst']:.4f}×。",
        f"固定矩阵配对bootstrap95：{result['bootstrap95_incremental']}；仅反映已保存样本波动。",
        "",
        "## 改动、机制与负结果",
        "",
    ]
    if op == "norm_quant":
        lines += [
            "每行只计算一次 `precise_div(1,scale)`，再以乘法替代向量逐元素precise_div；Z的地址和store提前到统计归约前。",
            "这改变除法的舍入路径，沿用原±1 code/scale/重构合同检查，不能称严格INT8逐位等价。",
            "提前写Z在无别名条件下合法；实际编译的存活区间、寄存器和标量/向量除法工作量改变。",
            "例如(1,4097)父版本117 regs降到96，shared仍4096B；不是通过减少kernel数取得收益。",
            "reshape分层sum/max的fold256明显退步，early_z单项在部分点也回退；所有初筛样本保留，B级，不能当最终效果。",
            "源码也可用一次tl.div_rn、广播乘法和移动tl.store表达；本轮没有再次运行源码回写性能对照。",
        ]
    else:
        lines += [
            "去掉每轮把已选index改成INTMAX的向量更新，保留原始ids；是否已选由score-key哨兵控制。",
            "尾块耗尽时显式输出INTMAX，保持原两阶段候选约定；然后直接在TTIR中展开K次选择。",
            "固定ids使更多索引能在需要时重新生成，减少动态循环状态；展开改变代码体积和调度，单项贡献由初筛消融观察。",
            "uint64 packed score/index方案虽然把两次reduce合成一次，却比parent慢；64位归约失去部分32位硬件归约优势是机制线索。",
            "旧unroll文件因数字开头SSA名字不合法解析失败，未运行；修正版v2和原失败文件都保留。",
            "源码可保留只读ids并用tl.static_range(K)表达展开；本轮未再次测源码回写。",
            "第一次main/repeat出现主控调度重叠，两轮全部排除并保留，最终只使用main_isolated/repeat_isolated。",
        ]
    screen = json.loads((folder / "results/screen/results.json").read_text())
    lines += ["", "| 候选 | 30点初筛增量geo（B） | 最差增量 |", "|---|---:|---:|"]
    for name, value in screen["summary"].items():
        if name not in ("baseline", "parent"):
            lines.append(
                f"| {name} | {value['parent_over']['geomean']:.4f}× | {value['parent_over']['worst']:.4f}× |"
            )
    count = sum(
        len(v)
        for d in (main, repeat)
        for row in d["records"]
        for v in row["validation"].values()
    )
    lines += [
        "",
        "## 完整性、sanitizer及复现",
        "",
        f"主/复测共{count}项版本/分布检查通过；全部候选完整数值验证先于筛选计时。",
        "memcheck与racecheck均通过：0 errors，racecheck 0 warnings。工具使用kernel名称过滤，覆盖全部30映射；",
        "压力检查与sanitizer内候选验证沿用原脚本的小rows策略，实际每项rows/N在原始JSON列出，不宣称所有压力分布都用最大batch。",
        f"[主测](expanded/results/{main_name}/results.json) · "
        f"[独立复测](expanded/results/{repeat_name}/results.json) · ",
        "[冻结映射](expanded/manifest.json) · [选择](expanded/selection.json) · "
        "[初筛](expanded/results/screen/results.json) · ",
        "[memcheck](expanded/results/memcheck.log) · [racecheck](expanded/results/racecheck.log)",
        "",
        "```bash",
        f"CUDA_VISIBLE_DEVICES={6 if op == 'norm_quant' else 7} "
        "/root/venvs/ir-ops-dev/bin/python complex_suite/round2_runner.py \\",
        f"  --op {op} --expanded --run new_repeat --names {chosen} --rounds 5 --rep 20 --seed 234",
        "```",
        "",
        "在仓库根执行，输出目录必须不存在；候选再生使用各round2/make_candidates.py --expanded，已有IR不同会拒绝覆盖。",
        "累计加速相对本轮扩展矩阵的原baseline，不与上轮7/8点平均混算。",
    ]
    (ROOT / op / "round2/report.md").write_text("\n".join(lines) + "\n")
    return result


if __name__ == "__main__":
    save(
        ROOT / "complex_suite/round2_norm_topk_summary.json",
        {op: report(op) for op in ("norm_quant", "partial_topk")},
    )
