"""Recompute all reported numbers from original paired samples and freeze selection."""

import json
import statistics

from common import ROOT, digest, save


def read(path):
    return json.loads((ROOT / path).read_text())


def main():
    main_run = read("results/main/benchmark.json")
    repeat = read("results/repeat/benchmark.json")
    assert len(main_run["points"]) == len(repeat["points"]) == 12
    initial = read("selection_pre_main.json")
    mem = (ROOT / "results/memcheck_v2.log").read_text()
    race = (ROOT / "results/racecheck_v2.log").read_text()
    assert "ERROR SUMMARY: 0 errors" in mem
    assert "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)" in race
    for path in ("accuracy_all", "accuracy_repeat", "memcheck_v2", "racecheck_v2"):
        assert read(f"results/{path}/accuracy.json")["passed"]
    result = dict(
        evidence=(
            "A: frozen 12-point matrix, 7 paired main rounds, "
            "independent seed123 5 paired rounds; accepted speedups limited to six D64 points."
        ),
        groups={},
        sanitizer=dict(
            memcheck="0 errors", racecheck="0 hazards, 0 errors, 0 warnings"
        ),
        sources={
            p: digest(ROOT / p)
            for p in (
                "selection_pre_main.json",
                "results/main/benchmark.json",
                "results/repeat/benchmark.json",
            )
        },
    )
    lines = [
        "# Attention backward dK/dV：直接 TTIR 优化实验",
        "",
        "固定官方 fused dK/dV 提取基线下，`r1_dv_late` 在 D64 的 6 个点稳定加速；D128 保留 baseline。不是完整 Attention backward 的端到端收益。",
        "",
        (
            "基线来自 [官方 Triton v3.7.0 fused-attention](https://github.com/triton-lang/triton/bl"
            "ob/v3.7.0/python/tutorials/06-fused-attention.py)，将 `_attn_bwd` 截在 dQ 段之前，概率恢复、d"
            "P/dS、dK/dV 四个 dot 和因果分段保持原样。`frozen_source.py` 同时保存未截断完整 backward。12 点源码/文件入口 LL"
            "VM、PTX、cubin 全部字节一致；提取输出与完整官方 backward 的 dK/dV 逐位一致。"
        ),
        "",
        (
            "环境：官方 Triton 3.7.0 / PyTorch 2.12.0+cu130 / 独占物理 GPU4 H100 80GB，4 warps、5 stages"
            "、1 CTA。官方固定配置未经逐 shape autotune，尤其 D128 存在明显 spill；结果不证明战胜调优后的其他实现。冻结代码和来源哈希见 [p"
            "rovenance](upstream/provenance.json)，契约见 [optimization_plan.md](optimization_pla"
            "n.md)。"
        ),
        "",
        "## A 级性能结果",
        "",
        (
            "指标为预分配输出的 CUDA graph 设备 kernel 时间；不含 forward、K 预缩放、delta、分配、编译和 Python。每点主测7对/复测"
            "5对，rep=20ms，随机对内顺序；复测为独立进程与新输入seed=123。未锁频，无主动清缓存。所有原始样本和顺序在 [主测](results/main/b"
            "enchmark.json) 与 [复测](results/repeat/benchmark.json)。"
        ),
        "",
        "| 组 | 主测几何均值 | 复测几何均值 | 主测最差 | 最终选择 |",
        "|---|---:|---:|---:|---|",
    ]
    for g in ("d64_c0", "d64_c1", "d128_c0", "d128_c1"):
        vals = []
        for data in (main_run, repeat):
            points = [x for x in data["points"] if x["group"] == g]
            values = [
                statistics.median(
                    p["speedup"] for p in x["candidates"]["r1_dv_late"]["pairs"]
                )
                for x in points
            ]
            vals.append(
                dict(
                    geo=statistics.geometric_mean(values),
                    min=min(values),
                    max=max(values),
                )
            )
        proposed = initial["proposed"][g]
        chosen = (
            proposed
            if proposed != "baseline"
            and all(v["geo"] >= 1.02 and v["min"] >= 0.95 for v in vals)
            else "baseline"
        )
        path = ROOT / "ir" / g / (chosen + ".ttir")
        alias = path.with_name("optimized.ttir")
        if alias.exists():
            assert alias.read_bytes() == path.read_bytes()
        else:
            alias.write_bytes(path.read_bytes())
        result["groups"][g] = dict(
            candidate_tested="r1_dv_late",
            main=vals[0],
            repeat=vals[1],
            selected=chosen,
            selected_ttir_sha256=digest(path),
        )
        lines.append(
            f"| {g} | {vals[0]['geo']:.4f}× | {vals[1]['geo']:.4f}× | {vals[0]['min']:.4f}× | `{chosen}` |"
        )
    d64 = []
    for data in (main_run, repeat):
        values = [
            x["candidates"]["r1_dv_late"]["median_speedup"]
            for x in data["points"]
            if x["shape"][3] == 64
        ]
        d64.append(statistics.geometric_mean(values))
    result["d64_geomean"] = dict(main=d64[0], repeat=d64[1])
    lines += [
        "",
        f"**D64 六点合计：主测 {d64[0]:.4f}×，独立复测 {d64[1]:.4f}×。** 表中 D128 是真实负结果，不能用回退选择的1×代替实测，也不把混合策略的合成数值称为全12点实测。",
        "",
        "| (B,H,N,D,causal) | 主测 baseline μs | 主测候选 μs | 主测加速 | 复测加速 |",
        "|---|---:|---:|---:|---:|",
    ]
    for p, r in zip(main_run["points"], repeat["points"]):
        v = p["candidates"]["r1_dv_late"]
        w = r["candidates"]["r1_dv_late"]
        assert p["shape"] == r["shape"]
        lines.append(
            f"| {tuple(p['shape'])} | {v['baseline_median_us']:.3f} | {v['candidate_median_us']:.3f} | "
            f"{v['median_speedup']:.4f}× | {w['median_speedup']:.4f}× |"
        )
    aa = []
    for data in (main_run, repeat):
        x = [statistics.median(y["ratio"] for y in p["aa"]) for p in data["points"]]
        aa.append((min(x), max(x)))
    result["aa_median_range"] = aa
    lines += [
        "",
        f"A/A 每点3对的中位比范围：主测 {aa[0][0]:.4f}–{aa[0][1]:.4f}，复测 {aa[1][0]:.4f}–{aa[1][1]:.4f}。这些检查只量化本次波动，未证明全部系统噪声被消除。",
        "",
        "## 修改与机制边界",
        "",
        (
            "`r1_dv_late` 直接移动 TTIR 循环中 `pT -> fp16` 的 cast 和 dV dot，使其在 dK dot 后执行；causal 的两"
            "个循环都修改。数学公式、FP16转换位置相对于各自消费者、每个 accumulator 的加法顺序、签名、grid、编译options都保持一致。没有从优化 P"
            "ython kernel 重新导出候选。"
        ),
        "",
        (
            "C 级静态证据见 [static_artifacts.json](results/static_artifacts.json)：D64非因果寄存器255→250"
            "，spill均0；因果255→255，spill均0，SASS指令数1378→1354。所有组 HGMMA 与 barrier 的静态数量不变。机器码和调度改变"
            "，但没有NCU消融证据把约12%的收益归因给某个单一资源量。"
        ),
        "",
        "D128非因果 spill 178→204、因果182→160；后者 spill 下降却仍明显变慢。这是“减少spill不保证加速”的本次反例，不能仅凭寄存器/溢出数字选择方案。",
        "",
        "## 初筛与负结果（B 级）",
        "",
        (
            "7个候选均保留在各组 ir/：unsigned div/rem、delta提前、dV后移、dP提前、load.cg、连续head地址折叠，以及 dV后移+uns"
            "igned 组合。初筛为完整12点但每点仅3对，选拔记录见 [screen](results/screen/benchmark.json) 与 [selecti"
            "on_pre_main.json](selection_pre_main.json)。D64非因果 unsigned 初筛几何0.9246×，dV后移+unsi"
            "gned仅1.0562×，没有继承单项dV后移的全部收益。load.cg接近1×；delta/dP提前在D128退步。没有丢弃不利候选或用重复运行挑最小耗时。"
        ),
        "",
        "## 验证与范围",
        "",
        (
            "初轮224项通过（28输入 × baseline与7候选），独立seed复测56项通过。候选输出均与baseline逐位相同；FP64原始注意力数学参考门槛固定"
            "为rtol=0.03, atol=0.008。包含N128/256/1024/2048、causal/noncausal、D64/128、zero_dO、con"
            "stant和large QK；每轮4个小例另做CPU FP64 autograd。检查输入不修改、NaN sentinel完全覆盖及有限性。"
        ),
        "",
        (
            "最终两个版本各在完整28输入运行memcheck/racecheck：[memcheck](results/memcheck_v2.log) **0 error"
            "s**；[racecheck](results/racecheck_v2.log) **0 hazards / 0 errors / 0 warnings**，"
            "各含56项数值检查。早先filter参数写成`regex:`被工具拒绝，未执行kernel；失败原日志保留，修正为`regex=`后重新运行，未覆盖原结果。"
        ),
        "",
        (
            "仅覆盖连续FP16、N为128倍数、无dropout、无别名、预先具备LSE/delta和预缩放K的dK/dV子算子。固定配置结果不外推ragged序列、其他D"
            "/dtype、其他GPU或整个训练step。这是直接TTIR候选实验，没有构造文章中的A/B/C盲测，因此不能据此证明TTIR信息优于源码信息；dV调度变换也可"
            "能回写Triton源码表达。"
        ),
        "",
        (
            "复现入口见 [README.md](README.md)。冻结选择与IR哈希见 [selection.json](selection.json)，编译器/lib"
            "triton/ptxas哈希见 [toolchain.json](results/toolchain.json)。"
        ),
    ]
    save(ROOT / "selection.json", result)
    (ROOT / "report_round1.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
