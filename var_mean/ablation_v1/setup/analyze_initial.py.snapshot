"""Derive directly paired conditional effects from randomized complete timing blocks."""

import argparse
import json
import math
import os
import re
import statistics
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(path):
    return json.loads(Path(path).read_text())


def contrast(data, before, after, indices):
    ratios = []
    rng = np.random.default_rng(42)
    sampled = []
    per_shape = []
    for index in indices:
        row = data["records"][index]
        values = np.array(
            [b["times_us"][before] / b["times_us"][after] for b in row["blocks"]]
        )
        ratios.append(float(np.median(values)))
        sampled.append(
            np.median(
                rng.choice(values, size=(2000, len(values)), replace=True), axis=1
            )
        )
        per_shape.append(
            dict(
                shape=row["shape"],
                paired_ratios=values.tolist(),
                median=ratios[-1],
                before_us=statistics.median(
                    b["times_us"][before] for b in row["blocks"]
                ),
                after_us=statistics.median(b["times_us"][after] for b in row["blocks"]),
            )
        )
    geo = math.exp(statistics.mean(map(math.log, ratios)))
    samples = np.exp(np.log(np.stack(sampled)).mean(axis=0))
    return dict(
        before=before,
        after=after,
        points=len(indices),
        geomean=geo,
        speedup_percent=(geo - 1) * 100,
        latency_reduction_percent=(1 - 1 / geo) * 100,
        ci95=np.quantile(samples, [0.025, 0.975]).tolist(),
        worst=min(ratios),
        best=max(ratios),
        per_shape=per_shape,
    )


def analyze(data, manifest):
    assert data["valid"] and len(data["records"]) == 30
    assert data["canonical_cubin_identical"] and data["canonical_ptx_identical"]
    for bits, item in manifest["variants"].items():
        assert data["identities"][bits]["ttir_sha256"] == item["sha256"]
    for record in data["records"]:
        assert len(record["blocks"]) == data["rounds"]
        for block in record["blocks"]:
            assert sorted(block["order"]) == sorted(manifest["variants"])
            assert all(
                value > 0 and math.isfinite(value)
                for value in block["times_us"].values()
            )
    groups = {
        "all": list(range(30)),
        "short": [i for i, r in enumerate(data["records"]) if r["shape"][1] <= 1024],
        "long": [i for i, r in enumerate(data["records"]) if r["shape"][1] > 1024],
    }
    comparisons = {
        "all_variants": [("000", bits) for bits in manifest["variants"]],
        "edges": [
            (bits, bits[:factor] + "1" + bits[factor + 1 :])
            for factor in range(3)
            for bits in manifest["variants"]
            if bits[factor] == "0"
        ],
    }
    return {
        group: {
            kind: {
                f"{before}->{after}": contrast(data, before, after, indices)
                for before, after in pairs
            }
            for kind, pairs in comparisons.items()
        }
        for group, indices in groups.items()
    }


def fmt(result):
    lo, hi = result["ci95"]
    return f"{result['geomean']:.4f}× [{lo:.4f}, {hi:.4f}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "report_final.md")
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    results = args.results_dir
    manifest = load(HERE / "manifest.json")
    main_data = load(results / "main.json")
    repeat = load(results / "repeat.json")
    assert main_data["gpu_uuid"] == repeat["gpu_uuid"]
    assert main_data["runner_sha256"] == repeat["runner_sha256"]
    assert main_data["manifest_sha256"] == repeat["manifest_sha256"]
    assert [r["shape"] for r in main_data["records"]] == [
        r["shape"] for r in repeat["records"]
    ]
    source = load(results / "source_feasibility_v3.json")
    assert source["passed"]
    for filename in [
        "accuracy.json",
        "repeat_accuracy.json",
        "memcheck_accuracy.json",
        "racecheck_accuracy.json",
    ]:
        record = load(results / filename)
        assert record["passed"]
    assert "ERROR SUMMARY: 0 errors" in (results / "memcheck.log").read_text()
    assert (
        "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
        in (results / "racecheck.log").read_text()
    )
    first, second = analyze(main_data, manifest), analyze(repeat, manifest)
    derived = dict(main=first, repeat=second, source_variants=source["variants"])
    result_path = results / "derived.json"
    if result_path.exists():
        assert load(result_path) == derived
    else:
        result_path.write_text(json.dumps(derived, indent=2) + "\n")
    standalone = {"A": "000->100", "B": "000->010", "C": "000->001"}
    removal = {"A": "011->111", "B": "101->111", "C": "110->111"}
    all_first = first["all"]
    full = all_first["all_variants"]["000->111"]
    lines = [
        "# var_mean 三项改动的完整消融实验",
        "",
        f"完整组合对冻结基线的30点几何加速为 **{fmt(full)}**，独立进程复测 "
        f"**{fmt(second['all']['all_variants']['000->111'])}**。",
        "这是相同输入、预分配输出的 CUDA graph kernel 时间。下表的百分比表示吞吐意义的加速比增量；",
        "耗时降幅为 `1-1/加速比`，两者不要混用。方括号是配对块条件 bootstrap 95% 区间。",
        "",
        "| 改动 | 单独加入基线：加速比与区间 | 单独加速增量 | 最终组合中加入该项：加速比与区间 | Triton源码可实现 |",
        "|---|---:|---:|---:|---|",
    ]
    descriptions = {
        "A": "A：短行展开与输入复用",
        "B": "B：仅长行循环1024→256",
        "C": "C：N的16整除提示",
    }
    for factor in "ABC":
        one = all_first["edges"][standalone[factor]]
        marginal = all_first["edges"][removal[factor]]
        lines.append(
            f"| {descriptions[factor]} | {fmt(one)} | {one['speedup_percent']:+.2f}% | {fmt(marginal)} | 是，见编译与运行探针 |"
        )
    ac = first["all"]["all_variants"]["000->101"]
    ac_repeat = second["all"]["all_variants"]["000->101"]
    b_marginal = first["all"]["edges"]["101->111"]
    lines += [
        "",
        f"本轮反例：仅A+C为 **{fmt(ac)}**，独立复测 **{fmt(ac_repeat)}**，比三项全开更快。",
        f"在A+C基础上加入B得到{b_marginal['geomean']:.4f}×，"
        f"即耗时增加{(1/b_marginal['geomean']-1)*100:.2f}%。",
        "这说明此前完整组合的正收益不能证明每一项在最终上下文中都有正贡献。旧optimized与其报告保持原样。",
    ]
    lines += [
        "",
        "“最终组合中加入该项”直接比较其余两项已打开的版本和完整版本，等价于去掉该项后付出的性能代价。",
        "单项收益依赖另外两项的状态与shape，不能将三个百分比相加，也不能把上述单项加速比相乘当成最终加速。",
        "",
        "## 改动定义与实验边界",
        "",
        "- A：N≤128/512/1024分别使用对应宽度的单tile，一次load后复用输入计算mean和中心化方差。",
        "- B：仅N>1024时采用256宽两遍中心化循环；N≤1024仍保留原1024宽循环。",
        "  B单独打开需要运行时分支，该实际分支成本也包含在测量里。它与此前“所有N都改为256”的候选不是同一版本。",
        "- C：为N添加`tt.divisibility=16`前提；输入契约和主机包装已检查N可被16整除。提示不改变输入数值。",
        "- 三位编码顺序为ABC。000的PTX和完整cubin与原baseline字节一致；111的TTIR与原optimized字节一致。",
        "- 继承`var_mean-row-f32-v3`：连续FP32二维输入，最后一维按行归约，correction=0/1；",
        "  不包含全张量两阶段、FP16/BF16、非连续输入和API端到端时间。未改已有最优选择或FlagGems算子源码。",
        "- 30shape固定为M∈{1,17,128,1024,4096}×N∈{128,512,1040,4096,8192,16384}，性能correction=1，等权。",
        "",
        "## 全部8个组合",
        "",
        "| ABC | 打开的改动 | 30点主测 | 独立复测 | 短行主测(N≤1024) | 长行主测(N>1024) | registers/spill/shared bytes |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for bits in manifest["variants"]:
        key = f"000->{bits}"
        resource = main_data["identities"][bits]
        labels = "+".join(f for f, bit in zip("ABC", bits) if bit == "1") or "baseline"
        lines.append(
            f"| {bits} | {labels} | {fmt(first['all']['all_variants'][key])} | "
            f"{fmt(second['all']['all_variants'][key])} | "
            f"{first['short']['all_variants'][key]['geomean']:.4f}× | "
            f"{first['long']['all_variants'][key]['geomean']:.4f}× | "
            f"{resource['n_regs']}/{resource['n_spills']}/{resource['shared']} |"
        )
    lines += [
        "",
        "## 12个直接配对的条件效应",
        "",
        "每一条比较只翻转一位，before/after两版的耗时在同一个随机完整块内配对。",
        "短行10点，长行20点；每点先取块内时间比的中位数，再按shape取等权几何平均。",
        "即使某一shape没有进入改动的分支，整个kernel的控制流与编译调度仍可能变化，所以短/长子组的效应不必局限于其逻辑分支。",
        "",
        "| 加入项 | before→after | 全30点 | 独立复测 | 短行 | 长行 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key, value in all_first["edges"].items():
        before, after = key.split("->")
        factor = next(f for f, a, b in zip("ABC", before, after) if a != b)
        lines.append(
            f"| {factor} | {key} | {fmt(value)} | {fmt(second['all']['edges'][key])} | "
            f"{first['short']['edges'][key]['geomean']:.4f}× | {first['long']['edges'][key]['geomean']:.4f}× |"
        )
    lines += [
        "",
        "## 上游Triton能否实现",
        "",
        "三项均在本机固定的官方Triton3.7.0上编译、运行并通过CPU FP64比较，不是仅依据语法推测。",
        "源码算法在[source_probes.py](source_probes.py)，对齐包装在[source_probes_v3.py](source_probes_v3.py)，"
        "最终记录见[source_feasibility_v3.json](results/source_feasibility_v3.json)。",
        "",
        "| 改动 | 源码写法 | 已验证内容 |",
        "|---|---|---|",
        "| A | 运行时`if N<=128/512/1024`，各分支使用constexpr宽度的`tl.arange`、一次`tl.load`与两次`tl.sum` | 所有分支和尾部掩码运行正确 |",
        "| B | 长行分支调用`BLOCK=256`的`for off in range(0,N,BLOCK)`两遍归约，短行`BLOCK=1024` | 同一运行时N签名与int64行索引可表达 |",
        "| C | 允许N参与JIT对齐特化，保持N为运行时参数 | 入口参数上的提示在本例被丢弃，正确方式见下文 |",
        "",
        "C的三个探针结果需要区分：",
        "",
        "1. 在被do_not_specialize禁用特化的入口N上调用`tl.multiple_of(N,16)`：数值正确，但C开关两侧TTIR/PTX/cubin相同，提示未保留。",
        "2. 对`N.to(tl.int64)`调用`tl.multiple_of`：cast上保留了整除属性，但没有恢复本例目标向量化load。",
        '3. 使用`@triton.jit(do_not_specialize=["M","correction"])`，从禁用列表移除N：',
        "   编译器在运行时`%N: i32`参数上生成与TTIR改写相同的`tt.divisibility=16`，并生成v4向量化load。",
        "   合法N均≥16且可被16整除，共享一个对齐类别，N没有变成constexpr，也不是按N值逐一特化。",
        "",
        "前两组记录分别保留在[source_feasibility.json](results/source_feasibility.json)与"
        "[source_feasibility_v2.json](results/source_feasibility_v2.json)。",
        "因此C能在Triton层实现，但本例不能把一个被编译器丢弃的提示写法当作实现证据。",
        "本地固定版本的`triton.language.multiple_of`文档及core.py哈希保存在探针记录；不能仅凭文档推断该提示在每个位置都有效。",
        "在线文档检索接口本次不可用，因此以实际安装的固定版本源码、导出TTIR和设备执行为依据。",
        "[属性与向量化审计](results/source_audit_final.json)逐一核对三组导出TTIR/PTX及其哈希，关键源码导出保存在source_ir。",
        "这里证明源码可实现相应算法与提示，不声称所有源代码写法都会产生相同TTIR/机器码或相同性能。",
        "源码探针没有参与本报告的TTIR性能计时。",
        "",
        "| 源码ABC | N保持运行时参数 | 导出TTIR中的整除属性（摘要） |",
        "|---|---|---|",
    ]
    for bits, variant in source["variants"].items():
        attrs = "；".join(
            re.findall(r"%N: i32(?: \{[^}]+\})?", line)[0]
            for line in variant["divisibility_lines"]
            if "%N: i32" in line
        )
        attrs = attrs.replace("|", "\\|").replace("`", "")
        lines.append(f"| {bits} | {variant['n_is_runtime']} | `{attrs}` |")
    lines += [
        "",
        "## 30个shape的逐点结果",
        "",
        "表中为每点主测配对加速比中位数。完整绝对耗时、7轮执行顺序和各条件比较的原始比值见JSON。",
        "",
        "| M | N | A | B | C | AB | AC | BC | ABC |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = ["100", "010", "001", "110", "101", "011", "111"]
    for i, row in enumerate(main_data["records"]):
        values = [
            first["all"]["all_variants"][f"000->{bits}"]["per_shape"][i]["median"]
            for bits in order
        ]
        lines.append(
            f"| {row['shape'][0]} | {row['shape'][1]} | "
            + " | ".join(f"{v:.4f}×" for v in values)
            + " |"
        )
    controls = load(results / "controls.json")
    repeat_controls = load(results / "repeat_controls.json")
    assert controls["valid"] and repeat_controls["valid"]
    aa1 = {tuple(r["shape"]): r for r in controls["records"] if r["name"] == "aa"}
    aa2 = {
        tuple(r["shape"]): r for r in repeat_controls["records"] if r["name"] == "aa"
    }
    drift = {
        str(shape): aa2[shape]["baseline_us"] / r["baseline_us"]
        for shape, r in aa1.items()
    }
    lines += [
        "",
        "## 验证、计时和限制",
        "",
        f"设备为物理GPU4 H100，UUID `{main_data['gpu_uuid']}`。官方Triton3.7.0，独立venv Python3.12.3，完整环境与工具哈希见pipeline。",
        "两版和所有组合均为4warps、3stages、1CTA、FP fusion开启，原生TTIR文件编译与同一CompiledKernel启动方式；",
        "baseline源码路径与文件路径的30点LLVM/PTX/cubin和两项输出一致。",
        "同shape、同输入、每一块随机排列全部8版，主测7块、独立进程seed123复测5块，每版rep=20ms；",
        "每版先10次启动和50ms不计时graph预热，编译/加载/预热均不计入时间。计时按物理UUID加协作锁；",
        "负载期间采样功耗、时钟、温度和GPU进程，未发现本卡外部进程。采样不能排除所有瞬时干扰。",
        "未主动锁频、未清缓存，不是冷缓存或端到端数据。全部shape以前暴露，属于消融和回归确认，不是新holdout。",
        "区间在固定shape内重采样整个块的配对比值（2000次），再汇总等权几何均值；不覆盖全部系统误差和选择偏差。",
        f"A/A几何比为{controls['summary']['aa']['geomean']:.6f}/{repeat_controls['summary']['aa']['geomean']:.6f}，",
        (
            f"重复两次启动对照为{controls['summary']['double']['geomean']:.6f}/"
            f"{repeat_controls['summary']['double']['geomean']:.6f}，应接近0.5。"
        ),
        f"3代表点跨进程baseline耗时比repeat/main：`{drift}`。",
        "",
        "TTIR两seed各5076项数值检查；CPU FP64是主参考，variance rtol=2e-4/atol=2e-6，mean rtol=2e-5/atol=2e-5，",
        "原生GPU FP32差异诊断保留。每次重置输出NaN并检查前后canary，边界与压力分布沿用原实验。",
        f"源码8组合合计{len(source['records'])}项数值检查，保持原签名、运行时N和int64行索引。",
        "memcheck与racecheck各256项数值检查（8宽度×2分布×2correction×8版本），覆盖128/512/1024三个短分支及长行；",
        "均0errors，racecheck为0hazards/0warnings。仍只证明实际执行的子集。",
        "",
        "早期生成器遗漏分支局部张量常量，编译/精确身份门槛阻止了其计时；失败文件保存在failed_generation_v1。",
        "GPU5在计时前被占用，入口拒绝启动；切换GPU4后两轮未混卡，保留setup中的失败日志与原计划。",
        "",
        "## 证据与复现",
        "",
        "- [主测原始完整块](results/main.json)、[独立进程复测](results/repeat.json)、[全部派生统计](results/derived.json)",
        "- [manifest](manifest.json)、[源码可表达性](results/source_feasibility_v3.json)、"
        "[memcheck](results/memcheck.log)、[racecheck](results/racecheck.log)",
        "- `CUDA_VISIBLE_DEVICES=4 bash var_mean/ablation_v1/run.sh` 从仓库根目录运行，生成新结果目录。",
        "- 原来的optimized.ttir、baseline.ttir和r2选择保持原哈希。此处没有宣称新的更优候选，也没有把不同进程的候选分数混排。",
    ]
    text = "\n".join(lines) + "\n"
    relative_results = os.path.relpath(results, args.output.parent)
    text = text.replace("(results/", f"({relative_results}/")
    for name in ["source_probes.py", "source_probes_v3.py", "manifest.json"]:
        text = text.replace(
            f"({name})", f"({os.path.relpath(HERE / name, args.output.parent)})"
        )
    if args.output.exists():
        assert args.output.read_text() == text
    else:
        args.output.write_text(text)
    print("complete", fmt(full))


if __name__ == "__main__":
    main()
