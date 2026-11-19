"""Report source-native comparison with all shapes, paired uncertainty and validity flags."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def geomean(values):
    return math.exp(statistics.mean(math.log(x) for x in values))


def paired_summary(records):
    rng = np.random.default_rng(19)
    logs = []
    for row in records:
        pairs = np.asarray([p["speedup"] for p in row["pairs"]])
        logs.append(
            np.log(
                np.median(
                    pairs[rng.integers(0, len(pairs), size=(2000, len(pairs)))], axis=1
                )
            )
        )
    draws = np.exp(np.mean(logs, axis=0))
    ratios = [r["speedup"] for r in records]
    return dict(
        geometric_mean=geomean(ratios),
        arithmetic_mean=statistics.mean(ratios),
        ci95=[float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        min=min(ratios),
        max=max(ratios),
        faster=sum(v > 1 for v in ratios),
        invalid=[r["n"] for r in records if not r["valid"]],
    )


def load(name):
    path = RESULTS / name / "summary.json"
    d = json.loads(path.read_text())
    assert d["passed"] and [r["n"] for r in d["records"]] == list(range(256, 4097, 128))
    for r in d["records"]:
        assert all(x["passed"] for x in r["correctness"].values())
    return d


def repair(original, name):
    path = RESULTS / name / "summary.json"
    data = json.loads(path.read_text())
    expected = [r["n"] for r in original["records"] if not r["valid"]]
    assert [r["n"] for r in data["records"]] == expected
    assert data["seed"] == original["seed"] and data["rounds"] == 7
    replacements = {r["n"]: r for r in data["records"]}
    rows, source_map = [], {}
    for old in original["records"]:
        new = replacements.get(old["n"], old)
        assert new["input_hashes"] == old["input_hashes"]
        assert new["tutorial_config"] == old["tutorial_config"]
        assert (
            new["tutorial_kernel"]["stages"]["cubin"]
            == old["tutorial_kernel"]["stages"]["cubin"]
        )
        assert [k["stages"]["cubin"] for k in new["flaggems_kernels"]] == [
            k["stages"]["cubin"] for k in old["flaggems_kernels"]
        ]
        assert all(v["passed"] for v in new["correctness"].values())
        rows.append(new)
        source_map[str(old["n"])] = name if old["n"] in replacements else "original"
    return rows, source_map, data["records"]


def main():
    global RESULTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    RESULTS = args.results
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    fm, fr, um, ur = [
        load(n)
        for n in [
            "flagtree_main",
            "flagtree_repeat",
            "upstream_main",
            "upstream_repeat",
        ]
    ]
    reference_uuid = json.loads(
        (RESULTS / "flagtree_main/environment.json").read_text()
    )["gpu_uuid"]
    for name in [
        "flagtree_main",
        "flagtree_repeat",
        "upstream_main",
        "upstream_repeat",
    ]:
        d = json.loads((RESULTS / name / "environment.json").read_text())
        assert d["gpu_uuid"] == reference_uuid
        assert (
            d["source_sha256"]
            == hashlib.sha256(
                Path(
                    "/root/FlagGems/src/flag_gems/runtime/backend/_nvidia/hopper/ops/mm.py"
                ).read_bytes()
            ).hexdigest()
        )
    manifest = json.loads((HERE.parent / "baseline.json").read_text())
    for data in [um, ur]:
        for row in data["records"]:
            config = manifest["families"][manifest["shape_to_family"][str(row["n"])]]
            assert (
                row["tutorial_kernel"]["stages"]["cubin"]
                == config["autotune_cubin_sha256"]
            )
    for f, u in [(fm, um), (fr, ur)]:
        assert f["seed"] == u["seed"]
        for x, y in zip(f["records"], u["records"]):
            assert x["input_hashes"] == y["input_hashes"] and x["n"] == y["n"]
    for x, y in zip(fm["records"], fr["records"]):
        assert [k["name"] for k in x["flaggems_kernels"]] == [
            "mm_kernel_general_host_tma"
        ]
        assert [k["name"] for k in y["flaggems_kernels"]] == [
            "mm_kernel_general_host_tma"
        ]
        assert x["tutorial_config"] == y["tutorial_config"]
        assert (
            x["tutorial_kernel"]["stages"]["cubin"]
            == y["tutorial_kernel"]["stages"]["cubin"]
        )
        assert [v["stages"]["cubin"] for v in x["flaggems_kernels"]] == [
            v["stages"]["cubin"] for v in y["flaggems_kernels"]
        ]
    main_summary, repeat_summary = [paired_summary(d["records"]) for d in [fm, fr]]
    repaired_main, main_sources, main_rechecks = repair(fm, "repair_main")
    repaired_repeat, repeat_sources, repeat_rechecks = repair(fr, "repair_repeat")
    repaired_summaries = [
        paired_summary(rows) for rows in [repaired_main, repaired_repeat]
    ]
    cross = []
    for f, u in [(fm, um), (fr, ur)]:
        ratios = [
            y["tutorial_us"] / x["flaggems_us"]
            for x, y in zip(f["records"], u["records"])
        ]
        cross.append(
            dict(
                geometric_mean=geomean(ratios),
                arithmetic_mean=statistics.mean(ratios),
                min=min(ratios),
                max=max(ratios),
                faster=sum(v > 1 for v in ratios),
                invalid_union=sorted(
                    {r["n"] for d in [f, u] for r in d["records"] if not r["valid"]}
                ),
            )
        )
    tables = [
        "| M=N=K | 教程/FlagTree µs | FlagGems/FlagTree µs | 配对加速比 | 独立复测 | upstream教程 µs | 跨环境比值 | 有效性（主/复测） |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for i, (x, y, u) in enumerate(zip(fm["records"], fr["records"], um["records"])):
        status = (
            ("有效" if x["valid"] else "波动超限")
            + "/"
            + ("有效" if y["valid"] else "波动超限")
        )
        tables.append(
            f"| {x['n']} | {x['tutorial_us']:.3f} | {x['flaggems_us']:.3f} | {x['speedup']:.3f}× | "
            f"{y['speedup']:.3f}× | {u['tutorial_us']:.3f} | {u['tutorial_us']/x['flaggems_us']:.3f}× | {status} |"
        )
    env = json.loads((RESULTS / "flagtree_main/environment.json").read_text())
    fgconfigs = {
        tuple(sorted(r["flaggems_kernels"][0]["constants"].items()))
        for r in fm["records"]
    }
    tfconfigs = {tuple(sorted(r["tutorial_config"].items())) for r in fm["records"]}
    bad = bool(main_summary["invalid"] or repeat_summary["invalid"])
    validity_statement = (
        "**存在波动超限点，以上是完整31点的观测汇总，不能作为全部31点均稳定通过的加速保证。**"
        if bad
        else "两轮全部31点A/A控制和漂移检查均通过。"
    )
    report = f"""# FlagGems Hopper MM 与 Triton 教程：31点实测

主对比在**同一个FlagTree环境、同一张H100 GPU0**执行，
两边都按各自原始候选集逐shape调参后冻结：FlagGems相对教程的31点几何平均观测加速比为 **{main_summary['geometric_mean']:.3f}×**，
独立进程复测 **{repeat_summary['geometric_mean']:.3f}×**；
算术平均为{main_summary['arithmetic_mean']:.3f}× / {repeat_summary['arithmetic_mean']:.3f}×。
主测/复测分别{main_summary['faster']}/31、{repeat_summary['faster']}/31点观测快于教程。


{validity_statement} 主测无效标记点：{main_summary['invalid']}；
复测：{repeat_summary['invalid']}。
未丢弃这些点，
也未用最快轮次替换。
固定31点、已记录配对轮次条件下的95%bootstrap区间为[{main_summary['ci95'][0]:.3f},{main_summary['ci95'][1]:.3f}] /
[{repeat_summary['ci95'][0]:.3f},{repeat_summary['ci95'][1]:.3f}]，
不覆盖无效控制、调参选择偏差或所有跨进程系统变化。


## 比较对象和计时

FlagGems源码：[hopper/ops/mm.py](/root/FlagGems/src/flag_gems/runtime/backend/_nvidia/hopper/ops/mm.py)，
commit `{env['flaggems_commit']}`，
文件SHA256 `{env['source_sha256']}`。
没有修改此文件或FlagGems库源码。
实际运行其`mm_out(a,b,out=c)`以预分配输出，
实际路径全部为 `general_mm → mm_kernel_general_host_tma`，
每点实际kernel、配置、完整metadata和IR/cubin hash见shape JSON。
`mm`对这组连续FP16输入走相同设备路径，
只额外分配输出；
本指标不包含该分配及Python包装耗时。


FlagTree侧使用已有`/usr/bin/python`，
FlagTree0.7.0，
实际加载Triton3.6.0、PyTorch2.11.0+cu130。
实际triton/__init__.py和libtriton.so均核对FlagTree本地RECORD相符，
真实路径及hash见environment.json。
当前H100缺少FlagTune模型包，
默认AUTO按源码回退default路径，
日志保留；
不是强制运行某个cost model。
FlagGems使用原声明默认候选集（运行时记录{fm['records'][0]['flaggems_config_count']}配置），
没有人为缩减或改调参策略，
缓存限定在本比较目录。
TLE可用，
但本批FP16/H100不满足BF16/H20专用手工warp-specialized分支条件，
不能称这批在测该分支。


教程使用已冻结的官方v3.7.0 tutorial03计算源码，
**未加之前的unsigned TTIR优化**；
主比较在FlagTree环境中对官方16配置重新逐点autotune，
然后保存/冻结获选配置；
独立复测核对每点教程和FlagGems获选cubin字节相同，
禁止悄悄重新选择更快配置。
该环境实际共选出{len(tfconfigs)}套教程配置、{len(fgconfigs)}套FlagGems配置。
两边都是正常Triton编译/原生加载，
没有独立llc/ptxas路线或cubin热替换；
本比较是跨实现系统性能，
不是纯TTIR编辑收益实验。


输入为M=N=K=256…4096、步长128的完整31点，
连续行主序FP16，
FP16输出。
NumPy按seed+n生成相同CPU输入再上传，
逐点记录数据hash；
同环境配对使用同一份输入，
跨环境每个进程块也核对输入hash相同。
先正确性再计时：native FP16、禁TF32的FP32参考，
256/1024/4096另用CPU FP64；
容差atol=1e-2、rtol=1e-3，
输出NaN sentinel、有限性、输入不修改均检查。
四个完整进程块共186次实现/shape正确性检查通过，
具体误差逐点保存，
不把组合容差等同于绝对误差上限。


计时使用同一环境内同一`triton.testing.do_bench_cudagraph`，
每点5轮、rep=20ms、对内随机先后顺序、原始样本全部保留。
FlagGems计时调用实际公共`mm_out`，
包含其所有需要的设备工作；
descriptor等host准备、JIT/调参/模块加载不计入设备replay。
两边计时前预热10次并同步，
每轮还有同一教程kernel的A/A控制；
A/A中位比值偏离1超过3%或baseline轮次max/min超过1.15标记无效。
设备阶段检查未发现外来GPU0计算进程，
但不是连续独占证明；
未锁定/核实锁频，
无主动cache flush。
指标是**重复输入CUDA graph设备时间**，
不能当作含分配的Python调用或应用端到端加速。


## 对照原upstream教程环境

补充在原`/root/venvs/ir-ops-dev/bin/python`、官方Triton3.7.0/PyTorch2.12.0+cu130环境中，
重新运行原六组冻结教程baseline（未使用optimized.ttir）。
进程顺序为FlagTree主测 → upstream主测 → FlagTree独立复测 → upstream独立复测，
同一物理GPU0，
后两块seed123。
此项baseline保留原六组映射，
和FlagTree环境新选的教程配置可能不同。


`upstream教程耗时 / FlagTree FlagGems耗时`的全31点几何平均观测比值为 **{cross[0]['geometric_mean']:.3f}×** /
**{cross[1]['geometric_mean']:.3f}×**。
这不是同进程配对，
包含PyTorch、编译器、runtime及冻结配置不同的影响；
跨环境有效性标记并集为{cross[0]['invalid_union']} / {cross[1]['invalid_union']}，
因此只作观测参照，
不能归因于单独的kernel算法或FlagTree编译器。
没有拿历史GPU6结果直接除以本次GPU0时间。


## 全部逐点结果

加速比方向均为教程时间/FlagGems时间，
大于1表示FlagGems更快。
配对中位加速比不必等于两列耗时中位数相除。
跨环境列为主测进程块的两个中位数相除；
其单独有效性请结合原始upstream记录。


{chr(10).join(tables)}

## 原始证据与复现

[results/flagtree_main/summary.json](results/flagtree_main/summary.json)、[flagtree_repeat](results/flagtree_repeat/summary.json)、[upstream_main](results/upstream_main/summary.json)、[upstream_repeat](results/upstream_repeat/summary.json)保存全部样本、精度、输入hash、配置和标记；
每组environment.json记录真实解释器、分发包、模块来源、compiler/timer/源码hash、命令和GPU状态。
原生阶段产物在各运行的artifacts目录，
可按源码和获选配置重建，
hash保存在JSON。
最早GPU5 smoke发现其他进程而拒绝运行，
其失败日志保留；
后续全部在GPU0。
最小点smoke只用于流程验证，
不替代上述31点统计。


参考[比较契约](README.md)，
重放入口为[run.sh](run.sh)。
只读审查按KernelForge review-agent在结果完成后核查；
该框架当前未注册MM执行适配器，
不能把本地对比称其CLI qualification。
原纯TTIR实验baseline/协议文件保持不变。

"""
    (output_dir / "report.md").write_text(report)
    recheck_rows = [
        "| 轮次 | 重测shape | 配对加速比 | A/A | baseline漂移 max/min | 有效性 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for label, rows in [("主测", main_rechecks), ("复测", repeat_rechecks)]:
        for row in rows:
            recheck_rows.append(
                f"| {label} | {row['n']} | {row['speedup']:.3f}× | {row['control_speedup']:.3f} | "
                f"{row['baseline_drift']:.3f} | {'有效' if row['valid'] else '仍超限'} |"
            )
    recheck_text = (
        "\n## 一次固定范围的有效性重测\n\n"
        "按契约只对原先标记点做一次7轮重测，保持输入hash、两边配置及cubin完全相同；"
        "不按加速比选取轮次，不循环重试。原始完整结果如上保留。"
        "修复映射和summary哈希保存于analysis.json。\n\n"
        f"按该预先列定替换映射重构31点，几何均值为{repaired_summaries[0]['geometric_mean']:.3f}× / "
        f"{repaired_summaries[1]['geometric_mean']:.3f}×，剩余无效点为"
        f"{repaired_summaries[0]['invalid']} / {repaired_summaries[1]['invalid']}。"
        "仍有超限点，因此本轮结论保留为约1.15×的观测优势，不宣称所有点稳定验证。"
        "附加重测共12次实现/shape精度检查通过，和原四进程块合计198次；这不是198个独立shape。\n\n"
        + "\n".join(recheck_rows)
        + "\n\n![31-shape speedup comparison](speedups.png)\n"
    )
    (output_dir / "report.md").write_text(report + recheck_text)
    result = dict(
        primary=dict(main=main_summary, repeat=repeat_summary),
        repaired_primary=dict(main=repaired_summaries[0], repeat=repaired_summaries[1]),
        repair_sources=dict(main=main_sources, repeat=repeat_sources),
        repair_summary_sha256={
            name: hashlib.sha256(
                (RESULTS / name / "summary.json").read_bytes()
            ).hexdigest()
            for name in ["repair_main", "repair_repeat"]
        },
        cross_environment_observational=cross,
        raw_summary_sha256={
            name: hashlib.sha256(
                (RESULTS / name / "summary.json").read_bytes()
            ).hexdigest()
            for name in [
                "flagtree_main",
                "flagtree_repeat",
                "upstream_main",
                "upstream_repeat",
            ]
        },
        shapes=31,
        precision_checks=186,
        additional_repair_precision_checks=12,
        source_unchanged=True,
        valid_primary_claim=not bad,
        scope="Cross implementation/environment comparison; not direct TTIR optimization",
        analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (RESULTS / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = [r["n"] for r in repaired_main]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(
        sizes,
        [r["speedup"] for r in repaired_main],
        "o-",
        label="Main (one scoped recheck)",
    )
    axes[0].plot(
        sizes,
        [r["speedup"] for r in repaired_repeat],
        "s--",
        label="Independent repeat",
    )
    for label, rows in [("main", repaired_main), ("repeat", repaired_repeat)]:
        bad_rows = [r for r in rows if not r["valid"]]
        axes[0].scatter(
            [r["n"] for r in bad_rows],
            [r["speedup"] for r in bad_rows],
            marker="x",
            color="red",
            s=100,
            zorder=5,
            label=f"Flagged {label}",
        )
    axes[0].set_title("Same FlagTree environment: tutorial / FlagGems")
    axes[0].set_ylabel("Paired median speedup")
    axes[0].legend(fontsize=8)
    axes[1].plot(
        sizes,
        [
            u["tutorial_us"] / f["flaggems_us"]
            for f, u in zip(fm["records"], um["records"])
        ],
        "o-",
        label="Main process blocks",
    )
    axes[1].plot(
        sizes,
        [
            u["tutorial_us"] / f["flaggems_us"]
            for f, u in zip(fr["records"], ur["records"])
        ],
        "s--",
        label="Repeat process blocks",
    )
    axes[1].set_title(
        "Cross-environment reference (observational; includes flagged cases)"
    )
    axes[1].set_ylabel("Upstream tutorial / FlagGems")
    axes[1].set_xlabel("Square dimension: M=N=K, FP16")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.axhline(1, color="gray", linewidth=1)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "speedups.png", dpi=150)
    plt.close(fig)
    print(json.dumps(result["primary"], indent=2))
    print("cross", cross)


if __name__ == "__main__":
    main()
