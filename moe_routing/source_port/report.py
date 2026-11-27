"""Read original three-way samples and generate the final report and summary."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RATIOS = ("baseline_over_ir", "baseline_over_source", "ir_over_source")


def read(path):
    return json.loads((ROOT / path).read_text())


def main():
    main_run = read("results/main/results.json")
    repeat = read("results/repeat/results.json")
    initial = read("results/initial_v2/results.json")
    mem = read("results/memcheck/results.json")
    race = read("results/racecheck/results.json")
    assert all(run["passed"] for run in (initial, main_run, repeat, mem, race))
    assert main_run["aa_passed"] and repeat["aa_passed"]
    assert "0 errors" in (ROOT / "results/memcheck.log").read_text()
    assert "0 errors, 0 warnings" in (ROOT / "results/racecheck.log").read_text()
    runs = {"main": main_run, "repeat": repeat}
    summary = {
        "operator": "moe_routing",
        "comparison": "baseline TTIR / membership TTIR / faithful source JIT",
        "evidence_level": "A, frozen eight-shape matrix only",
        **{name: run["geomean_ratios"] for name, run in runs.items()},
        "source_and_ir_elf_text_identical_all_points": all(
            item["evidence"]["pipeline"]["stage2_ir_source"]["same_elf_text"]
            for run in runs.values()
            for item in run["records"]
        ),
        "source_frontend_keeps_membership_structure": all(
            item["evidence"]["pipeline"]["stage2_ir_source"][
                "source_has_broadcast_shape"
            ]
            and not item["evidence"]["pipeline"]["stage2_ir_source"][
                "source_has_dense_membership_shape"
            ]
            for item in main_run["records"]
        ),
        "accuracy_checks": {
            name: sum(len(item["accuracy"]) for item in run["records"])
            for name, run in (
                ("initial", initial),
                ("main", main_run),
                ("repeat", repeat),
                ("memcheck", mem),
                ("racecheck", race),
            )
        },
        "sanitizer": {"memcheck": "0 errors", "racecheck": "0 errors, 0 warnings"},
        "aa_ranges": {
            name: [
                min(item["timing"]["aa_median"] for item in run["records"]),
                max(item["timing"]["aa_median"] for item in run["records"]),
            ]
            for name, run in runs.items()
        },
        "points": [],
    }
    rows = []
    for main_item, repeat_item in zip(main_run["records"], repeat["records"]):
        m, e = main_item["tokens"], main_item["experts"]
        time = main_item["timing"]["median_us"]
        ratio = main_item["timing"]["median_ratios"]
        rr = repeat_item["timing"]["median_ratios"]
        summary["points"].append(
            dict(
                tokens=m,
                experts=e,
                main_us=time,
                main=ratio,
                repeat=rr,
            )
        )
        rows.append(
            f"| {m}×{e} | {time['baseline']:.3f} | {time['ir']:.3f} | "
            f"{time['source']:.3f} | {ratio['baseline_over_ir']:.4f}× | "
            f"{ratio['baseline_over_source']:.4f}× | {ratio['ir_over_source']:.4f}× | "
            f"{rr['ir_over_source']:.4f}× |"
        )
    geo = main_run["geomean_ratios"]
    rg = repeat["geomean_ratios"]
    total = sum(summary["accuracy_checks"].values())
    text = f"""# MoE membership 源码回写：TTIR 与公开 Triton 的三路实测

忠实源码回写保留了原TTIR优化收益。
完整8点主测的 baseline/IR、baseline/source、IR/source 几何均值分别为
**{geo['baseline_over_ir']:.6f}×、{geo['baseline_over_source']:.6f}×、{geo['ir_over_source']:.6f}×**；
独立复测分别为
**{rg['baseline_over_ir']:.6f}×、{rg['baseline_over_source']:.6f}×、{rg['ir_over_source']:.6f}×**。
这组结果为A级性能证据，仅限父组冻结8点。
源码与membership TTIR的ELF `.text`逐字节相同，资源相同；
微小计时差异不足以支持两种表达存在性能优劣。

## 回写内容和三路身份

[membership_source_v1.py](membership_source_v1.py) 仅替换stage2的group membership：
原 `[E,8]` 比较/AND/sum变成公开 `tl.broadcast_to` 的 `[8,E/8]` 布尔张量，再 `tl.reshape` 到 `[E]`。
stage1使用原冻结源码的同一个JIT函数。
原8点、4 warps/3 stages、grid、两kernel、workspace、device ABI及数值合同均不变，没有重新调优。
这是用户明确授权的独立源码回写组；父目录纯TTIR基线、候选和历史结果未改。

三路被测对象分别为baseline TTIR文件编译、membership TTIR文件编译、
原stage1源码JIT与新stage2源码JIT返回的真实CompiledKernel。
新源码导出TTIR再文件编译只用于pipeline核验，没有替代计时中的源码对象。
源码launcher保留Python签名中的constexpr槽位；底层device ABI与文件入口一致。
初次harness漏掉中间constexpr槽位而抛出TypeError，见[失败日志](results/initial.log)；
修正仅涉及launcher，冻结源码未改，失败轮没有性能数据。

## 前端和机器码

每次运行逐点检查原baseline source/file及优化source/file的LLVM/PTX/cubin字节一致，
并比较三路完整backend options和target。全部通过。
优化源码前端的TTIR保留 `[8,E/8]` broadcast/reshape，消除了旧 `[E,8]` membership中间量。

跨“membership TTIR vs 优化源码”比较时，完整LLIR/PTX/cubin不同，
但两个E配置的ELF `.text`完全相同；
ELF section逐项比较只有 `.debug_line`、`.nv_debug_ptx_txt`、`.rela.debug_line` 不同，
见[section证据](results/elf_section_comparison.json)。
不能把`.text`一致写成完整cubin一致。
stage2资源：E256为25 registers、16B shared、0 spill；E512为28 registers、16B shared、0 spill，
两种优化路径均相同；baseline分别为32 registers、1024/2048B shared、0 spill。

## 完整两kernel设备时间

GPU2 H100、官方Triton3.7.0，预分配CUDA graph replay；
主测7轮、复测5轮，每项rep20ms，每轮随机三路执行顺序。
每个比值使用同一轮时间对，逐点取配对比中位数，再做几何汇总。
不计分配、编译、加载、Python调用；未清cache、未锁时钟。
每点输入显式seed和X/bias SHA256、全部执行顺序与样本保存在
[主测](results/main/results.json) 和 [复测](results/repeat/results.json)。
A/A主测各点中位比范围{summary['aa_ranges']['main'][0]:.6f}–{summary['aa_ranges']['main'][1]:.6f}，
复测{summary['aa_ranges']['repeat'][0]:.6f}–{summary['aa_ranges']['repeat'][1]:.6f}，均通过[0.97,1.03]门槛。

| tokens×experts | baseline μs | IR μs | source μs | baseline/IR | baseline/source | IR/source主测 | IR/source复测 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

时间列各自取中位数；配对比中位数不必等于两列时间之比。
`IR/source`大于1表示源码耗时更低，不表示TTIR收益更大。

## 精度与内存/同步检查

成功初验、主测、复测、memcheck、racecheck各144项（三路径×8点×6分布），
总计{total}项，全部通过。
分布沿用normal、ties、group_ties、bias_dominant、saturated、negative；
三个输出相对baseline逐位一致，CPU稳定indices完全相等，
CPU权重满足rtol=2e-5/atol=2e-6，且输入哈希不变、sentinel及canary通过。
新源码路径实际包含在两种sanitizer执行中：
[memcheck](results/memcheck.log) 0 errors；
[racecheck](results/racecheck.log) 0 errors、0 warnings。

## 可以得出的结论

本例的TTIR优化可以忠实回写公开Triton，前端保留目标结构，最终设备指令与TTIR方案相同。
因此这里没有测出必须直接改TTIR才能获得的独占收益。
这个结果只回答既有变换能否回写以及两条编译路径的性能，
不衡量Agent在未见答案时发现该优化的概率，也不是A/B/C盲化优化能力试验。
复现命令见[README](README.md)，机器可读三路汇总见[summary.json](summary.json)。
"""
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (ROOT / "report.md").write_text(text)
    print(json.dumps({name: summary[name] for name in ("main", "repeat")}, indent=2))


if __name__ == "__main__":
    main()
