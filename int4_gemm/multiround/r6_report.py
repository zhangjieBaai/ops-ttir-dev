"""Recompute r6 evidence and report the tested and effective final paths."""

import hashlib
import json
import math
import statistics
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "r6"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(values):
    return {
        "geomean": math.exp(statistics.mean(math.log(x) for x in values)),
        "minimum": min(values),
        "maximum": max(values),
    }


def recompute(data, name):
    result = {}
    for metric, numerator in (
        ("incremental", "parent"),
        ("anchor_relative", "anchor"),
        ("cumulative", "baseline"),
    ):
        values = []
        for row in data["records"]:
            t = row["timing"]
            value = statistics.median(
                sample["us"][numerator] / sample["us"][name] for sample in t["samples"]
            )
            assert abs(value - t[metric][name]) < 1e-12
            values.append(value)
        result[metric] = aggregate(values)
        assert result[metric] == data["summary"][name][metric]
    return result


def main():
    selection = read(OUT / "selection.json")
    manifest = read(OUT / "manifest.json")
    assert (
        len(manifest["cases"])
        == len({tuple(c["shape"]) for c in manifest["cases"]})
        == 30
    )
    assert (OUT / "manifest.json").read_bytes() == (
        ROOT / "r5/manifest.json"
    ).read_bytes()
    assert manifest["source_sha256"] == sha(ROOT.parent / "tutorial.py")
    runs = {
        name: read(OUT / f"results/{name}/results.json")
        for name in ("validate", "screen", "main", "repeat", "memcheck", "racecheck")
    }
    checks = {}
    aa = {}
    for name, data in runs.items():
        assert data["passed"] and not data["failures"] and len(data["records"]) == 30
        checks[name] = sum(len(row["validation"]) for row in data["records"])
        for case, row in zip(manifest["cases"], data["records"]):
            assert (case["id"], case["shape"], case["config"]) == (
                row["case"],
                row["shape"],
                row["config"],
            )
            assert all(
                row["pipeline"][key]
                for key in ("passed", "llir_ptx_cubin_identical", "outputs_bitwise")
            )
            for kernel_name, kernel in row["kernels"].items():
                assert kernel["ttir_sha256"] == sha(
                    OUT / "ir" / case["id"] / f"{kernel_name}.ttir"
                )
            assert all(
                check["baseline_raw_int16_bits"] and check["canary_inputs_unchanged"]
                for check in row["validation"]
            )
        if name in ("screen", "main", "repeat"):
            values = [
                statistics.median(a["us"][0] / a["us"][1] for a in row["timing"]["aa"])
                for row in data["records"]
            ]
            assert data["aa_passed"] and min(values) >= 0.97 and max(values) <= 1.03
            aa[name] = {"minimum": min(values), "maximum": max(values)}
            for candidate in data["valid_names"]:
                recompute(data, candidate)
    for case in manifest["cases"]:
        folder = OUT / "ir" / case["id"]
        assert (folder / "parent.ttir").read_bytes() == (
            ROOT / "r5/ir" / case["id"] / "optimized.ttir"
        ).read_bytes()
        assert (folder / "baseline.ttir").read_bytes() == (
            ROOT / "r5/ir" / case["id"] / "baseline.ttir"
        ).read_bytes()
        assert (folder / "anchor.ttir").read_bytes() == (
            ROOT / "r5/ir" / case["id"] / "anchor.ttir"
        ).read_bytes()
        assert (folder / "optimized.ttir").read_bytes() == (
            folder / f"{selection['selected']}.ttir"
        ).read_bytes()
    nibble_values = []
    for n in range(16):
        value = struct.unpack("<f", struct.pack("<I", 0x4B000000 | n))[0] - 8388608.0
        assert struct.pack("<f", value) == struct.pack("<f", float(n))
        nibble_values.append({"nibble": n, "fp32_bits": struct.pack("<f", value).hex()})
    candidate = selection["candidate_tested"]
    selected = selection["selected"]
    representative = next(
        row for row in runs["validate"]["records"] if row["case"] == "m128_n11008_k4096"
    )
    resources = {
        name: {
            "registers": kernel["n_regs"],
            "spills": kernel["n_spills"],
            "shared_bytes": kernel["metadata"]["shared"],
            "elf_text": kernel["elf_text"],
        }
        for name, kernel in representative["kernels"].items()
    }
    summary = {
        "operator": "int4_gemm",
        "round": "r6",
        "shape_count": 30,
        "parent": "multiround/r5/optimized (r3 group128)",
        "anchor": "round2 r2_group_pair",
        "candidate_tested": candidate,
        "accepted": selection["accepted"],
        "selected": selected,
        "final_effective_candidate": (
            candidate if selection["accepted"] else "r3/group128"
        ),
        "main": recompute(runs["main"], candidate),
        "repeat": recompute(runs["repeat"], candidate),
        "final_main": recompute(runs["main"], selected),
        "final_repeat": recompute(runs["repeat"], selected),
        "checks": checks,
        "checks_total": sum(checks.values()),
        "aa": aa,
        "sanitizer": selection["sanitizer"],
        "sanitizer_routes": runs["memcheck"]["valid_names"],
        "kernels": selection["kernels"],
        "identity_check": read(OUT / "identity_check.json"),
        "representative_resources": {"shape": [128, 11008, 4096], "kernels": resources},
        "candidate_screen": {
            n: runs["screen"]["summary"][n]
            for n in read(OUT / "candidates.json")["names"]
        },
        "source_port_measured": False,
        "source_expressibility": (
            "Public join/trans/reshape, casts/bitcast, integer shifts/or and scalar FP32 "
            "operations express all three transformations."
        ),
        "evidence": (
            "A for qualified full30 main/repeat performance; C for code/resource "
            "identities; no profiler causality claim."
        ),
        "audit": {
            "raw_ratios_recomputed": True,
            "hash_chain_matches": True,
            "all30_pipeline": True,
            "nibble_cpu_exhaustive": nibble_values,
        },
        "evidence_sha256": {
            name: sha(OUT / f"results/{name}/results.json") for name in runs
        },
        "script_sha256": {
            name: sha(ROOT / name)
            for name in (
                "r6_generate.py",
                "r6_run.py",
                "r6_workflow.py",
                "r6_report.py",
            )
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    promoted = (
        "接受新候选" if selection["accepted"] else "未达到晋升门槛，保留 r3 group128"
    )
    lines = [
        "# INT4 r6：整数拼接与精确 nibble 转换",
        "",
        f"30个固定shape，本轮最终候选 `{candidate}`：**{promoted}**。",
        f"相对本轮parent，主测/独立复测几何平均 **{summary['main']['incremental']['geomean']:.6f}× / "
        f"{summary['repeat']['incremental']['geomean']:.6f}×**；"
        f"最差分别为 {summary['main']['incremental']['minimum']:.6f}× / {summary['repeat']['incremental']['minimum']:.6f}×。",
        "",
        f"最终保留路径相对第二轮anchor：**{summary['final_main']['anchor_relative']['geomean']:.6f}× / "
        f"{summary['final_repeat']['anchor_relative']['geomean']:.6f}×**；"
        f"相对原始baseline累计：**{summary['final_main']['cumulative']['geomean']:.6f}× / "
        f"{summary['final_repeat']['cumulative']['geomean']:.6f}×**。",
        "这些数字全部由本轮同场实际路径配对计算；拒绝候选时使用测得的parent时间，不乘历史加速比。",
        "",
        "## 本轮新变换",
        "",
        "- `interleave_u8_v1`：先对整数low/high nibble做join、转置和reshape，"
        "再在[BK,BN]上执行一次FP32转换及反量化。此前cast_after仅移动FP16截断，"
        "本轮改变了更早的整数表示及反量化张量形状。",
        "- `magic_fp32_v1`：将0..15扩为uint32，OR上0x4b000000，bitcast为FP32后减8388608。"
        "得到与uint8转FP32相同的值；16种nibble已做CPU穷举位型核验，所有GPU压力输出也逐位一致。",
        "- `interleave_magic_v1`：上述两项组合。保留原FP32减零点、乘scale的次序、FP16舍入点和dot累积次序。",
        "",
        "| 候选 | 初筛增量geo（B级） | 最差 | 与parent相同可执行.text点数 |",
        "|---|---:|---:|---:|",
    ]
    for n, metrics in summary["candidate_screen"].items():
        lines.append(
            f"| {n} | {metrics['incremental']['geomean']:.6f}× "
            f"| {metrics['incremental']['minimum']:.6f}× | {summary['identity_check'][n]}/30 |"
        )
    lines += [
        "",
        "三个候选在全部30映射都改变了可执行.text；这是真实lowering变化，不是调试信息差异。",
        "例如M128,N11008,K4096的静态资源如下。寄存器/shared的变化只提供机制线索，"
        "没有动态NCU采样，不能把回退精确归因于某条指令或某项资源。",
        "",
        "| 版本 | 寄存器 | spill | shared bytes |",
        "|---|---:|---:|---:|",
    ]
    for n in ["parent", *summary["candidate_screen"]]:
        r = resources[n]
        lines.append(
            f"| {n} | {r['registers']} | {r['spills']} | {r['shared_bytes']} |"
        )
    lines += [
        "",
        "## 数值、测量与边界",
        "",
        f"{sum(checks.values())}项版本/分布检查通过，包含初始验证、初筛、主测、独立复测和两种sanitizer；"
        "五种原分布、raw FP16位型、CPU FP64小shape、canary及输入不变检查均保留。",
        "全部30点的原源码与文件baseline LLVM/PTX/cubin相同，当前TTIR哈希与各性能/数值/sanitizer记录一致。memcheck零错误，racecheck零错误、零警告。",
        "两种sanitizer覆盖baseline、anchor、parent与冻结的最终候选；其余两个候选完成数值验证和初筛。",
        "性能为预分配CUDA graph设备时间，不含编译、分配或Python开销。GPU0独占，H100、官方Triton3.7.0；"
        "配置/grid/ABI/workspace未变，无autotune，无锁频或冷cache控制。",
        "初筛3轮rep10ms；冻结候选后主测7轮rep20ms、独立seed123复测5轮rep20ms。rep为do_bench_cudagraph参数，全部外层随机顺序与配对样本保留。",
        "晋升要求主/复测增量geo均>=1.02且最差>=.95，以及数值和sanitizer通过。初筛收益不是最终证据。",
        "",
        "| 轮次 | A/A最小中位数 | A/A最大中位数 |",
        "|---|---:|---:|",
    ]
    for phase, values in aa.items():
        lines.append(f"| {phase} | {values['minimum']:.6f} | {values['maximum']:.6f} |")
    lines += [
        "",
        "公开Triton可通过join/trans/reshape、整数位运算和cast/bitcast表达本轮变换；本轮未做源码回写实测，不声称源码版本有相同机器指令或性能。",
        "",
        "## 全30点直接对照",
        "",
        "时间为主测中位数，单位us；加速比为各轮配对比值的中位数。最终路径可能与候选不同，见首段选择。",
        "",
        "| M,N,K | baseline us | anchor us | parent us | candidate us | 候选主测增量 | 候选复测增量 | 最终复测累计 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a, b in zip(runs["main"]["records"], runs["repeat"]["records"]):
        t, u, v = a["timing"], a["timing"]["median_us"], b["timing"]
        lines.append(
            f"| {tuple(a['shape'])} | {u['baseline']:.3f} | {u['anchor']:.3f} | {u['parent']:.3f} "
            f"| {u[candidate]:.3f} | {t['incremental'][candidate]:.6f}× "
            f"| {v['incremental'][candidate]:.6f}× | {v['cumulative'][selected]:.6f}× |"
        )
    lines += [
        "",
        "[预先计划](optimization_plan.md) · [机器可读汇总](summary.json) · "
        "[初始精度](results/validate/results.json) · [初筛](results/screen/results.json) · "
        "[主测](results/main/results.json) · [独立复测](results/repeat/results.json) · "
        "[选择](selection.json) · [memcheck](memcheck.log) · [racecheck](racecheck.log)",
        "",
        "复现：在新的结果目录中使用 `CUDA_VISIBLE_DEVICES=0 /root/venvs/ir-ops-dev/bin/python "
        "int4_gemm/multiround/r6_run.py --stage r6 --run <new-run-name> "
        "--names baseline,anchor,parent,<candidate> --rounds 7 --rep 20`。"
        "生成器拒绝覆盖不同内容的同名TTIR，已有测量目录也不能覆盖。",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "accepted": summary["accepted"],
                "candidate": candidate,
                "main": summary["main"],
                "repeat": summary["repeat"],
                "final_main": summary["final_main"],
                "final_repeat": summary["final_repeat"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
