"""Archive artifact hashes/resources and write a report from completed experiment evidence."""

import argparse
import hashlib
import json
import shutil
import struct
from pathlib import Path

from common import ROOT, save_json, sha256

RESULTS = ROOT / "results"


def text_section_hashes(path):
    # CUDA cubins are little-endian ELF64. Keep the evidence path dependency-free.
    data = Path(path).read_bytes()
    assert data[:6] == b"\x7fELF\x02\x01"
    shoff = struct.unpack_from("<Q", data, 40)[0]
    shsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 58)
    sections = [
        struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shsize)
        for i in range(shnum)
    ]
    table = sections[shstrndx]
    strings = data[table[4] : table[4] + table[5]]

    result = {}
    for section in sections:
        start = section[0]
        name = strings[start : strings.index(b"\0", start)].decode()
        if name.startswith(".text."):
            result[name] = hashlib.sha256(
                data[section[4] : section[4] + section[5]]
            ).hexdigest()
    return result


def read(name):
    return json.loads((RESULTS / name).read_text())


def main():
    global RESULTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--select", action="store_true", help="Freeze winner using full benchmark only"
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    RESULTS = args.results_dir.resolve()
    RESULTS.mkdir(parents=True, exist_ok=True)
    alias = (
        ROOT / "ir/optimized.ttir"
        if RESULTS == ROOT / "results"
        else RESULTS / "optimized.ttir"
    )
    full = read("full_benchmark.json")
    if args.select:
        assert len(full["records"]) == 60 and full["rounds"] == 7
        name = max(
            full["summary"], key=lambda label: full["summary"][label]["geometric_mean"]
        )
        selection = dict(
            kernel=name,
            sha256=sha256(ROOT / f"ir/{name}.ttir"),
            criterion="Highest geometric mean over 30 shapes in 7-round full benchmark",
            benchmark_sha256=sha256(RESULTS / "full_benchmark.json"),
        )
        path = RESULTS / "selection.json"
        if path.exists():
            assert read("selection.json") == selection, "Frozen selection changed"
        save_json(path, selection)
        shutil.copyfile(ROOT / f"ir/{name}.ttir", alias)
        print(json.dumps(selection, indent=2))
        return
    selection = read("selection.json")
    winner = selection["kernel"]
    assert selection["sha256"] == sha256(alias)
    repeat = read("independent_repeat.json")
    assert len(repeat["records"]) == 30 and repeat["rounds"] == 5
    names = [
        "baseline",
        "canonical",
        "arithmetic",
        "single_tile",
        "tile_dispatch_v2",
        "tile_dispatch_math_v2",
    ]
    resources = {}
    for name in names:
        artifact = ROOT / "artifacts" / name
        metadata = json.loads((artifact / "metadata.json").read_text())
        resources[name] = dict(
            n_regs=metadata["n_regs"],
            n_spills=metadata["n_spills"],
            shared=metadata["shared"],
            text_sections=text_section_hashes(artifact / "kernel.cubin"),
            files={
                p.name: sha256(p) for p in sorted(artifact.iterdir()) if p.is_file()
            },
        )
    assert (
        resources["baseline"]["text_sections"]
        == resources["canonical"]["text_sections"]
    )
    save_json(RESULTS / "artifact_manifest.json", resources)
    accuracy = read("accuracy_full.json")
    mem = read("memcheck_accuracy.json")
    race = read("racecheck_accuracy.json")
    assert accuracy["passed"] and mem["passed"] and race["passed"]
    for tool in ["memcheck", "racecheck"]:
        log = (RESULTS / f"{tool}.log").read_text()
        assert (
            "ERROR SUMMARY: 0 errors" in log
            if tool == "memcheck"
            else "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)" in log
        )
    summary = full["summary"][winner]
    again = repeat["summary"][winner]
    rows = [r for r in full["records"] if r["kernel"] == winner]
    re_rows = {r["N"]: r for r in repeat["records"]}
    r = resources[winner]
    lines = [
        "# LayerNorm TTIR 首轮实验报告",
        "",
        f"推荐 TTIR：`{alias.relative_to(ROOT) if alias.is_relative_to(ROOT) else alias}`，与 `{winner}.ttir` 字节一致。",
        "",
        f"H100 上 30 个 shape 的加速比：算术平均 **{summary['arithmetic_mean']:.3f}×**，"
        f"几何平均 **{summary['geometric_mean']:.3f}×**；独立进程、更换随机种子的 5 轮复测分别为 "
        f"**{again['arithmetic_mean']:.3f}× / {again['geometric_mean']:.3f}×**。",
        "",
        "## 基线与范围",
        "",
        "官方 Triton v3.7.0 教程 `_layer_norm_fwd_fused` 前向；从源码导出冻结 baseline.ttir，"
        "TTIR 重编译的 cubin 与源码编译一致，Y/Mean/Rstd 逐位一致。"
        "官方 Triton 3.7.0 后端，PyTorch 2.12.0+cu130，H100 80GB HBM3 / SM90，隔离 GPU 7。",
        "",
        "FP16 输入/输出/W/B，FP32 累加和统计量；8 warps、1 CTA、3 stages。"
        "baseline tile=8192，优化版在 TTIR 内按 N 选择逻辑 tile。N/stride 为正的 16 的倍数，"
        "指针对齐、列连续、X/Y 同行 stride、不支持别名。未验证其他 dtype、非对齐 N 或特殊非有限输入。",
        "",
        "## 采用的 TTIR 修改",
        "",
        "- 对 N≤1024/2048/4096/8192 分别添加单 tile 路径，使用对应 tensor 宽度。",
        "- 对 8192<N≤16384 展开两次 masked tile；N>16384 保留通用三遍循环。",
        "- 保留 mean→中心化 variance→输出的算法；展开使 mean/variance 的输入加载可以被后端合并，"
        "输出阶段在统计量 store 后仍保留一次 X 加载，未宣称所有输入只加载一次。",
    ]
    if "math" in winner:
        lines.append(
            "- 共用 1/N，再乘法计算 mean/variance；`math.sqrt + arith.divf` 改为 `math.rsqrt`。"
        )
    lines += [
        "",
        "所有候选均直接编辑 TTIR；没有手工修改 TTGIR/LLVM/PTX/cubin，编译选项保持相同。"
        "浮点计算允许容差内舍入变化，未要求优化版与 baseline 逐位等价。",
        "",
        "## 正确性与资源",
        "",
        f"6 个合法版本各 142 个用例、共 {len(accuracy['records'])} 次检查通过。"
        "包含原 31 个 shape、小 M、constant/offset/zero_mean、大尺寸通用路径、分支边界、"
        "padding=16 和 eps=1e-6/1e-3。Y 对 native ATen 最大绝对误差 <1e-2；"
        "Mean/Rstd 对 FP64 参考分别按 (rtol,atol)=(2e-5,1e-5)/(2e-4,1e-5) 验证；"
        "每版运行前重置输出 sentinel 并检查行尾 padding 未写坏。",
        "",
        f"baseline 与最终版各 17 个边界/padding 用例经过 memcheck 和 racecheck，工具均报告 0 errors。"
        f"最终版资源为 {r['n_regs']} registers/thread、n_spills={r['n_spills']}、shared={r['shared']} B；"
        "baseline 为 60 registers/thread、n_spills=0、shared=32 B。"
        "优化版有局部内存 spill，当前收益不代表寄存器占用得到改善。",
        "",
        "## 计时与逐 shape 结果",
        "",
        "配对执行 baseline/candidate，每轮随机候选与对内顺序；相同输入、预分配输出。"
        "完整矩阵 7 轮，独立复测 5 轮。每次 `do_bench_cudagraph` 内部进行 10 次 graph replay，"
        "报告其均值，再对轮次配对加速比取中位数。所有原始配对样本和顺序均保存。"
        "未锁时钟，未显式控制缓存冷热；指标是重复调用下的 graph kernel 时间，"
        "不含分配或 CPU launch，不等于端到端或冷缓存收益，也不与旧 LLVM 报告直接比较。",
        "",
        "| N（M=4096） | baseline µs | 优化 µs | 配对加速比 | 独立复测 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['N']} | {row['baseline_us']:.3f} | {row['candidate_us']:.3f} | "
            f"{row['speedup']:.3f}× | {re_rows[row['N']]['speedup']:.3f}× |"
        )
    lines += [
        "",
        "## 消融、负结果与证据",
        "",
        "| 候选 | 6 点筛选几何平均 | 寄存器/thread | n_spills |",
        "|---|---:|---:|---:|",
    ]
    screening = read("screening.json")
    for name, item in screening["summary"].items():
        lines.append(
            f"| {name} | {item['geometric_mean']:.3f}× | {resources[name]['n_regs']} | "
            f"{resources[name]['n_spills']} |"
        )
    lines += [
        "",
        "canonical 仅去除调试位置，其 ELF .text 哈希与 baseline 相同，计时约 1.000×。"
        "初始 tile_dispatch / tile_dispatch_math 因零常量尺寸不匹配在 C0 失败，原文件和失败说明保留，"
        "未执行或计入正确性结果。v2 使用分支局部零常量修正。"
        "结构修改与算术修改有独立候选，但未对每个分支宽度或单项算术做完整消融，不能拆分总收益。",
        "",
        "完整结果：`accuracy_full.json`、`screening.json`、`full_benchmark.json`、"
        "`independent_repeat.json`、`memcheck.log`、`racecheck.log`、`selection.json`、"
        "`artifact_manifest.json` 均在 `results/`。复现命令见 README。",
        "",
        "## 下一轮方向",
        "",
        "优先研究双 tile 路径的值存活期和 spill，拆分输入复用与动态循环消除的收益；"
        "增加受控冷缓存和应用调用测量后再评估部署收益。扩展 BF16/FP32 或非 16 对齐尺寸时，"
        "应另行从教程导出匹配签名/属性的 baseline 并验证，不复用本轮精度结论。",
        "",
    ]
    (
        ROOT / "report.md" if RESULTS == ROOT / "results" else RESULTS / "report.md"
    ).write_text("\n".join(lines))
    manifest = {
        str(p.relative_to(ROOT.parent)): sha256(p)
        for p in sorted(ROOT.rglob("*"))
        if p.is_file()
        and "artifacts" not in p.parts
        and "__pycache__" not in p.parts
        and p.name != "source_manifest.json"
    }
    save_json(RESULTS / "source_manifest.json", manifest)
    print(f"Report written for {winner}: {summary}")


if __name__ == "__main__":
    main()
