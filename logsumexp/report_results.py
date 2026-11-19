"""Recompute summaries and render the retained paired evidence without retiming."""

import argparse
import json
from pathlib import Path

from benchmark import bootstrap, summary
from common import ROOT, key, manifest, save_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--repeat", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    data = [json.loads(path.read_text()) for path in (args.main, args.repeat)]
    selected = args.candidate
    rows = [[r for r in d["records"] if r["name"] == selected] for d in data]
    assert all(len(r) == 30 for r in rows)
    assert all(d["aa_passed"] for d in data)
    assert data[0]["physical_uuid"] == data[1]["physical_uuid"]
    assert {key(r["shape"]) for r in rows[0]} == {key(r["shape"]) for r in rows[1]}
    scores = [dict(**summary(r), bootstrap_95=bootstrap(r)) for r in rows]
    validation = [
        [
            r
            for r in run
            if manifest()["mapping"][key(r["shape"])]["partition"] == "validation"
        ]
        for run in rows
    ]
    validation_scores = [
        dict(**summary(run), bootstrap_95=bootstrap(run)) for run in validation
    ]
    accepted = all(score["geometric"] > 1.01 for score in validation_scores) and all(
        score["worst"] >= 0.90 for score in scores
    )
    assert accepted, "Candidate did not meet frozen retention criteria"
    result = dict(
        candidate=selected,
        accepted=accepted,
        evidence_level="A: full matrix and independent process repeat",
        main=dict(
            path=str(args.main),
            sha256=sha256(args.main),
            summary=scores[0],
            validation=validation_scores[0],
        ),
        repeat=dict(
            path=str(args.repeat),
            sha256=sha256(args.repeat),
            summary=scores[1],
            validation=validation_scores[1],
        ),
    )
    save_json(args.output, result)
    lines = [
        "# logsumexp TTIR 第一轮实验报告",
        "",
        f'选中 `{selected}`。完整 30 点几何平均加速 {scores[0]["geometric"]:.4f}×，'
        f'独立进程复测 {scores[1]["geometric"]:.4f}×（A级，仅冻结矩阵）。',
        "",
        f'主测配对轮次 bootstrap 95% 区间 {scores[0]["bootstrap_95"]}；复测 {scores[1]["bootstrap_95"]}。'
        "区间仅估计已冻结 shape 上的轮次波动，不代表应用分布或 shape 总体置信区间。",
        "",
        f'10 个 validation 点几何平均 {validation_scores[0]["geometric"]:.4f}× / '
        f'{validation_scores[1]["geometric"]:.4f}×；最差点 '
        f'{scores[0]["worst"]:.4f}× / {scores[1]["worst"]:.4f}×。',
        "",
        "## 逐点结果",
        "",
        "| M,N,K | 组 | baseline µs | candidate µs | 主测加速 | 复测加速 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    repeated = {key(r["shape"]): r for r in rows[1]}
    for r in rows[0]:
        lines.append(
            f'| {key(r["shape"])} | {r["group"]} | {r["baseline_us"]:.3f} | '
            f'{r["candidate_us"]:.3f} | {r["median_speedup"]:.4f}× | '
            f'{repeated[key(r["shape"])]["median_speedup"]:.4f}× |'
        )
    lines += [
        "",
        "## 边界与证据",
        "",
        "唯一计算改动是 TTIR；原始 Python kernel、51 点映射、配置、grid、launcher 均冻结。"
        "基线来自用户指定 FlagGems logsumexp.py，原始提交及完整源码快照见 upstream/provenance.json。"
        "逐点源码/TTIR 的 LLVM、PTX、完整 cubin 字节一致，TTGIR 经相同 parser round trip 一致。",
        "",
        "使用 upstream Triton 3.7.0 / PyTorch 2.12.0+cu130 / H100 SM90。"
        "文件入口从 make_ttgir 继续，经原生编译、加载和启动；无独立 llc、无二进制替换。"
        "RECORD 哈希核对证明本地安装一致，不是独立远程 wheel 来源认证。",
        "",
        "计时为预分配输入输出的 CUDA graph replay 单 kernel 设备时间；"
        "7 轮主测、5 轮独立进程复测，rep=20ms，保存全部样本和顺序。"
        "无清缓存，无主动锁频；锁频配置未知。GPU 前后和运行中遥测均保存。"
        "GPU7 首次初筛发生其他进程干扰，整轮已弃用，见 round1_screen/invalid_timing.json；"
        "主测和复测均在同一物理 GPU0，不能与 GPU7 的绝对时间混比。",
        "",
        "FP16 contiguous，axis=1；有限值/-inf，51 个 shape ×8 种分布；"
        "oracle 在 CUDA FP64 上执行并舍入到 FP16，"
        "并对照 CUDA native FP32 ATen。rtol=1e-3、atol=2e-3，输出 guard 与输入保留检查。"
        "不支持 +inf/NaN、空输入、任意 stride、其他 dtype 或完整 FlagGems API；未映射 shape 拒绝。",
        "",
        "## 复现",
        "",
        "```bash",
        "CUDA_VISIBLE_DEVICES=0 bash logsumexp/test_accuracy.sh",
        "CUDA_VISIBLE_DEVICES=0 bash logsumexp/test_performance.sh",
        "```",
        "",
        "`results/selection.json` 绑定候选和完整结果。每组 `optimized.ttir` 与所选不可变候选逐字节一致。"
        "资源变化、候选负结果、sanitizer 状态由随附 acceptance.json/补充说明记录；"
        "资源指标本身不构成性能因果证明。",
    ]
    report = ROOT / "report_round1.md"
    with report.open("x") as stream:
        stream.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
