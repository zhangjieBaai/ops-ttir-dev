"""Keep public reports self-contained while provenance stays in raw results."""

import re

FLAGGEMS_REVISION = "ef8aea78fed8321c3e02a15bf31e847cd0579e4f"
FLAGGEMS_SOURCES = {
    "mean": [("Mean", "ops/mean.py")],
    "var_mean": [("VarMean", "ops/var_mean.py")],
    "sum": [("Sum", "ops/sum.py")],
    "logsumexp": [("Logsumexp", "ops/logsumexp.py")],
    "int4_gemm": [
        ("INT4 融合矩阵乘", "ops/_weight_int4pack_mm_with_scales_and_zeros.py")
    ],
    "moe_routing": [
        ("Hopper grouped top-k", "runtime/backend/_nvidia/hopper/grouped_topk.py")
    ],
    "norm_quant": [
        ("残差融合 RMSNorm", "fused/fused_add_rms_norm.py"),
        ("动态激活量化", "fused/act_quant.py"),
    ],
    "partial_topk": [("分块 Top-K", "ops/topk.py")],
}

SOURCES = {
    "layer_norm": "官方 Triton v3.7.0 LayerNorm 教程的前向实现。固定 BLOCK_SIZE=8192，"
    "导出 TTIR 后直接优化；基线与优化版均经相同 Triton 原生后端编译。",
    "mean": "FlagGems 的 mean_dim 实现。冻结原始启发式配置后导出 TTIR，连续输入视为 [M,N,K]，"
    "沿 N 取均值；只修改 K≥1024 的向量循环路径。基线的源码编译与 TTIR 编译产物一致。",
    "var_mean": "FlagGems 的方差与均值行归约实现。固定 BLOCK_N=1024，保留 int64 行索引语义。"
    "基线采用先求均值、再求中心化平方和的两遍算法；两版经相同 Triton 原生后端编译。",
    "logsumexp": "FlagGems 的 logsumexp 实现，包含内层与非内层归约路径。"
    "按原启发式配置导出 TTIR 后优化，基线与优化版共享 Triton 原生编译链和主机接口。",
    "vector_add": "官方 Triton 向量加教程。固定上游版本后导出 TTIR，"
    "基线的源码编译与 TTIR 编译在 LLVM IR、PTX 和 cubin 阶段一致。",
    "layer_norm/backward_dx": "官方 Triton v3.7.0 LayerNorm 教程的 dX 计算段，保留两次归约和 dX 写出。"
    "基线与优化版都不计算 dW/dB，也不使用梯度累加锁，因此不代表完整融合反向算子的性能。",
    "int4_gemm": "FlagGems 的 INT4 反量化融合矩阵乘实现。基线已在同一设备 kernel 中融合 "
    "INT4 解包、分组反量化和 Tensor Core GEMM，不以先完整反量化再矩阵乘作为性能基线。",
    "moe_routing": "FlagGems 面向 NVIDIA Hopper 的 grouped/biased top-k routing 实现。"
    "基线保留两个设备 kernel：第一阶段计算 8 个连续专家组的带 bias 最高两分之和，"
    "第二阶段选择 4 个组、8 个专家，并使用未加 bias 的 sigmoid 分数生成归一化权重。",
    "norm_quant": "参考 FlagGems 残差融合 RMSNorm 与动态激活量化数据流建立的融合研究基线，"
    "并非生产算子的原样提取。单 kernel 完成残差加法、RMSNorm、amax 和 INT8 量化，"
    "输出 FP32 残差 Z、逐行 scale 和 INT8 Q。",
    "attention_dkdv": "官方 Triton v3.7.0 融合 Attention 教程的反向 dK/dV 计算部分。"
    "基线每个 CTA 负责 128 个 key；LSE、delta 和预缩放 K 已提前准备，计时只包含 dK/dV kernel。",
    "partial_topk": "采用 FlagGems 分块重复选择思路建立的两阶段研究基线。每 1024 个输入先保留 K 项，"
    "再从局部候选中选择全局 Top-K；并非生产实现的原样提取，也不使用全排序作为基线。",
}


OVERVIEW = {
    "layer_norm": ("LayerNorm 前向", "官方 Triton v3.7.0 LayerNorm 教程"),
    "mean": ("Mean", "FlagGems mean_dim 实现"),
    "var_mean": ("var_mean", "FlagGems 方差与均值行归约实现"),
    "mm": ("MM†", "官方 Triton v3.7.0 矩阵乘教程，采用同一 host ABI 的匹配基线"),
    "sum": ("Sum", "FlagGems 全量两阶段 sum 实现"),
    "logsumexp": ("Logsumexp", "FlagGems logsumexp 实现"),
    "vector_add": ("Vector add", "官方 Triton 向量加教程"),
    "layer_norm/backward_dx": (
        "LayerNorm backward dX",
        "官方 Triton v3.7.0 LayerNorm 教程的 dX 计算段",
    ),
    "int4_gemm": ("INT4 反量化融合 GEMM", "FlagGems INT4 反量化融合矩阵乘实现"),
    "moe_routing": (
        "Grouped／biased MoE routing",
        "FlagGems Hopper grouped/biased top-k routing 实现",
    ),
    "norm_quant": (
        "Residual RMSNorm＋动态 INT8",
        "融合研究基线，参考 FlagGems 残差 RMSNorm 与动态激活量化",
    ),
    "attention_dkdv": (
        "Attention backward dK／dV",
        "官方 Triton v3.7.0 融合 Attention 教程的 dK/dV 计算段",
    ),
    "partial_topk": ("Partial Top-K", "两阶段研究基线，参考 FlagGems 分块 Top-K 思路"),
}


def overview(operators):
    lines = [
        "| 算子 | 算术平均加速比 | 几何平均加速比 | 算子来源 |",
        "|---|---:|---:|---|",
    ]
    assert len(operators) == len(OVERVIEW) == 13
    for operator in operators:
        name, source = OVERVIEW[operator["operator"]]
        lines.append(
            f"| {name} | {operator['arithmetic']:.6f}× | {operator['geometric']:.6f}× | {source} |"
        )
    lines.extend(
        [
            "",
            "平均值对各算子自身测试矩阵内的shape加速比等权计算。† MM包含计时有效性及同步检查限制，详见正文。",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def public_section(operator, text):
    source = SOURCES.get(operator)
    if source is None:
        match = re.search(r"(?m)^来源：([^\n]+)$", text)
        source = match[1] if match else None
    if source and operator in FLAGGEMS_SOURCES:
        links = [
            f"[{label}](https://github.com/flagos-ai/FlagGems/blob/{FLAGGEMS_REVISION}/src/flag_gems/{path})"
            for label, path in FLAGGEMS_SOURCES[operator]
        ]
        label = "参考源码" if operator in ("norm_quant", "partial_topk") else "上游源码"
        source += f" {label}：" + "、".join(links) + "（链接固定到实验使用版本）。"
    if source:
        text = re.sub(r"(?m)^来源：[^\n]+$", lambda _: "来源：" + source, text)
    text = re.sub(r"(?m)^证据：[^\n]+\n?", "", text)
    text = re.sub(
        r"(?:完整加入/移除及2×2|完整移除和两阶段交叉|倒数/早写Z完整2×2及最终移除|"
        r"完整交互|完整local/merge 2×2及移除对照)见\[消融报告\]\([^)]+\)。",
        "",
        text,
    )
    text = text.replace(
        "原数值合同和边界检查通过；当前最终版为r2_mixed_stream。",
        "数值与边界检查通过。",
    )
    text = text.replace(
        "基线/提取源码路径、实际optimized编译链、数值压力及计时控制有独立存证。",
        "基线与优化版的编译链一致性、数值压力和计时控制均已独立验证。",
    )
    return text.rstrip() + "\n"
