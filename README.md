# ops-ttir-dev

直接修改 Triton TTIR 的算子优化实验，首个算子为 LayerNorm forward。

[全项目实验报告](all_report.md)。LayerNorm 实验入口：[layer_norm/README.md](layer_norm/README.md)。代码、冻结 TTIR、候选生成器和结果均由 Git 管理；可再生的 TTGIR/LLVM/PTX/cubin 放在忽略的 `layer_norm/artifacts/`。

MM 基线实验入口：[mm/README.md](mm/README.md)。性能矩阵、优化过程与验证范围见 [MM 优化总结](mm/optimization_summary.md)。

跨算子的统一实验要求见 [TTIR_EXPERIMENT_REQUIREMENTS.md](TTIR_EXPERIMENT_REQUIREMENTS.md)，包含环境、baseline 冻结、Triton 原生编译/加载/启动链及验证规范。

跨算子复用的优化知识见 [knowledge/README.md](knowledge/README.md)：TTIR 的表达力边界、已验技术目录、失败模式、诊断读法，以及每个数字的证据索引。

## 五个复杂算子的扩展实验

五项实验各自在独立算子目录组织，统一沿用LayerNorm的“契约→验证→主复测→优化→逐项消融→完整shape→复现”形式。

| 算子 | 实验入口 | 当前完整结果 | 最终版消融 | 契约 |
|---|---|---|---|---|
| INT4 反量化 GEMM | [README](int4_gemm/README.md) | [30点](int4_gemm/report.md) | [逐项加入/移除、2×2](int4_gemm/report_ablation.md) | [消融计划](int4_gemm/ablation_plan.md) |
| MoE routing | [README](moe_routing/README.md) | [30点](moe_routing/report.md) | [逐项加入/移除、阶段交叉](moe_routing/report_ablation.md) | [消融计划](moe_routing/ablation_plan.md) |
| Residual RMSNorm＋INT8 | [README](norm_quant/README.md) | [30点](norm_quant/report.md) | [逐项加入/移除、2×2](norm_quant/report_ablation.md) | [消融计划](norm_quant/ablation_plan.md) |
| Attention dK/dV | [README](attention_dkdv/README.md) | [32点](attention_dkdv/report.md) | [D64/D128和结构交互](attention_dkdv/report_ablation.md) | [消融计划](attention_dkdv/ablation_plan.md) |
| Partial Top-K | [README](partial_topk/README.md) | [30点](partial_topk/report.md) | [逐项链和两阶段2×2](partial_topk/report_ablation.md) | [消融计划](partial_topk/ablation_plan.md) |

每个目录都有独立运行与报告生成入口、冻结IR、原始样本、哈希和CSV；`complex_suite/`保留共用基础设施与跨算子历史。当前最终IR在各目录`multiround/r6/ir/`，首轮根`ir/optimized.ttir`不能代替当前选择。跨算子对照见[总览](all_report.md)，逐轮搜索、源码回写及负结果见[历史索引](complex_suite/README.md)。

[本次152点最终版消融结果](complex_suite/final_ablation_report.md) · [机器可读汇总](complex_suite/final_ablation_summary.json)。

[总报告](all_report.md)按13项实验分别展开，覆盖379个shape。每个shape统一只列基线时间、优化时间、加速比三项性能数据，并保留基线来源、优化方法增益和最终算术/几何平均。详细数据见[CSV](complex_suite/operator_details.csv)和[JSON](complex_suite/operator_details_summary.json)。从原始样本重新生成全部正文与独立报告：`python complex_suite/report_operator_details.py`。
