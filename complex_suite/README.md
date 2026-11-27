# 复杂融合算子 TTIR 实验

[分算子报告与最终加速比](../all_report.md) · [首轮汇总](report.md)

当前实验已按LayerNorm形式拆入五个算子目录，逐项消融不是逐轮搜索结果的简单拼接。首轮小矩阵、源码回写、扩展矩阵和r3–r6历史继续保留；本目录提供共用编译/计时/报告工具与历史索引。当前入口：

| 算子 | 独立入口 | 完整最终结果 | 最终版消融 |
|---|---|---|---|
| INT4 GEMM | [README](../int4_gemm/README.md) | [30点](../int4_gemm/report.md) | [报告](../int4_gemm/report_ablation.md) |
| MoE routing | [README](../moe_routing/README.md) | [30点](../moe_routing/report.md) | [报告](../moe_routing/report_ablation.md) |
| Norm＋INT8 | [README](../norm_quant/README.md) | [30点](../norm_quant/report.md) | [报告](../norm_quant/report_ablation.md) |
| Attention dK/dV | [README](../attention_dkdv/README.md) | [32点](../attention_dkdv/report.md) | [报告](../attention_dkdv/report_ablation.md) |
| Partial Top-K | [README](../partial_topk/README.md) | [30点](../partial_topk/report.md) | [报告](../partial_topk/report_ablation.md) |

每个目录都有`ablation_plan.md`、`run_experiment.sh`及原始结果/哈希/CSV。消融保持r6最终选择、冻结源码和原baseline不变，性能必须先过数值/编译链核查；加入/移除是条件效应，存在交互，不相加为总收益。以下继续列历史协议与公共工具。

[本次完整消融复核（152点）](final_ablation_report.md) · [带原始证据哈希的统一汇总](final_ablation_summary.json)。`report_final_ablation.py`从分算子核验汇总生成此索引，不乘历史平均。

[全部算子完整正文](../all_report.md)在13个独立标题下列出来源、每shape的基线时间/优化时间/加速比、方法收益和最终算术/几何均值，覆盖379个shape。`python complex_suite/report_operator_details.py`核验已保存的证据哈希、从原始配对样本重算并生成全部正文、独立report.md、[CSV](operator_details.csv)及[详细JSON](operator_details_summary.json)。

| 实验 | 首轮协议 | 分算子报告与复现 |
|---|---|---|
| INT4 fused GEMM | [协议](../int4_gemm/optimization_plan.md) | [结果](../int4_gemm/report.md) / [复现](../int4_gemm/README.md) |
| Grouped/biased routing | [协议](../moe_routing/optimization_plan.md) | [结果](../moe_routing/report.md) / [复现](../moe_routing/README.md) |
| Norm+INT8 | [协议](../norm_quant/optimization_plan.md) | [结果及命令](../norm_quant/report.md) |
| Attention dK/dV | [协议](../attention_dkdv/optimization_plan.md) | [结果](../attention_dkdv/report.md) / [复现](../attention_dkdv/README.md) |
| Partial Top-K | [协议](../partial_topk/optimization_plan.md) | [结果及命令](../partial_topk/report.md) |

`support.py`是Norm/Top-K共用的环境、编译、哈希与配对计时工具；`finalize.py`从最终原始JSON核验门槛、生成报告和optimized别名；不修改冻结源码或baseline。使用 `/root/venvs/ir-ops-dev/bin/python`，先隔离空闲GPU。同一设备不并发基准和sanitizer。

`article_reference.txt`保存本轮用户提供的研究建议，`static_analysis.json`记录选定点的PTX静态计数及产物哈希，不能将静态计数当动态性能因果证据。所有性能结论均相对各自固定baseline，无跨库最强或A/B/C信息优势声明。

[优化方法及 Triton 源码表达能力报告](optimization_methods_and_triton.md)：逐项解释改动、公开 API 对应写法，以及源码可表达和实际编译收益之间的边界。

[直接改Triton源码的性能实测](source_port_comparison.md)：五类已知TTIR优化的忠实源码回写，包含实际源码JIT/优化TTIR/原基线三路对照。

[第二轮TTIR优化：扩展152点](round2_report.md)，含相对上一轮parent的增量和相对原baseline的累计收益。

[第三至第五轮TTIR优化](multiround_report.md)：固定152点、三个顺序搜索轮，包含最终同轮实测、逐轮晋升／回退和源码表达边界。[机器可读汇总](multiround_summary.json)由`multiround_report.py`读取五类最终记录生成。

[第六轮TTIR优化](round6_report.md)：继续保持152点，新增MoE、Attention D128、Top-K收益；完整原始对照和负结果保留。

[全部152个shape的加速比](round6_per_shape.md)：统一使用独立复测，展示原始基线与最终优化版的加速比；附[CSV](round6_per_shape.csv)和[JSON](round6_per_shape.json)。
