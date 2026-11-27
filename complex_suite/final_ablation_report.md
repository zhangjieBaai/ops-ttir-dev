# 五个复杂算子的最终版消融复核

2026-11-24：五类实验已分别放在算子目录，沿用LayerNorm的契约、完整主/复测、逐项加入、最终移除、全shape和复现形式。补测保持r6最终选择，不重新择优，也不混合历史轮次平均。下列为本次消融会话中final相对原baseline的几何平均，A级证据。

| 算子 | shape数 | 主测 | 独立复测 | 本次协议 |
|---|---:|---:|---:|---|
| [INT4反量化＋GEMM](../int4_gemm/report_ablation.md) | 30 | 2.527885× | 2.531543× | 7/5轮，20ms，GPU0/1 |
| [Grouped/biased MoE](../moe_routing/report_ablation.md) | 30 | 1.542132× | 1.539492× | 5/3轮，10ms，GPU2/3 |
| [Residual RMSNorm＋INT8](../norm_quant/report_ablation.md) | 30 | 1.433742× | 1.436816× | 7/5轮，20ms，GPU6串行 |
| [Attention dK/dV](../attention_dkdv/report_ablation.md) | 32 | 1.800224× | 1.797149× | 7/5轮，20ms，GPU4/5，1200MHz |
| [Partial Top-K](../partial_topk/report_ablation.md) | 30 | 1.275797× | 1.271497× | 7/5轮，20ms，GPU7/1 |

每个shape先取同轮baseline/候选比值中位数，随后跨shape求几何平均。同卡不并发GPU任务，跨卡只比较各自会话内的配对比；不跨卡比较绝对时延。INT4释放GPU1后用于Top-K复测。Attention运行结束已恢复自动频率。原r6表与本次消融表属于不同会话，数值小幅变化不代表重新优化。

## 拆分后各自验证的问题

- INT4：窄加载/K mask完整2×2、组内复用、GROUP128右移，最终逐项移除。首轮内部组合保留为起点，不虚分其收益。
- MoE：membership、有序key、gather、rank、双值归约、常量特化、batch4的加入/移除，以及完整两阶段交叉。
- Norm：因式分解、联合统计、.cs、倒数、早写Z；倒数/早写Z完整2×2与最终移除。
- Attention：D64独立对照；D128的64-key、结构改写、dV后移以及最终去结构但保留dV，明确联合移除和条件效应。
- Top-K：整数key重算负结果、循环状态、静态ids、展开、合同特化、头部归并、packed比较，以及完整local/merge 2×2。

各进程均先核验源码/TTIR baseline原生LLVM/PTX/cubin一致及原数值合同，再计时并检查A/A。记录全部原始样本、输入/IR/后端哈希和资源。本次中间消融未全部重跑sanitizer；正式最终版沿用r6的memcheck/racecheck通过证据。公开Triton可表达和当前源码版性能/机器码等价分别陈述，首轮源码结果不外推到r6。

条件贡献不能相加；小于约1%的变化不据此宣称稳定优化。Top-K第一次新主测因控制别名冲突中止，旧数据保留并明确排除；正式结果全部来自修正后的v2完整主/复测。

## 复现与原始证据

先读各算子README和ablation_plan，再设置独占GPU执行对应run_experiment.sh，使用全新结果ID。各自的报告生成器核查原始样本和哈希；本索引再核验所有源文件哈希，生成命令：`python complex_suite/report_final_ablation.py`。

[INT4反量化＋GEMM汇总](../int4_gemm/ablation/results/v2_summary.json) · [实验入口](../int4_gemm/README.md)
[Grouped/biased MoE汇总](../moe_routing/ablation_v1/summary.json) · [实验入口](../moe_routing/README.md)
[Residual RMSNorm＋INT8汇总](../norm_quant/ablation_v1/summary.json) · [实验入口](../norm_quant/README.md)
[Attention dK/dV汇总](../attention_dkdv/results/ablation_v2/summary/summary.json) · [实验入口](../attention_dkdv/README.md)
[Partial Top-K汇总](../partial_topk/ablation_v1/summary.json) · [实验入口](../partial_topk/README.md)
