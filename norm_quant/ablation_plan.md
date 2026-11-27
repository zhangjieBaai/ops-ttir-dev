# Norm＋动态 INT8 最终版消融

2026-11-24。固定第六轮30个shape、原始配置、ABI/grid、单kernel、无别名、有限FP16输入和原FP64量化容差。当前最终版逐组等于第二轮 `reciprocal_early`；第三至六轮未晋升的候选不纳入优化链。原baseline、候选和历史结果保持冻结。

逐项加入：baseline → factor（amax因式分解）→ joint（联合sum/max）→ stream（输出.cs）→ reciprocal（量化除法改标量倒数复用）→ final（提前写Z）。在stream上另测early_z，构成倒数/早写Z的完整2×2。从final移除.cs，或把联合统计改回两个FP32归约；取消倒数和取消早写分别对应early_z、reciprocal。因式分解是联合统计的结构前提，不能声称独立移除；报告条件效应，不能相加。

使用官方 `/root/venvs/ir-ops-dev/bin/python`，独占GPU6，同卡串行。每个进程先核验30组源码/TTIR baseline LLVM/PTX/cubin及options完全一致，再完成30×9×7分布数值检查；normal完整M，压力分布min(M,3)，小行另对CPU FP64，检查canary和输入不修改。所有验证完成才开始CUDA graph性能测量。主测7轮/rep20ms/seed42，独立进程复测5轮/rep20ms/seed123，所有路径100ms预热，随机交错并记录每轮A/A、输入哈希、IR及后端产物哈希。A/A逐shape中位数[0.97,1.03]；失败保留并使用新ID复跑，不重新择优。最终版已有r6 sanitizer证据，新增结构消融不冒称全部重新通过sanitizer。

新增IR、manifest与结果在 `ablation_v1/`，可再生后端产物在已忽略的 `artifacts/ablation_v1/`。复现入口 `run_experiment.sh`，报告 `report_ablation.md`。本轮未做当前最终版源码回写；第一轮源码表达/性能证据只适用于第一轮。
