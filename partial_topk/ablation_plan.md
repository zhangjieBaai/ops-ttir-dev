# Partial Top-K 最终版消融

2026-11-24。固定r6的30个shape、有限FP32、K=8/16、chunk1024、两kernel及workspace、原grid/options。value数值精确、index逐位精确；±0数值等价，同分最小原始index优先。保留冻结原baseline和最终选择。

逐项链：baseline → 每轮重算有序key（保留负结果）→ key移到循环外并携带状态 → 静态ids → 展开K循环 → r5 parent（已验收的合同特化）→ heads（各有序块头部归并）→ final（packed uint64头部选择）。static_unroll与parent若机器码或IR相同须明确记录，不能把相同路径包装成优化收益。

完整2×2阶段交叉：baseline local/merge、最终local＋baseline merge、baseline local＋最终merge、最终local/merge。全部计入完整两个kernel序列，确认两种局部列表的排序/workspace约定兼容。对应从最终版分别移除整个local或merge组合；不把阶段收益当成单项贡献。heads→final隔离packed头部选择；parent→heads含结构改动，不能拆成缓存、比较和寄存器的独立因果收益。

官方 `/root/venvs/ir-ops-dev/bin/python`，主测独占GPU7，独立复测在INT4释放后独占GPU1；同卡没有并发GPU任务。只比较每张卡本会话内baseline/候选比值，不跨卡比较绝对时延。每个进程先完成全部30×2 baseline源码/TTIR LLVM/PTX/cubin/options核验；再完成30×10×8分布检查，normal完整M，其余min(M,2)，小行对CPU FP64稳定排序，覆盖±0、subnormal、最低有限值、尾块、tie、canary、输入不修改。所有验证先于性能。主测7轮/20ms/seed42，独立进程复测5轮/20ms/seed123，所有路径100ms预热，随机交错，保存每轮A/A、输入及IR/后端哈希；逐shape A/A中位数[0.97,1.03]。不重新选择最终版。最终版已有r6 memcheck/racecheck，新组合仅声称实际完成的检查。

新IR、manifest和原始结果保存在 `ablation_v1/`，后端产物在 `artifacts/ablation_v1/`。入口 `run_experiment.sh`，结果 `report_ablation.md`。源码回写历史不证明r6头部归并已做源码性能对照。

首次新主测`split_20261124_main`完成数值验证后发现计时控制别名覆盖真实parent，已中止并排除，原始部分数据保留。修复后完整重新执行`split_v2_20261124_main/repeat`，不混用旧样本；详见 `ablation_v1/rejected_run.json`。常规`run_experiment.sh`在用户指定同一空闲卡串行完成两进程，仍可复现相同协议。
