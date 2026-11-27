# Attention dK/dV 源码回写合同

本轮用户明确请求“把现有 TTIR 改动写成 Triton 源码并实测”，因此新增独立源码回写实验；不改父目录任何冻结文件。唯一新kernel来自 r1_dv_late 的忠实回写：将 dV 的 FP16 cast 和 dot 移到 dK dot 后，helper中其他语句及wrapper不改；black/isort后冻结，见source_manifest.json。没有重调warp/stage/tile。

三路：baseline=原baseline.ttir，ir=原r1_dv_late.ttir（四组均测，保留D128负结果），source=本轮frozen_port.py正常JIT产物。三路计时经相同 CompiledKernel launcher；source绝不替换成导出的TTIR编译产物。额外pipeline仅验证source->TTIR->recompile的LLIR/PTX/cubin一致；同时逐12点重跑原baseline源码gate。

继承原完整12点、连续FP16、N为128倍数、D64/128、causal/noncausal、4warps/5stages、BLOCK_M1=32/BLOCK_N1=128和固定ABI/grid。精度继承28输入(12性能点+16边界压力)、rtol=0.03/atol=0.008、CPU FP64 autograd小例、输入不可改/输出覆盖/有限性。正确性通过后每点随机三路顺序7轮20ms主测，独立进程seed123的5轮20ms复测。每轮记录三路时间和baseline/IR、baseline/source、IR/source全部比值，A/A每点3对，保存输入seed/hash、source/TTIR/各产物/ELF.text哈希及资源。

数值先于性能。memcheck/racecheck专查新源码全28输入，完整记录数值和工具summary。源码/原IR结果无论更快、等速或更慢均报告；不据单次快慢重新调优，也不把源码与IR不同的debug信息误当机器指令差异。只有全矩阵+独立复测用于A级性能结论；资源与顺序观察为C级，不单独证明性能因果。


## 完整重测v2（计时有效性修订）

共享冻结门槛来源 `complex_suite/source_port_plan.md`：A/A每点中位比必须位于[0.97,1.03]。原main/repeat大D64非因果点超限，两轮全部保留并排除最终性能结论。v2保持三路kernel/输入/seed/rounds/rep不变，对所有路线统一增加预热为每路500ms，完整重跑12点main_v2和repeat_v2，不挑最优历史轮次。已通过数值与sanitizer无需重复；若v2仍超限保留未资格结论。
