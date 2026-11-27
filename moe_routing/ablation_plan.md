# MoE routing 最终版消融契约

研究已经验收的 r6 `batch4`，不重新搜索、不修改 optimized 别名。冻结原始 baseline 与现有选择保持不变。官方 Triton 3.7.0、H100/SM90、FP32，E=256/512、G=8、top4 groups、top8 experts、正16倍数 tokens，完整原30点，4 warps、3 stages、1 CTA，两个 kernel 和原工作区/host launch。数学、sigmoid 的 tanh 形式、稳定 tie、原始权重及归一化保持原合同。NaN/Inf 和其他 E/G/K 不在契约内。

## 固定对照

`make_ablation.py` 只直接改写冻结 TTIR；生成后再次检查最终两阶段与 r6 optimized 的字节一致。16条路线的每条均计完整两阶段：

1. baseline → membership 广播 → uint32 有序 key → 循环后 gather → 8×8 稳定 group rank → stage1 top2 pair reducer → 两阶段固定整数特化 → stage1 batch4。前后两版除声明改动外相同；这是按该顺序的条件收益。
2. 从最终版分别去掉 membership、ordered key、gather、rank、pair reducer、整数特化。去掉 batch4 复用上述 add_constants；不另造同字节别名。
3. baseline、仅最终 stage1、仅最终 stage2、完整最终版进行两阶段交叉消融。

去掉 pair 时，在相同 batch4 二维数据流内恢复“max1、最大值计数、若最大值唯一则屏蔽、max2”的原算法，不能把去掉 batch 一并算作 pair 贡献。去掉整数特化恢复运行时整数操作数及输出 stride，保留既有固定 tensor 形状和 batch4（形状固定与标量特化是不同控制）。不改变编译选项。

## 先验证再计时

每个主实验独立进程先 `verify_pipeline.py`；runner 另对全部30点核查原冻结源码与 baseline 文件入口 LLIR/PTX/cubin、后端 options 和输出一致。每个版本 × 30点 × 8分布检查 group score 与 weight 的 FP32 原始位型、精确 indices、CPU稳定索引及权重容差、输入不变和 canary，总3840项/进程。8分布为normal/ties/group_ties/bias_dominant/saturated/negative/negative_bias/wide_bias。任何失败阻止性能测量。

固定主测seed42、5轮、rep10ms；独立进程seed123、3轮、rep10ms。同会话随机交错完整各路线，每路线100ms稳定阶段，保存全部外层样本和baseline A/A，逐shape A/A中位数须在[0.97,1.03]。同GPU串行；主测GPU2、独立复测GPU3可同时运行。自动频率，不清cache，计预分配CUDA graph两kernel设备时间，不含编译/分配/CPU launch。每版本各阶段保存输入TTIR与下游所有哈希、寄存器/spill/shared。新结果路径不可已存在。

主结论取同shape每轮baseline/版本比值的中位数，再跨shape几何平均。条件贡献用同一轮前版/后版直接比值计算，不乘历史轮平均，不相加贡献。完整矩阵且通过独立复测为A；子集/失败或未复测项为B；编译资源为C。该轮不晋升新候选；既有最终版sanitizer证据引用r6，新增消融版本没有额外sanitizer则明确声明。

## 复现

从仓库根目录运行 `CUDA_VISIBLE_DEVICES=2 bash moe_routing/run_experiment.sh OUTPUT 42 5`；复测使用空闲GPU3及新OUTPUT、seed123、3轮。脚本先检查环境/生成不可变对照，再verify_pipeline、完整数值/计时，最后`report_ablation.py --main ... --repeat ...` 从两份原始记录生成报告及CSV。初筛、旧的六点消融和未完成main30/repeat30记录保持历史原样。
