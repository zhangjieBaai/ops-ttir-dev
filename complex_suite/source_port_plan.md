本轮由用户明确要求“把目前TTIR优化改动直接改Triton，实测性能”发起，单独建立源码回写对照组。该请求授权本组编写源码候选；原先纯TTIR实验的冻结源码、baseline、候选、映射和结果保持不变。本轮不重调优，不增加其他算法优化。

每个算子比较三条实际执行路径：B=原baseline TTIR文件编译产物，I=上一轮优化TTIR文件编译产物，S=忠实回写公开Triton源码后，经正常前端编译得到的CompiledKernel。S计时使用源码产生者，导出的TTIR重编译只用于pipeline核验，不能替代S路线。对照完整实际backend options、target、ABI、grid、kernel数、workspace和launcher，源码基线仍核对原冻结TTIR与LLVM/PTX/cubin。

五项为INT4 scalar_unsigned_v1（7点/GPU0）、MoE membership（8点/GPU2）、Norm stream_v1（8点/GPU6）、Attention r1_dv_late（12点/GPU4，包括D128负结果）、Top-K ordered_carry_v3（6点/GPU7）。固定官方Triton3.7.0，独占GPU，原数值合同和压力分布不放宽。源代码先格式化再冻结；产生实质修订必须新版本名并保留旧结果。

新源码完整正确性先于计时；三路同输入，计时输入先设seed且记录哈希。主测7轮/20ms，独立进程复测5轮/20ms，随机交错顺序并保留全部原始时间、A/A。报告baseline/IR、baseline/source、IR/source三种直接比值；IR/source>1表示源码更快。不用上一轮历史时间估算本轮比值。A/A点中位数在[0.97,1.03]外的组不作为最终有效结论，保留原因与重测记录。

保存源码/输入TTIR/各阶段/完整cubin与ELF.text哈希及资源。源码位置/命名可导致完整二进制不同，分别判断计算结构、机器指令和性能，不把同思路等同相同IR。至少核查关键变换是否在源码make_ttir后保留。若源码被前端改写，首先报告差异；必要时仅尝试有明确理由的公开语法v2，不无限搜索。

新源码kernel运行memcheck/racecheck；原合同适用：Norm量化允许相对FP64最多差1 code，Top-K允许±0数值相等，其余按各目录契约。结论是忠实回写后的实测性能，不是盲化A/B/C优化能力比较。
