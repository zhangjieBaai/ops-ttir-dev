# MoE membership 的忠实 Triton 源码回写

用户明确要求的新独立研究组；它允许公开 Triton 源码回写，不覆盖上一轮的纯 TTIR 实验记录。原 baseline、membership TTIR、配置映射及结果只读。新源码在首次编译前 black/isort，随后冻结文件 SHA256；不重调优、不扩展优化搜索。

stage1 使用上一轮原始源码的同一个 JIT function。stage2 将原 `[E,G]` 比较/归约实现的 group membership 改为 `tl.broadcast_to(group_selected[:,None], (G,E/G))` + `tl.reshape(...,(E,))`，对应已经接受的 membership TTIR。其余计算、签名、grid、两 kernel、workspace、配置均不变。

沿用8点 FP32、E256/512、G8、top4 groups、top8 experts、原 tanh sigmoid、偏置仅用于选择、原权重归一化和低索引 tie 语义。数值分布、CPU参考、逐位 baseline/IR/source 输出比对、输入不变和canary，均沿用父组6分布契约。验证覆盖完整8点，不把新测试默认为原历史结果。

三条被测路径：baseline TTIR 文件编译；membership TTIR 文件编译；`membership_source_v1.py` 真实源码 JIT 得到的 CompiledKernel（stage1也是原始源码JIT对象）。源码导出的 TTIR 再文件编译仅用于 pipeline 对照，不替换被测源码对象。每点先重验原 baseline source/file LLVM/PTX/cubin及完整编译options，再验证新source/file一致性和三路完整options。

同一GPU2独占，主测7轮、独立进程复测5轮，rep20ms；每轮将三路顺序随机打乱，各执行一次完整两kernel CUDA graph replay。保存同轮三项耗时，计算 baseline/IR、baseline/source、IR/source 配对比的中位数和全矩阵几何均值，另做baseline A/A控制。预分配输出，编译/分配/主机调用不计时；未清cache、未锁时钟。每点性能输入由显式seed派生，记录X/bias SHA256。

保存所有样本、执行顺序、源码与IR及下游产物哈希、ELF.text哈希、寄存器/shared/spill和环境。检查新源码前端是否保留broadcast/reshape结构、源码与TTIR机器指令是否一致；无论快慢都报告，不以相近时间推断机器码相同。source路径另做完整数值的memcheck/racecheck。主测/复测原始目录不可覆盖。
