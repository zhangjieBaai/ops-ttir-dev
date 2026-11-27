# 第三至第五轮TTIR优化

用户要求继续多轮尝试。本次明确执行r3/r4/r5三个顺序优化轮次，不把多seed或复测次数叫作新优化轮。每轮保持既有152点（四类各30、Attention32）、dtype、数值合同、固定配置、ABI/grid、kernel数和workspace。

仅直接修改TTIR；不从优化Python kernel导出，不改TTGIR/LLVM/PTX/cubin，不使用inline PTX。原实验目录和选择记录不可覆盖；新工作在各算子multiround/。

四路含义：baseline=原始版本，anchor=第二轮已验收版本，parent=本轮开始时已验收版本，candidate=本轮新候选。r3 parent=anchor；后续只有通过正式验收的候选才能成为parent。每轮测直接parent/candidate、anchor/candidate、baseline/candidate，不能乘历史平均得到累计收益。

每轮预先给出2–3个不同机制候选，完整C0/C1后初筛3轮rep10；冻结全矩阵候选（Attention沿用完整D/causal组），主测7轮rep20、独立进程5轮rep20，随机交错且保存所有外层配对样本、输入seed/hash、IR/各产物hash、资源与A/A。A/A中位门槛[0.97,1.03]，晋升要求两正式运行增量geo>=1.02、最差>=0.95、数值及sanitizer通过，否则保留parent。失败与回退保留；机器码全域相同的候选可按C级无新代码证据结束，不伪造性能收益。

官方Triton3.7.0 venv。GPU0/2/4/6/7分别供INT4/MoE/Attention/Norm/TopK独占。Attention沿用已验证SM1200MHz条件，临时设置并在finally恢复自动频率；其余卡不主动锁频。同卡不并发性能、正确性或sanitizer。

10/20ms指do_bench_cudagraph的rep参数。内部10次replay返回均值，保存的是全部外层调用结果。每次正式运行先验证原baseline源码/文件LLVM/PTX/cubin以及完整options一致。主机计时只launch设备kernel，PyTorch参考仅用于未计时校验。

允许复用未变候选的数学参考，减少重复CPU/GPU排序开销；每个版本仍单独重置输出、启动、检查参考、canary和输入不修改。每轮保证新的IR哈希与验证记录对应。跨算子不算总平均；形状数量不因重复三轮而变成456个不同shape。
