本次按用户要求继续三个顺序搜索轮r3/r4/r5，沿用第二轮32点、4warps/5stages、连续FP16、原ABI/grid、数学参考及逐位输出合同。baseline为原教程TTIR，anchor固定为第二轮各组已验收版本；parent随上一轮验收结果更新，未达门槛保留旧parent。只直接修改TTIR，不生成优化Python源码，不改下游IR、不用inline PTX、不调参。

r3两个新结构候选：structure在D128组合第二轮未共同采用的64-row外层展开与指针重算，D64尝试在已后移dV的parent上将dP提前；query_pair将相邻两个查询迭代显式合入一次循环，保持每个accumulator的原更新顺序。所有查询迭代数在原N为128倍数的合同下为偶数（masked8次，unmasked为4的倍数）。

r4两个新候选：只对Q/dO的循环读取使用cg（区别于第一轮所有load一起cg）；在本轮parent上提前delta地址/读取到QK之前（区别于原baseline单项）。r5两个新候选：输出DK/DV使用cs；对Q/dO循环load使用ca/evict_last。每轮从当前parent产生，不把历史其他parent上的负结果直接搬作本轮结论；若机器代码相同则记录C级identity。

每轮完整44输入精度（32正常+每组3个N128压力）优先，再3轮rep10ms初筛。按原D/causal完整8点组冻结候选，7轮rep20ms主测和独立seed123的5轮复测。所有路线含baseline/anchor/parent/candidate并保存同轮直接比值，A/A必须在[0.97,1.03]；增量geo两次>=1.02且最差>=0.95、数值与sanitizer通过才晋升。候选失败及未晋升均保留，不根据复测重新选择。

GPU4独占并暂时锁定SM1200MHz；初始实际500ms、每次计时前实际100ms同路线图预热，计时本体为原do_bench_cudagraph。rep10/20是库参数，库内部10次重放返回均值，保存的是全部外层配对样本。try/finally保证正常/异常退出恢复自动频率，记录实际设置、频率和返回码。每次正式运行重新核对原source/file LLVM/PTX/cubin。每轮最终版本接受memcheck/racecheck；源码可表达性仅静态说明，不声称本次做源码回写。
