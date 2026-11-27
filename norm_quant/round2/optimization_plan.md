继承../../complex_suite/round2_plan.md及../optimization_plan.md。parent=stream_v1，原8点不变。

本轮候选预算5个：reciprocal（逐元素precise_div改为一次标量precise reciprocal再乘，改变舍入须按原Q误差合同检查）；early_z（Z的地址生成/store移至统计归约前，减少z跨归约存活）；reciprocal_early（两项组合）；fold256（在组合上把sum/max输入reshape为[B/256,256]先axis0再axis0，探索归约组织）；div_approx（仅将向量precise_div改为arith.divf，作精度/性能比较）。禁止扩大量化误差门槛，Z和scale合同不变。前四种公开Triton也能表达；本轮不作源码复现。

选择按所有8点geo，不按单shape拼接。性能收益未知，各候选均先测试normal/zero/constant/offset/cancellation/tiny/signed_weight。最终额外检查大幅度、最大权重与零权重等输入，再做memcheck/racecheck。
