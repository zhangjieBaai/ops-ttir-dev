继承../../complex_suite/round2_plan.md及../optimization_plan.md。parent=ordered_carry_v3，原6点、两阶段和稳定index合同不变，有限FP32含subnormal与±0。

候选预算4个：packed64（把uint32有序score key和反向原始index打包到uint64，循环只归约一次max64，降低两阶段归约通信但可能失去redux32且增加状态）；unroll（手动展开K选择循环，去掉loop-carried状态与scalar计数，观察代码体积和寄存器代价）；static_ids（保留原始ids，不随选择更新；score哨兵控制已选元素，耗尽时显式输出INTMAX）；packed_unroll（64位key与展开组合）。均从parent TTIR直接重写，不调用优化源码编译。

packed64编码=(uint64(score_key)<<32)|(uint64(0x7fffffff-id))。真实id为[0,N)，tail padding id为INTMAX，其反向key为0；已选值写入((-inf有序key)<<32)|0。耗尽后输出-inf/INTMAX，与原两阶段约定一致。score key最小合法finite值大于-inf key，所以不会重新选中已选元素；ties按较小id。除了最终value/indices，还检查local工作区的非法candidate不能影响全局TopK。
