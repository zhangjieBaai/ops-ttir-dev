# Top-K三个顺序优化轮

继承../../complex_suite/multiround_plan.md；anchor=round2的static_unroll_v2，原30点、两阶段、稳定索引及±0数值等价合同不变。

- r3：分4/8个逻辑组，各组使用uint32 max/min-index，再把局部胜者装为uint64做一次全局选择；只在小组胜者上使用64位归约，区别于第二轮全向量packed64。逻辑分组不保证一定映射一warp，需看下游。
- r4：将每轮两个scalar输出store改为每2/4/8项批量store，探索通信、store等待与保留结果寄存器的取舍。选择顺序和所有输出值不变。
- r5：把零符号归一化移至key编码之后；利用固定grid/finite合同去掉证明恒真的load mask和不可能触发的耗尽判断；组合对照。局部非整块尾部保留保护，merge始终有至少K个finite候选。

候选保持原stage数和workspace，不修改host算法。stable ids与负数/subnormal/最低finite值必须完全符合原参考。

## 2026-11-24 回顾性文字更正

上面 r3 的“第二轮全向量 packed64”描述不准确。实际冻结的第二轮 parent `static_unroll_v2` 使用全向量 uint32 max 与 i32 min-index 两次归约，没有使用全向量 packed64。r3 实际执行的是：将原先的两次全向量归约改为分组 uint32 max/min-index，再对少量组胜者执行 uint64 packed 归约。预声明文字保留，候选、基线、结果和晋升规则未因此变动；报告以实际 TTIR 与生成器为准。
