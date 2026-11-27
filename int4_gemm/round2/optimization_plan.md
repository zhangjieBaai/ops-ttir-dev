# INT4 第二轮：当前最佳 TTIR 上的增量优化

父版本为原7点各自的 `scalar_unsigned_v1`。固定原shape、group128、配置、ABI/grid和数值合同：FP16反量化舍入及dot累积次序保持，输出必须与原baseline逐位相同。仅在本目录直接改TTIR；不改旧源码、IR、选择和结果。

六个预声明候选：

1. `r2_narrow`：将重复 `[BK/2,BN]` scale/zero load显式改为 `[1,BN]` load、扩展和广播。
2. `r2_full_k`：K是128倍数，BK64/128且每次k0按BK步进，因此删除activation/packed的恒真K mask，保留M/N尾部mask。
3. `r2_narrow_full_k`：前两项组合。
4. `r2_a_late`：在组合上把A地址/读取整段移到解包反量化后、dot前，探索活动状态与流水取舍。
5. `r2_scale_cg`：在组合上仅对scale/zero窄load测试.cg缓存策略，作为消融。
6. `r2_group_pair`：BK64时将原两次64步迭代显式展开为一个128步外循环，组内只读取一次scale/zero；两个64-K dot按原次序连续累积。BK128点沿用narrow/full-K组合。K/group均128整除，没有不完整pair。

候选保持同一父版本图和原完整后端options，不调整num_stages/num_warps或maxnreg，不内联PTX、不修改下游产物。全部候选先编译和7点×5压力分布精度，再3轮/10ms筛选。按完整7点增量geo冻结唯一候选，再7轮/20ms主测和独立seed123的5轮/20ms复测。每轮随机顺序对照original baseline、parent、candidate，记录直接parent/candidate增量与baseline/candidate累计比值及A/A。最终要求两轮增量geo≥1.02且最差≥0.95、A/A在[0.97,1.03]、sanitizer通过；否则保留parent。所有失败保留，不据复测重新择优。

GPU0独占，官方venv Triton3.7.0；完整baseline源码/file LLVM/PTX/cubin及options核验，所有计时kernel从文件入口原生编译。保存输入seed/hash、产物hash、ELF.text、资源及全部原始samples。公开Triton原则上也能表达这些数据流/循环改写，但本轮不另做源码回写实验。
