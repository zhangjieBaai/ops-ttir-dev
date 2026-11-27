# 大 N partial Top-K：固定两阶段直接 TTIR 实验

新研究 baseline 采用 FlagGems `ops/topk.py:topk_stage1_kernel` 的分块重复选择结构；第二阶段在候选上再次选择，不全排序。为明确跨块 tie 行为，用原始全局 index 的 min 作为同分优先级。不是该文件原样提取或已充分 autotune 的生产实现；不把 torch.topk 作为唯一性能对照。

合同：连续 FP32 `[rows,N]`、有限值（包括 ±0）、无 NaN/Inf，降序 K 个 value 与 INT32 index，tie 取较小原始 index，输出有序且全部精确相等。N>=K。两个 kernel，block=1024，4 warps、3 stages；中间保存每chunk K对候选，merge block=next_power_of_two(chunks*K)。两版完整设备序列、配置、ABI、grid、workspace均一致，不归因为多kernel算法改变。

冻结6点 `(rows,N,K)`：`(1,65536,8),(16,65536,8),(128,65536,8),(16,262144,8),(16,262144,16),(4,65537,8)`。最后一点含不足K个元素的尾块，padding候选为-inf且index为INT_MAX，合并不会选中。压力：normal、all_equal、integer_ties、signed_zero、increasing；小rows另CPU稳定排序检查。

候选最多3个：`joint_v1` 将每轮 max→比较→min-index 改为 tuple 稳定 argmax reduce；`packed_v1` 将FP32单调排序key与反向index装入64位做单max（±0归一化、最终value仍从未改bit pattern的输入取，若不安全则拒绝）；`joint_cg_v1` 对 joint input load改.cg测试缓存。只改冻结TTIR，不经候选源码导出。主选方向为joint，不保证正收益。

GPU7独占，官方Triton3.7.0 venv；pipeline检查先于候选。3轮10ms初筛、7轮20ms主测、独立进程5轮20ms复测，预分配CUDA graph完整两kernel设备时间。所有精度先于性能，保存原始配对样本、A/A、IR/后端hash/资源。初筛按全矩阵geo固定候选，主/复测均>=1.02、最差>=0.95和AA在0.97–1.03才晋升，最后memcheck/racecheck。未测范围不外推。

候选生成前修订：64-bit packed路线本轮暂不实施，使用 `local_joint_v1`（只改local，merge保留baseline）作归因消融；总计joint/local_joint/joint_cg三种，不改变合同或性能矩阵。

初筛后的有据扩展：joint系列全矩阵geo约0.81；v2将float max改为有序uint32 max，但每轮重算key仍约0.98。v3在冻结TTIR内把key编码移出循环，并把loop-carried values从FP32改为有序uint32，保留min-index tie规则；±0编码前归一化，padding与已选值用-inf对应key。v3初筛所有6点获益，主测命令固定ordered_carry_v3为最终候选，v2/joint同场作为负消融，复测只验v3，不按复测择优。以上未改变kernel数、工作区、grid、options与ABI。

最终合同澄清：values精确指浮点数值相等（±0等价），不要求保留零符号位。原baseline的max也不保证输出零位型对应选中index；v3归一化±0后仍严格遵守最小原始index tie规则。v2主/复测新增正负subnormal/最低finite用例，全部通过。
