# MoE routing 最终版 TTIR 消融实验

研究已验收的 r6 `batch4`，不修改原始 baseline、冻结源码、最终选择或 optimized 别名。官方 Triton 3.7.0 / H100 / FP32；完整30点（E=256/512，各15个tokens），G=8、选4组和8专家、稳定小索引tie，4 warps / 3 stages / 1 CTA。所有路线计完整原两阶段设备序列。实验合同见[ablation_plan.md](ablation_plan.md)。

## 各版本相对原始 baseline

| 版本 | 几何加速 |
| --- | ---: |
| 原始 baseline | 1.0000× |
| 组选择广播/reshape | 1.1855× |
| 再加 uint32 有序 key | 1.2566× |
| 再加循环后 gather | 1.3573× |
| 再加 8×8 稳定选组排名 | 1.4056× |
| 再加 stage1 top2 双值归约 | 1.4326× |
| 再加两阶段固定整数特化 | 1.5064× |
| 最终：再加 stage1 batch4 | 1.5421× |
| 最终去掉 membership 广播 | 1.4411× |
| 最终去掉有序 key | 1.4391× |
| 最终去掉循环后 gather | 1.4172× |
| 最终去掉稳定排名 | 1.4788× |
| 最终去掉双值归约（保留 batch4） | 1.5342× |
| 最终去掉整数特化（保留 tile/batch） | 1.5047× |
| 只替换最终 stage1 | 1.0535× |
| 只替换最终 stage2 | 1.4286× |

主测GPU2、seed42、5轮；独立进程GPU3、seed123、3轮；rep10ms，每版本100ms稳定阶段，同会话随机交错。全部逐shape A/A中位数通过[0.97,1.03]，完整矩阵及独立复测证据为A级；不同GPU保留各自的配对baseline，不比较两张GPU的绝对耗时。仅计预分配CUDA graph两kernel设备时间，自动频率、不清cache；不含分配、编译和CPU launch，不代表端到端或冷cache。

## 逐项加入的条件收益

| 对照 | 改动 | 条件加速 |
| --- | --- | ---: |
| baseline → add_membership | 组选择广播/reshape | 1.1855× |
| add_membership → add_ordered | 再加 uint32 有序 key | 1.0599× |
| add_ordered → add_gather | 再加循环后 gather | 1.0800× |
| add_gather → add_rank | 再加 8×8 稳定选组排名 | 1.0357× |
| add_rank → add_pair | 再加 stage1 top2 双值归约 | 1.0190× |
| add_pair → add_constants | 再加两阶段固定整数特化 | 1.0517× |
| add_constants → final | 最终：再加 stage1 batch4 | 1.0236× |

## 从最终版移除改动

以下为缺少改动版本耗时 / 最终版耗时，大于1表示该改动在其他最终变化已存在时有正贡献。

| 去掉的变化 | 条件加速 |
| --- | ---: |
| 最终去掉 membership 广播 | 1.0701× |
| 最终去掉有序 key | 1.0715× |
| 最终去掉循环后 gather | 1.0882× |
| 最终去掉稳定排名 | 1.0430× |
| 最终去掉双值归约（保留 batch4） | 1.0051× |
| 最终去掉整数特化（保留 tile/batch） | 1.0250× |
| stage1 batch4 | 1.0236× |

去掉pair reducer时保留batch4二维数据流，只恢复原始max1/计数/屏蔽唯一最大/max2归约。去掉整数特化时恢复运行时标量和地址stride，保留tensor形状与batch4；不声称支持新的运行时E/G/stride。去掉batch4复用add_constants，其他最终变化保持不变。

条件比值直接用同一shape、同一外层轮次的前版/后版时间，再取中位数和跨shape几何平均；不相乘历史轮平均。存在编译交互，加入和移除的条件收益可以不同，不能相加为总收益；低于约1%的差异不据此宣称稳定优化，未提供置信区间。

## 两阶段交叉消融

| 完整两次launch的组合 | 几何加速 |
| --- | ---: |
| 原始 baseline | 1.0000× |
| 只替换最终 stage1 | 1.0535× |
| 只替换最终 stage2 | 1.4286× |
| 最终：再加 stage1 batch4 | 1.5421× |

此处使用完整30点；历史stage_ablation/main和repeat的六点结果保留为B级，不将子集值混入本表。

## 编译资源与验证

| E | 路线 | stage1 regs/spill/shared B | stage2 regs/spill/shared B |
| ---: | --- | --- | --- |
| 256 | baseline | 18/0/0 | 32/0/1024 |
| 256 | add_membership | 18/0/0 | 25/0/16 |
| 256 | add_ordered | 18/0/0 | 28/0/16 |
| 256 | add_gather | 18/0/0 | 32/0/1024 |
| 256 | add_rank | 18/0/0 | 31/0/1024 |
| 256 | add_pair | 16/0/0 | 31/0/1024 |
| 256 | add_constants | 14/0/0 | 32/0/1024 |
| 256 | final | 14/0/16 | 32/0/1024 |
| 256 | drop_membership | 14/0/16 | 32/0/1024 |
| 256 | drop_ordered | 14/0/16 | 32/0/1024 |
| 256 | drop_gather | 14/0/16 | 28/0/16 |
| 256 | drop_rank | 14/0/16 | 32/0/1024 |
| 256 | drop_pair | 14/0/16 | 32/0/1024 |
| 256 | drop_constants | 16/0/16 | 31/0/1024 |
| 256 | stage1_only | 14/0/16 | 32/0/1024 |
| 256 | stage2_only | 18/0/0 | 32/0/1024 |
| 512 | baseline | 20/0/8 | 32/0/2048 |
| 512 | add_membership | 20/0/8 | 28/0/16 |
| 512 | add_ordered | 20/0/8 | 28/0/16 |
| 512 | add_gather | 20/0/8 | 32/0/2048 |
| 512 | add_rank | 20/0/8 | 32/0/2048 |
| 512 | add_pair | 16/0/16 | 32/0/2048 |
| 512 | add_constants | 13/0/16 | 31/0/2048 |
| 512 | final | 16/0/16 | 31/0/2048 |
| 512 | drop_membership | 16/0/16 | 29/0/2048 |
| 512 | drop_ordered | 16/0/16 | 32/0/2048 |
| 512 | drop_gather | 16/0/16 | 28/0/16 |
| 512 | drop_rank | 16/0/16 | 31/0/2048 |
| 512 | drop_pair | 18/0/16 | 31/0/2048 |
| 512 | drop_constants | 18/0/16 | 32/0/2048 |
| 512 | stage1_only | 16/0/16 | 32/0/2048 |
| 512 | stage2_only | 20/0/8 | 31/0/2048 |

资源表为C级编译事实，没有采集NCU动态指标，寄存器/shared变化不能单独证明加速原因。

每进程先运行独立verify_pipeline，再在runner中逐30点确认冻结源码和baseline文件入口LLIR/PTX/cubin与编译options一致、输出一致。每次3840项（16路线×30点×8分布）全部通过：group scores/weights的FP32原始位型与baseline完全一致、indices精确、CPU参考权重容差rtol2e-5/atol2e-6、canary、输入不变。final两阶段的IR哈希逐项等于已验收r6选择。

既有最终版[memcheck/racecheck](multiround/r6/report.md)已通过；本次新增消融路线没有额外运行sanitizer，不将最终版的结果外推为全部消融路线的sanitizer结论。

## 能否用普通 Triton 实现

| 改动 | 公开Triton表达 | 边界 |
| --- | --- | --- |
| membership | expand_dims/broadcast/reshape | 固定连续等长分组 |
| 有序key | bitcast、位运算、uint32 max | 有限分数和符号零正规化，仍需稳定tie |
| 循环后gather | tl.gather | 布局与编译收益需实测 |
| 8×8排名 | broadcast、比较、tl.sum | 用较小索引打破同分 |
| 双值top2 | tuple tl.reduce | 保留原舍入/最高两值的语义 |
| 整数特化 | constexpr | 同一固定合同内表达，JIT ABI可能不同 |
| batch4 | program_id、统一条件、二维arange/reduce | 原grid中非活动CTA返回 |

以上为源码可表达性说明，不是r6最终源码版的性能或机器码等价证据。已有[首轮源码实测](source_port/report.md)只覆盖当轮membership，不能代表本轮所有改动。

## 全部 shape

下表为固定同一会话的最终版时间与配对加速比；全部16版本原始数据见[CSV](ablation_v1/all_shapes.csv)。

| Shape (tokens, experts) | 基线时间（µs） | 优化时间（µs） | 加速比 |
|---|---:|---:|---:|
| (16, 256) | 7.2097 | 4.9840 | 1.446831× |
| (16, 512) | 8.1360 | 5.4030 | 1.506052× |
| (32, 256) | 7.3303 | 5.0293 | 1.457458× |
| (32, 512) | 8.1488 | 5.4378 | 1.496652× |
| (64, 256) | 7.4358 | 5.1342 | 1.446605× |
| (64, 512) | 8.4042 | 5.5726 | 1.508341× |
| (128, 256) | 7.5852 | 5.3510 | 1.417901× |
| (128, 512) | 8.6964 | 5.9152 | 1.470293× |
| (256, 256) | 8.4789 | 5.8707 | 1.444325× |
| (256, 512) | 9.9418 | 6.6642 | 1.492190× |
| (384, 256) | 9.5275 | 6.4014 | 1.485543× |
| (384, 512) | 11.4328 | 6.8015 | 1.681049× |
| (512, 256) | 10.6345 | 7.1443 | 1.488445× |
| (512, 512) | 12.9920 | 7.6048 | 1.707727× |
| (768, 256) | 13.0143 | 8.7664 | 1.484500× |
| (768, 512) | 16.3654 | 9.4330 | 1.734621× |
| (1024, 256) | 15.5713 | 10.5566 | 1.475091× |
| (1024, 512) | 19.9710 | 11.4572 | 1.743117× |
| (1536, 256) | 21.0500 | 14.5252 | 1.449509× |
| (1536, 512) | 27.3418 | 15.7308 | 1.739504× |
| (2048, 256) | 26.5819 | 18.7143 | 1.419401× |
| (2048, 512) | 34.5225 | 19.8597 | 1.738951× |
| (3072, 256) | 37.7724 | 26.5627 | 1.421859× |
| (3072, 512) | 49.1983 | 28.1969 | 1.744832× |
| (4096, 256) | 48.6883 | 34.5900 | 1.407663× |
| (4096, 512) | 63.7468 | 36.4580 | 1.748499× |
| (6144, 256) | 70.1035 | 50.2065 | 1.396309× |
| (6144, 512) | 91.8943 | 52.7581 | 1.741805× |
| (8192, 256) | 91.9724 | 66.0312 | 1.392752× |
| (8192, 512) | 121.0521 | 69.3266 | 1.746112× |

证据：[主测](ablation_v1/results/main/measurement/performance.json)、[独立复测](ablation_v1/results/repeat/measurement/performance.json)、[主测数值](ablation_v1/results/main/measurement/accuracy.json)、[复测数值](ablation_v1/results/repeat/measurement/accuracy.json)、[编译资源和哈希](ablation_v1/results/main/measurement/compile.json)、[冻结谱系](ablation_v1/manifest.json)、[审计汇总](ablation_v1/summary.json)。

复现：先保留CUDA_VISIBLE_DEVICES给独占空闲卡，分别运行`bash moe_routing/run_experiment.sh NEW_MAIN 42 5`和`bash moe_routing/run_experiment.sh NEW_REPEAT 123 3`；两次均完成后，`python moe_routing/report_ablation.py --main NEW_MAIN/measurement --repeat NEW_REPEAT/measurement`重新核对并生成报告。固定使用`/root/venvs/ir-ops-dev/bin/python`。新输出目录不可存在；既有候选与结果不覆盖。
