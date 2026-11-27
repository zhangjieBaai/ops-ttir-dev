# 第六轮TTIR优化

2026-11-24，继续从r5最终有效选择出发。每类执行一个新的有界搜索轮，保持原152个唯一shape、dtype、ABI、grid、kernel数、workspace与编译配置。原始baseline、第二轮anchor、r5 parent及新候选直接同轮对照，不乘历史加速比。旧实验不改；新候选写入各算子multiround/r6/。

延续multiround_plan.md：先数值与源码/file baseline pipeline，完整矩阵初筛3轮rep10，冻结候选后主7轮rep20、独立进程seed123复5轮rep20。A/A中位[0.97,1.03]；主/复增量geo>=1.02、最差>=.95及sanitizer通过才晋升，否则保留parent。失败和机器码恒等候选如实保留。仅改TTIR，无源码生成优化、inline PTX或配置调参。

GPU0/2/4/6/7分别独占INT4/MoE/Attention/Norm/TopK。Attention沿用1200MHz条件并在finally恢复自动频率，其余不主动锁频。

- INT4：先join uint8 nibble再统一FP32解码；使用整数构造精确0..15的FP32；两者组合。保留FP32减乘、FP16舍入与dot累加次序。
- MoE：stage1同一CTA处理2/4/8个连续组，原grid下仅对应组首CTA执行；stage2保持parent。比较组内warp利用率与空CTA调度成本。
- Attention：D128在已接受64-key结构上后移dV；或在相同CTA中将dV、dK分成两遍查询遍历，以重算概率换更少同时活跃状态。D64保留parent，不改pipeline depth。
- Norm：保持float clamp后直接调用公开libdevice float2int_rn再截断i8；将clamp移至整数舍入之后；将统计输入reshape成4组后先组内联合sum/max、再联合归约四个结果。前两项保持舍入逻辑，最后一项改变sum顺序，必须通过既定FP64误差及极端测试。合法有限FP16域下未clamp量化值约[-127,127]，远离i32溢出；仍显式整数clamp输出范围。
- Top-K：local保持r5；merge按chunk的局部有序K列表做头部归并。每次仅维护每个chunk当前头部的key/id和cursor，选中后推进对应chunk并加载下一项；分别使用uint32 max+min-id、packed uint64稳定max两种选择。固定workspace、grid、输出排序/tie及±0数值合同。merge tail必须mask，耗尽列表以-inf/INT_MAX处理；每个chunk内已排序是该结构的前提，不支持任意未排序中间输入。

Norm与Top-K使用现有runner函数，通过新CLI允许r6，旧runner不改。任何语法/数值失败版本保留，并用新版本名修正。第一轮源码回写等价结论不自动延续到这些新结构。
