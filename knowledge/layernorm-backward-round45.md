# LayerNorm backward dX 第四、五轮：10%整体目标与反例

目标是同一30点矩阵对原baseline的几何和算术平均都超过1.10×。固定官方Triton3.7.0、FP16输入输出/FP32运算、同一外部grid/options/ABI/launcher。最终是否达标必须看[完整报告](../layer_norm/backward_dx/report_round45.md)和[目标核验](../layer_norm/backward_dx/results/round4/composite_summary.json)，不能拿单点10%当成平均10%。

## 当前可用经验

- **缓存淘汰策略要按输入角色区分。** r4_evict对X/DY用evict_first，对W用evict_last，W在不同row间复用。N2560/3072出现局部收益，但部分中大N退步；只按整family主测冻结、独立复测验收，不能全尺寸替换。最终精确数值见A级[主测](../layer_norm/backward_dx/results/round4/full.json)、[复测](../layer_norm/backward_dx/results/round4/repeat.json)。
- **数学上冗余的mask不一定有性能收益。** 在有限X/W/DY/mean/rstd范围，给越界W/DY填0，wdy已为0，因此可删去统计向量的重复select，同时保留所有内存边界mask。小N可改善；大N的center_maskfold寄存器220→179但仍慢，不能由指令更少推出更快。参见[C级资源](../layer_norm/backward_dx/results/round4/binary_analysis.json)和上述主/复测。

## 明确的负结果

- **reshape折叠可能引入代价高的layout转换。** r4_fold256_v2/r4_fold1024_v2保持数据、不做reload，但b8192 shared从32B升到16KB。6点筛选几何约0.2475×/0.5642×。这是B级负结果；不能称所有分层规约都慢，也不能把共享内存容量本身当成唯一根因。[筛选](../layer_norm/backward_dx/results/round4/screen_batch1.json)。
- **保留整个分组状态会把寄存器推满。** r4_keep1024/2048在b16384分别254/255寄存器，N15872筛选仅约0.7906×/0.8886×，虽然没有重载仍不划算。切分张量并不等于降低总存活状态。
- **缩小向量宽度可能增加地址/寄存器开销。** make_range的保守contiguity2/4确实改变下游layout，但b8192寄存器80→128/127，未取得整体收益。真实地址连续性和声明的保守分析信息要区分；本实验没有伪造更强的对齐前提。[筛选](../layer_norm/backward_dx/results/round4/screen_batch2.json)。
- **输出代数展开也有抵消误差。** r4_affine保持中心化统计，但输出展开为两层FMA，在常规135用例通过后，72例压力测试仍有3例失败。不能只检查统计规约的稳定性，输出表达也必须验证。原始记录：[stress_batch2](../layer_norm/backward_dx/results/round4/stress_batch2.json)。
- **0 shared/无CTA同步不是充分条件。** 第五轮让一个CTA处理多行，窄循环版32–48寄存器，8行版shared为0且warpsPerCTA=[8,1]。但四个多行候选6点几何均低于0.83×；完整双行版b16384有142 spills。B级负结果见[筛选](../layer_norm/backward_dx/results/round5/screen.json)，不能据此宣称其他多行策略永远无效。

第五轮只改变TTIR内部行映射：原外部grid=M不变，CTA i处理i*R..i*R+R-1，超过M退出，M来自get_num_programs；独立验证小M、非R倍数的行数、padding和mask。其正确性不能继承原一CTA一行版本。

## 验证与归因边界

所有候选先原135个正确性用例和72个saved-stats压力用例，再测性能；失败的affine不计时。原始折叠候选SSA名字冲突发生在C0，保留失败IR与日志，用_v2修复，未将其计为数值失败或新算法。

NCU采用base clock/cache-control=all，常规测试采用预分配CUDA graph；两种时间分开。查看实际DRAM读写，不能从load条数推断二次读取必然来自HBM，也不能从occupancy/寄存器/barrier任一指标推断加速。后端产生的布局转换、循环与活跃状态要一起核查。

本轮NCU确认8行窄循环在N15872的DRAM读取260.15→520.23MB，active warps 12.23%→46.25%，耗时131.62→219.01µs。occupancy改善同时访存代价上升是实测事实，不代表所有多行实现必须读取两遍HBM。cache提示可能改变写回时机，NCU kernel区间的DRAM写字节少不能当作逻辑输出store被消除。

完整主测/独立复测最终组合的几何平均分别1.0359×/1.0379×，算术1.0367×/1.0389×；尚未达到1.10×。最终入口额外3轮直接复核几何1.0373×。晋升b1024/b2048的掩码消除版和b4096的缓存提示版，大两组保留旧选择。以上A级数字与单点/6点初筛不可混用。
