# Sum 消融与公开 Triton 表达验证

2026-11-20。继承原30个FP32连续全量sum性能点、误差标准、官方Triton3.7.0环境。
用户要求每项改动的收益以及能否用Triton实现。本轮不搜索新最优，不改现有baseline、候选或选择。

## 冻结对照

所有版本在同一GPU、同一输入、预分配mid/out、同一原生TTIR文件编译/启动/graph计时下比较。
候选：baseline，r1_i32，r1_fold，r1_i32_fold，r1_cg，r1_cs，r2_tail，
wide_two（小shape扩大归约到全量、grid1，仍保留第二个单元素拷贝kernel），
r2_single16384，single_tail（小shape single、大shape tail），python_single（公开Triton实现导出后文件编译）。
阈值固定16384；large路径baseline。wide_two是有效sum，用于将“扩大归约范围+减少CTA”与“移除第二kernel”拆开。
不能把扩大宽度而不改grid的无效算子当合法消融。wide_two第二kernel用原冻结sum_kernel_2的mid_size=1特化。

每项对baseline配对，另直接配对：i32→i32_fold、fold→i32_fold、wide_two→single、
single→single_tail、single→python_single。差分取直接配对比，不相减历史或不配对时间。
报告30点、受影响区间15点、逐点、算术/几何均值及95%bootstrap区间。
这些因素有交互，收益百分数不能相加。single包含新启动契约，不称固定启动TTIR收益。

## Triton可表达性证据

另写公开`@triton.jit`实现：program_id保留i32或显式to(int64)，cache_modifier，
reshape+两级sum，CTA一致if与masked/unmasked load，next_power_of_2(n)+grid1直接out。
每项实际编译、导出TTIR、同文件入口重编译、数值验证，检查完整options/target、stage产物hash与资源。
对照直接TTIR候选的ELF可执行.text字节：相同则说明该编译组合生成相同指令；不同仅能说计算意图可表达，
不声称生成完全相同代码。所有公开实现导出物不得替代已有直接TTIR实验候选或改写其历史来源。
python_single纳入正式计时，与已选single直接配对；其他公开实现如字节相同可以复用对应设备代码结论，
但不虚构独立性能样本。
本地官方semantic.py已确认tl.load公开参数仅接收.ca/.cg/.cv，不接受.cs；
因此cs保留TTIR消融，另保存公开API失败探针，不伪造py_cs等价API支持。

## 验证与测量

运行前隔离空闲GPU4（实际不可用则记录同次全部任务的新UUID），规范化UUID设备锁，1秒采样其他PID和频率/功耗。
先30点源码baseline pipeline，再全部版本数值：213实例小输入完整压力/特殊值、大输入原压力、尾部canary/输入未写。
wide_two只承诺mid[0]=全量sum，检查其余mid未写；single检查所有mid未写；普通two检查部分和。
public等价实现每项同样验证输出与中间约定，保存与直接TTIR的逐位比较，NaN按分类。
正式测量前检查数值/pipeline的实际TTIR+cubin哈希，warmup100ms，rep20ms，
main30点7轮、独立进程seed123复测30点5轮；所有比较edge随机顺序、对内随机，全部raw replay值保留。
每点A/A，区间[.97,1.03]；若越界整轮标记无效调查，保留原数据，不挑选样本。
不清cache、不锁频、重复buffer，完整设备序列计时，Python/分配/编译/加载不计。
结果只针对现有工作负载，无新holdout；独立seed不等于新shape独立验证。
新增wide_two、single_tail和公开Python变体运行memcheck/racecheck；失败不支持其收益结论。
精度与已有artifact哈希相符的历史数据可作为背景，不混入新配对汇总。

最多11个正式版本+公开等价映射，2次完整进程测量。无参数搜索，GPU预算30分钟。
所有负结果保留，原始数据、直接配对、可运行Triton例子与报告交付，pre-commit和readability通过，最终独立审查。
