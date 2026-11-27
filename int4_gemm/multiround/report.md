# INT4 第三至第五轮：顺序优化结果

固定30点与第二轮配置，三轮确实从当时已验收parent继续生成新TTIR，而非把多seed复测当作新轮次。
30个唯一shape每轮均重新核验，三轮不是90个不同shape。

| 轮次 | 实测候选 | 主测增量 | 独立复测增量 | 晋升 |
|---|---|---:|---:|---|
| [r3](r3/report.md) | group128 | 1.030644× | 1.030233× | 接受 |
| [r4](r4/report.md) | pid_g4 | 0.994102× | 0.994036× | 保留parent |
| [r5](r5/report.md) | preload_pair | 0.988765× | 0.988884× | 保留parent |

最终已验收路径相对第二轮anchor：主测/复测 **1.030287× / 1.030268×**。
相对原始baseline累计：**2.518282× / 2.517380×**。

r3使用固定GROUP128与连续stride的既有合同；r4探索合法尾组的PID重排；r5探索加载次序及两个量化group的循环展开。
在M128,N11008,K4096上，group128相对第二轮anchor的寄存器96→108、spill4→0，shared仍24576B。变量除法消除同时改变了分配/调度，不能将所有收益只归因于某条指令。
r4三个PID候选有16/30点的可执行.text与parent相同（单M tile适用恒等映射）；r5 preload_pair有17/30点.text相同。少量变化与计时波动分开记录；全量资源/静态PTX计数见[static_analysis.json](static_analysis.json)，不是动态NCU证据。
每轮完整精度后筛选，冻结后主7/复5，A/A门槛不变。原baseline源码/TTIR产物一致，全部工具和候选哈希已记录。
各轮新候选均做memcheck/racecheck，零错误、零警告；失败/退步保留。所有候选都可用公开Triton表达，未做源码版本性能比较。

[协议](optimization_plan.md) · [机器可读汇总](summary.json)
