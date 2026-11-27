# Norm三个顺序优化轮

继承../../complex_suite/multiround_plan.md；anchor=round2的reciprocal_early，沿用expanded30及原量化合同。

- r3：把rho和reciprocal合成一个scalar gain；或消去量化比值中的正rho，使用127/max(abs(u))。scale仍按原统计公式输出，零行保护不变，必须通过原±1code/scale/重构和极端输入合同。
- r4：利用abs(u)有限且非负，用单调uint32 bitcast做amax；比较sum/max混合类型联合reduce、分开的float-sum/uint-max和分开float对照。不是无条件更快。
- r5：输出阶段重载/重算u；比较full-row重载、1024块流式量化，以及预先限定N>=4096且rows>=512的流式策略。统计和Z仍完整，输出不遗漏；额外访存与活动状态相权衡。

上述结构公开Triton均有表达方式。本轮只测TTIR，不把第一轮的源码回写结论直接继承到新候选。
