# backward dX 第五轮：TTIR 内部多行调度

目标延续全30点几何/算术均值>=1.10×，不更改原始baseline、FP16/FP32数值契约、参考误差、外部grid=(M,1,1)、ABI、warps/stages、官方Triton3.7.0编译与launcher。

第四轮12种方案的正确性/6点初筛未提供整体突破，原始全部记录保留，不把B级初筛提升为最终收益。第五轮显式扩展此前“一CTA一行”内部映射约束：CTA i在TTIR内处理i*R+[0,R)行，M由tt.get_num_programs x获取，i*R>=M的CTA退出；所有有效地址仍严格行/列mask，无跨CTA通信。外部启动流程与baseline完全相同。每个候选都重新跑小M/非整除R/边界/padding，不能把前轮正确性继承到此算法。

先4种：r5_rows2_tile512、r5_rows4_tile512、r5_rows8_tile256采用两遍窄列循环，规约前中心化；r5_rows2_full直接双行完整向量，避免第二遍重载但可能寄存器溢出。全部使用FP32统计，最多本轮6候选。较窄tile力求把warp分给独立行，具体布局由官方后端决定，保存TTGIR核查，不指定layout/inline PTX。

先135 C1、72高偏移saved-stats测试，随后6点3轮10ms初筛。合格候选按全30点7轮20ms、独立seed123/5轮10ms、A/A阈值0.97..1.03，以及memcheck/racecheck验收。沿用分组晋升门槛vs parent>=1.02、worst>=0.95；整体1.10目标单独判断。
