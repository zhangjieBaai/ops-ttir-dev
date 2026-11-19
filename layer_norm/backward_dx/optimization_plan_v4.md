# backward dX 第四轮：全矩阵平均超过 1.10×

目标：M4096、N1024..15872步长512，全30点对冻结原始baseline的几何平均和算术平均都超过1.10×。保持FP16输入/输出、FP32运算、五组options/grid/ABI、官方Triton3.7.0链路、原数值阈值；不引入内联PTX或更改源码/低层IR。先保存当前parent及pipeline核查结果。

第三轮最终组合约1.028×，主要中大N未改善。NCU大N较高DRAM吞吐，单纯压寄存器/重载失败；本轮检验缓存策略和保留输入的规约表达。所有数值验证先于性能，原135例、48边界、72例saved-stats压力；性能6点初筛、全30点7轮20ms主测和5轮10ms独立seed123复测，A/A为0.97..1.03。家族晋升仍需两次对parent>=1.02、最差>=0.95且对baseline>=1.02；最终整体目标另核验，达不到不能声称完成。

首批6候选（最多12候选预算，后续候选先登记）：

1. r4_cg：parent所有向量load添加cg，保留其余计算和store；测试L1绕过。
2. r4_cg_stream：同1并对DX使用cs。
3. r4_fold256：parent的两个规约输入reshape为[B/256,256]先联合轴0规约，再规约到标量；不增加reload。
4. r4_fold1024：同3，折叠宽1024。
5. r4_keep1024：窄分组统计，但将已有xhat/wdy保留到输出，完全不重载；使用联合规约和行级输出系数。
6. r4_keep2048：同5，分组宽2048。

保存各候选资源和全部失败；若无足够收益，按实测改假设，不放宽精度或更改计时口径。NCU仅为诊断，不与graph时间混合。

C0修复：fold版本局部SSA与外部%x冲突，解析失败；原IR和日志保留，fold256_v2/fold1024_v2只重命名规约区域局部结果值，不改变算法。

第二批4个（累计10/12）：r4_evict仅对X/DY用evict_first、W用evict_last；r4_contig2/4对make_range添加保守的contiguity提示，降低下游向量宽度以检验调度/资源是否改善（不改变真实访问索引）；r4_affine保持中心化规约，输出展开为两层FMA，测试能否用原始X替代长期存活的xhat，新增压力测试须通过，不能以平均误差替代逐元素门槛。

第三批2个（12/12）：r4_maskfold/r4_center_maskfold。有限数据与有限保存stats下，masked DY/W都用零填充，则inactive wdy严格为零，统计规约中的xhat无需再select为零；输出inactive不写。删除两处向量select，保留实际load/store掩码。不改变有效元素、参考或精度；第二个在r3_centered_stream基础上做相同消除。若rstd/mean非有限不在本轮契约内。

最终完整矩阵入围固定为r4_evict（初筛N3072局部改善）和r4_center_maskfold（初筛N1024局部改善），其余候选明显整体或大N退步；只对这两个做主/复测及组级验收。目标1.10单独核对，不因为局部晋升而宣称完成。
