# backward dX 第三轮：减少规约前计算和改变联合规约表达

继承dX-only、五组baseline/options/grid/ABI、FP16输入输出/FP32累加、无别名及全部数值容差；不改前向、baseline、旧候选、编译链/选项，不用内联PTX。先核查pipeline并冻结本轮parent。

第二轮证据（C级）表明纯重载降低寄存器但大N实际DRAM读取260→502MB，动态循环整体失败。本轮不再重复全输入重载，也不为了性能把FP32中间值截到FP16。

最多8个候选，第一批4个：

1. r3_centered：规约sum((x-mean)*wdy)与sum(wdy)，把rstd因子移到行级系数，输出为rstd*wdy - rstd*c2 - (x-mean)*rstd^3*c1。保留中心化，不增加load；检验能否减少逐元素乘法和中间状态。
2. r3_centered_stream：在1上只对DX设置cs store，检验输出缓存策略。
3. r3_rawmoment：规约sum(x*wdy)与sum(wdy)，行级用FMA计算sum((x-mean)*wdy)；输出仍用中心化x表达。取消规约前逐元素减mean和乘rstd，但存在抵消误差风险，必须接受原容差和新增高偏移压力测试，失败即淘汰，不放宽容差。
4. r3_joined：把两个统计向量join为[B,2]后一次轴0规约，用两次单元素gather/reduce提取结果，改变并行布局表达；不指定warp指令或手改lowering。

剩余4个只根据C0/C1/初筛证据追加，提前登记；停止8个预算。不更改num_warps、行映射、dtype或FP32累加要求。

全部135原C1、第二轮48边界以及新增数值压力先于性能：高offset(32/256/4096)加小扰动、常量带单个可表示outlier、DY/W强相关或交替符号，小M和大行宽，比较saved-stats FP64与native梯度。新增检查对baseline也跑，baseline未通过时先调查，不能只放宽候选。

筛选原6点3轮rep10ms/A-A，前两名或证据足够的局部候选全30点7轮rep20ms，主测后冻结组级选择；独立seed123/5轮rep10ms只验收。各组对baseline和parent都报告，晋升须对parent几何>=1.02且最差>=0.95、对baseline几何>=1.02，AA中位数0.97..1.03。最终memcheck/racecheck、pipeline及别名身份验证通过后晋升，否则保留parent。NCU固定base clock/cache-control=all，仅诊断不替代常规graph计时。所有失败和负结果保留。

追加第二批2个（总6/8）：r3_group_center1024 / r3_group_center2048，沿用grouped展开加载与输出，规约前使用masked(x-mean)，c1在标量侧乘rstd²，输出仍整体乘rstd；分别分组1024/2048，不增加运行时分派。第一批screen_batch1的N8704 AA失败，整份不用于晋升。原矩新增8例baseline通过但候选失败，淘汰。

额外压力检查的参考修正：原stress_batch1中baseline和parent均17例失败，检查是由于reference()的autograd路径重算mean/rstd；大偏移低方差下与实际输入saved stats不同。保留原始结果，新增--saved-stats运行，以aten.native_layer_norm_backward显式传入相同mean/rstd，同时仍比FP64 saved-stats公式、原容差不变；重算统计量的native差异仅单独记录。这是独立输入stats接口下的参考对齐，不改变baseline、数值阈值或算子输入。常规135用例仍保留原native autograd要求。

第一批初筛整体AA失败后补跑screen_batch1_repeat，全部AA通过。小N的centered_stream对parent出现约3%–7%的候选收益，依据预设前两名/局部候选规则，作为第二个完整矩阵候选，运行full_stream(7轮/20ms)，冻结provisional_stream后运行repeat_stream(5轮/10ms,seed123)。grouped试验的独立decision.json保持原义；全轮最终决策写selection_final.json。禁止根据复测改选centered或其他版本。若通过则只晋升通过的family，其他family保留本轮r3_parent；另做stream的独立数值检查、mem/race和最终alias验证。
