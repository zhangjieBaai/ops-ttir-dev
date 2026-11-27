# Attention dK/dV 第二轮：CTA 内输出子块与循环状态

沿用父目录的全部12点、FP16/FP32合同、数学参考门槛、4warps/5stages、设备ABI/grid、一个kernel及无工作区。baseline为原始冻结TTIR；parent为D64的r1_dv_late、D128的baseline。原文件不改，本目录保存字节副本和哈希。

本轮只直接修改TTIR，不从优化Python kernel导出，不改下游IR/机器码，不用inline PTX。父版本源码pipeline先逐点核验，完整options一致。

候选预算为五项：`rows64_v1`、`rows32_v1`、`remat_ptrs_v1`、`rows64_remat_v1`、`rows32_remat_v1`。前两项在每个原128-key CTA内部串行处理两个64-key或四个32-key输出子块，减小K/V和dK/dV活动张量；grid仍为N/128。每个子块沿用原查询块遍历顺序：causal仍从原128-key块起点开始，以原8次16-query masked迭代，再接原32-query unmasked循环，避免改变每个输出的累加次序或越界。代价是重复读取Q/dO和增加串行子块循环。

`remat_ptrs_v1`将两个循环携带的Q/dO指针张量改成由现有scalar curr_m与初始地址重算，循环仅携带dk/dv/curr_m；两组合测试输出子块与指针状态同时改变的效果。所有签名、加载范围、mask、FP16舍入点和dot内K顺序不变。即使数学看似等价仍必须与baseline输出逐位相等，同时满足原FP64数学参考。

精度先覆盖原28输入（12正常+4组各4压力），再初筛3轮10ms。按原四个完整D/causal组，以parent/candidate增量几何均值冻结候选；后主测7轮20ms、独立seed123复测5轮20ms，500ms对等预热，随机版本顺序、A/A门槛[0.97,1.03]。增量两轮均>=1.02、组内最差>=0.95且sanitizer通过才晋升，否则保留parent。报告还列同轮baseline/candidate累计收益及失败/回退。

设备GPU4独占；Python固定/root/venvs/ir-ops-dev/bin/python。寄存器/spill变化是C级静态事实，不能替代性能验收；分块和指针重算也可用公开Triton循环表达，本轮不宣称TTIR独占能力。

用户在候选编译pilot期间要求每类扩展至约30点。本轮正式矩阵冻结为32点：每个D/causal组使用(B,H,N)=(1,4,128),(1,4,256),(1,4,512),(1,4,1024),(1,4,2048),(2,8,1024),(2,8,2048),(4,8,2048)。旧12点只作为历史/调试参照，不混入正式矩阵；另每组N128保留zero_do/constant/large压力，共44个输入。H4/8、N128倍数与原IR属性一致，仍逐点以原冻结源码导出结果核查。

基于最初几项筛选显示动态子块循环回退，预算内增加第六项rows64_unroll_v4：将固定两次64-key子块循环直接展开成两个SSA代码副本，排查嵌套循环对流水/调度的影响。原五候选与样本保留；展开版仍需全44输入数值和完整32点筛选，不能用小点性能代替。

计时修订：首次五候选screen32的大D64非因果点A/A中位0.9434超门槛，整轮不用于正式选拔；展开项screen_unroll32在完整132项正确性后，部分计时中止。两份完整正确性记录按shape/分布/名字和输入哈希合并为352项，源码/IR没有变化。新完整screen32_stable对每条路线除了500ms初始预热外，在每次实际计时前统一100ms同kernel预热；主测/复测沿用这一协议。旧样本不混合，失败记录见screen_disposition.json。展开v4曾因纯数字SSA名字加后缀解析失败，v5改成字母开头名字，原v4保留。

预热实现v2：检查官方testing.py发现do_bench_cudagraph内部10次retry，rep500/100不等于实际500/100ms，造成明显额外开销。保留并停止未完成screen32_stable，不用于选拔；重新完整screen32_stable_v2。仅预热改为每路线复用32-launch CUDA graph，并用CUDA event确认实际初始>=500ms、每次计时前>=100ms，逐项保存实际时长；真正计时仍调用原do_bench_cudagraph(rep10/20)。所有主/复测使用相同v2方法，候选和输入无变化。
