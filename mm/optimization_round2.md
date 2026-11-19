# MM 第二轮：借鉴匹配配置的 FlagGems IR，计划 v1

目标组为bm64_bn128_bk64_g8_w4_s4（768/896/1024/1152/1280/1408³）。用户指出1152³/1280³配置相同但FlagGems更快，要求研究其IR并尝试优化本组TTIR。来源为comparison/results中实际FlagTree/native产物，不将旧跨环境计时当作本轮优化收益。

不改变baseline.json及其绑定文件、六组baseline、dtype/layout/数值容差、num_warps/stages/CTAs、grid规则、9参数ABI、workspace约定与公共compile_ir/launch。不重新autotune；仅直接改TTIR，官方Triton3.7.0/指定venv/H100执行；其他五组选择不变。旧r1文件/别名/选择保留，不覆盖optimized.ttir；若本轮接受新版本，提供独立round2选择与公共入口，显式记录来源hash，避免破坏旧round1复现。

## 诊断与可迁移机制

- 对1152/1280的三套实际产物（FG/FlagTree、教程/FlagTree、教程/upstream）记录完整输入/产物hash、配置、资源与静态TTIR/TTGIR/PTX指标。相同配置不等于相同访存算法；TMA、参数ABI、编译器和实际资源差异分开陈述，不把寄存器变化当性能因果证明。
- FG host descriptor入口有不同参数，不能直接替换。device tt.make_tensor_descriptor若产生global_scratch并要求新allocator，属于当前workspace边界之外；只作编译可行性诊断，不进入功能排名，也不偷偷添加allocator或改launch约定。
- 借鉴FG的program ID公式：pid_m=group_base+(pid%group_size)，代替教程((pid%group_width)%group_size)。每个完整列chunk恰有group_size个连续pid，因此结果是同组内行的排列，覆盖同一tile集合，grid不变；需要测试partial-group。
- 借鉴FGdescriptor边界零填充：移除A行/B列索引取模，改为原始连续索引及M/N/K组合load mask，无效元素显式0。有效输出行列的dot输入不变；masked lanes可构造越界地址但不得访问，需memcheck验证。
- 借鉴FG只携带accumulator的循环：从k计算本轮A/B地址，避免指针iter_args和每次递增。保持dot顺序、掩码、tensor大小不变，核查lowering、资源和计时。

## 预算与验证

最多8份本组新功能候选：pid_direct、zero_pad、absolute_address、zero_pad+absolute、zero_pad+pid、absolute+pid、三者组合；最多1个有诊断依据的修正版本，旧同名文件不覆盖。r1_canonical已证明机器指令同baseline，可复用为A/A，但本轮重新检查其可执行section hash；不再试首轮已证伪的cg/unroll hint或直接重复unsigned。诊断性device descriptor不计入功能候选预算，不启动有新workspace的kernel。

在空闲GPU1（如被占用则选择另一空闲卡并记录，同轮不得混卡）先跑完整31点baseline pipeline。C0全options/target/ABI一致，拒绝候选global_scratch_size或workspace元数据变化；C1复用experiment_round1.validate，覆盖组全部6个性能点、原family_cases边界/压力、K分支，另增加M/N小尺寸与partial-group、padding=16。输出NaN sentinel、canary、输入不变、有限性，native FP16/禁TF32 FP32，小尺寸CPU FP64；atol=.01、rtol=.001。失败保存并禁止计时。

固定初筛4点：768/1024/1152/1280，3轮rep20ms，配对随机顺序、预分配输出、预热同步及A/A。候选稳定后在本组全部6点做7轮主测，再用seed123独立进程5轮复测，保持相同候选hash。896/1408在本轮初筛不暴露，但已参与过历史实验，不称全新独立holdout。若改变候选依赖其验证结果，这些点标为tuning，不挑子集选优。

接受标准沿用首轮本组标准：两进程几何均值均>=1.02、配对轮次bootstrap95%下界>1、每shape中位比值>=.95；每点A/A中位偏离<=2%、baseline max/min<=1.10。无效或缺项不能接受，可按固定失败范围做一次重测并保留所有原始记录、来源映射；仍无效/无收益则baseline fallback。最终候选必须通过memcheck/racecheck及真实公共入口的全部6点与未映射输入拒绝检查。没有新收益时明确负结果，不制造optimized成功。

每组6点等权汇总；不报告全31点整体改进，其他5组没有本轮新性能证据。全部功能/负结果、诊断、真实输入IR/cubin/hash/完整options、资源、样本、区间、source/plan/runner身份写results/runs/mm_opt_r2_<id>。保留旧接口，用小型本轮runner复用未修改的common与首轮accuracy/paired-statistics工具；计时前完成编译/加载，GPU不并发工作，无锁频/清cache声称。precommit、独立最终close审查和Git归档完成后交付。KernelForge只提供workflow/diagnosis/evidence/review，本地MM适配未被注册为框架qualification。
