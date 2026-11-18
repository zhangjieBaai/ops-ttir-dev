# MM TTIR 六组优化首轮计划 v1

用户已授权开始优化。此计划扩展先前仅基线阶段；保留baseline.json及其绑定的所有文件原字节，不重新autotune，不改变31点shape→6组映射。遵循TTIR_EXPERIMENT_REQUIREMENTS.md；KernelForge用作计划、诊断、证据和独立review流程，执行仍用本项目已验证的原生Triton TTIR适配器。KernelForge当前未注册matmul执行适配，不能将本地结果宣称为其CLI qualification。用户明确要求的CUDA graph kernel指标为主指标，优先于通用skill的不同默认指标。

## 不变的约束

FP16、FP32累加、SM90、指定venv/Triton3.7.0，每组原tile/GROUP/warps/stages/CTAs/完整options、ABI、grid、分配与公共launcher不变。baseline/候选都通过未修改的common.compile_ir(family,name)原生文件入口编译与common.launch启动。只直接改TTIR文本，不改教程Python或任何下游IR/binary。输入范围、对齐、无别名、容差沿用冻结契约；额外正确性用例可扩充，但两版相同。

## 执行与预算

1. 在空闲H100 GPU6上隔离执行，先重新跑31点baseline pipeline核查，保存环境、工具hash、完整options/产物和UUID。设备每个阶段检查外来进程；不锁定时钟、不主动清缓存，不能保证连续独占。
2. 诊断先检查TTIR/TTGIR/PTX、编译资源，必要时只为明确问题运行NCU；不以profiler时间作为收益。
3. 每组最多6个独立初始候选：canonical（仅去loc路径控制）、unsigned（有非负证明的索引div/rem转unsigned）、wrap_select（越界tile加载索引改为in-bounds选择合法0索引）、k_aligned（K整除BK时无尾mask的循环、否则原循环）、cache_cg（加载缓存提示）、unroll2（TTIR循环展开提示）。候选编译或正确性失败保留，不改同名文件。允许从有证据的单项组合最多2个/组，总预算≤48个文件；无收益的方向停止，不无限搜索。
4. 优先完成每组baseline与候选C0（解析/完整options/ABI）、C1（该组全部31性能点中的所属点+原family_cases21项+新增需要的分支边界）。每次重置NaN输出、检查canary/输入不修改/有限性，参考native FP16、禁TF32 FP32和小尺寸CPU FP64；atol=1e-2,rtol=1e-3。任何失败不计时。候选记录输入TTIR/cubin/hash/metadata和失败原因。
5. 初筛固定12点：256,384,640,768,1024,1408,1536,2048,2816,4096,2176,2304。每点3轮、rep=20ms，同输入和预分配输出、预热10次，同轮随机baseline/候选顺序，保存原始配对样本；canonical与baseline同时作为A/A路径/噪声控制。只在同组相同集合比较候选，按配对中位数几何均值选每组一个finalist；初筛是B级，不是最终收益。
6. 冻结finalist后完整31点各7轮rep=20ms；剩余19点首次候选性能暴露作为转移/回退检查。独立进程seed=123各5轮复测，A/A控制仍保留。每组按所属shape等权汇总，整体31点等权；不按最好点切换候选。256和2176组只有1点，无法构成独立shape holdout；所有已暴露点如再指导修改均注明tuning曝光，不能把换seed称新holdout。
7. 逐shape配对轮次bootstrap（种子固定，2000次）给各组及整体95%区间，只覆盖保存样本波动；报告跨进程变化、所有回退及A/A spread。最终接受某组候选：主/复测几何均值均≥1.02且95%区间下界>1.0，各shape主/复测中位加速比≥0.95；A/A偏离超过2%或baseline配对轮次漂移超过10%的点标记疑似无效，先查干扰，再用新结果目录重测受影响范围。阈值为本轮caller选择。未通过则保留该组baseline，明确“尝试但无可接受收益”。
8. 对每个最终配置（包括fallback baseline）执行memcheck/racecheck，含每组5个原padding边界，以及新增分支两侧。失败阻止接受。每组optimized.ttir必须对应selection.json哈希；fallback明确记录选择baseline而非优化成功。
9. 原始证据存mm/results/runs/mm_opt_r1_<id>/，KernelForge审查记录存既有KernelForge-workspace；不修改框架源码、不引入FlagTree/llc路线。最终运行pre-commit，独立close review并解决重大证据缺口，Git归档，给出逐6组结果、整体及全部限制。

## 可证伪机制假设

- 动态整数除余对小矩阵的索引开销可能可减；wrap_select保留有效输出对应的索引，越界行/列使用合法0地址且仍mask store，不能越界读取或影响有效dot结果。必须跨tile尾部和padding验证。
- K整除分支可能消除每次load的mask，改善lowering；两分支保留相同tensor形状、grid和累加顺序，非整除继续原循环。分支及资源上限可能抵消收益。
- cg/unroll可能改变访存/流水线/寄存器，也可能被后端消除或退步；必须检查实际产物并保留负结果，不把资源差异当独立因果证据。
- 如果三次有效局部读数平均新增收益≤5%或编译重复失败，在安全边界反思：换机制或停止该方向，记录排除项，不以更多同类变化掩盖停滞。
