第二轮继续优化当前已验收TTIR，不重复宣称上一轮已有收益。原baseline、source_port和历史optimized别名保持不变；所有新候选、父版本副本、哈希和结果保存在各算子的round2/目录。

仅改TTIR内容，不生成优化Python kernel导出候选，不修改下游IR/机器码、不使用内联PTX。沿用各算子的原shape矩阵、dtype、数值合同、grid/ABI、kernel数、工作区和完整编译选项。新编译变换不保证只能通过TTIR表达，报告要明确公开Triton是否也可表达。

三路含义：baseline=原始冻结版本；parent=本轮开始时已验收版本（Attention D128为baseline）；candidate=新TTIR。增量指标为parent时间/candidate时间，累计指标为baseline时间/candidate时间，两者都须同轮测量。

每个候选先C0编译及完整原数值/压力矩阵，后3轮10ms初筛。按全矩阵（Attention可沿用完整D/causal组）增量几何平均选单一候选冻结，再7轮20ms主测和独立seed123的5轮20ms复测；每轮随机版本顺序，保存全部样本，输入seed/hash和产物hash。所有路径同等预热，A/A点中位比必须在[0.97,1.03]；超限轮保留并标无效。两正式运行增量geo>=1.02、最差点>=0.95，且原合同数值与sanitizer均通过方可接受，否则保留parent。负结果可完成主/复测供归因。不能对复测逐shape挑最好者。

GPU0/2/4/6/7分别为INT4/MoE/Attention/Norm/TopK独占；环境为/root/venvs/ir-ops-dev/bin/python官方Triton3.7.0。重新验证原baseline源码与TTIR文件入口LLVM/PTX/cubin一致。静态资源与指令变化是C级事实，单次或筛选性能B级，正式完整/独立复测性能A级；不混同。

用户追加要求：每个算子扩充到约30个shape。本轮正式矩阵改为INT4/MoE/Norm/TopK各30点、Attention32点，共152点；原小矩阵仅作为pilot，已命名main的旧小矩阵不构成最终扩展矩阵结果。各expanded manifest/协议固定后才运行最终筛选和主/复测。保留原所有shape，新增点配置策略必须明示，不把固定family说成autotune。

计时口径补充：10ms/20ms是triton.testing.do_bench_cudagraph的rep参数，不是每个外层配对轮次的实际总GPU持续时间。固定Triton3.7内部有10次graph replay重试，默认返回其均值；本项目保存每次函数调用返回值组成的全部外层配对样本，不声称保存内部10次CUDA event明细。预热与计时分开，Attention后续如更换预热实现须显式版本化、对三路一致，并重跑完整选择/确认矩阵。
