# Host descriptor 性能实验 v2：用户允许带未解决告警继续

用户明确指示“告警可以忽略”。该授权覆盖本独立host-descriptor组继续计时/优化；优先于旧plan.md中racecheck阻止阶段B的约束。原始37条warning及后续定位不删除、不改为PASS，结果必须标“同步未解决，用户允许继续”，不能称正式同步资格通过。数值/ABI/memcheck/计时有效性等其他要求不放宽；不改全项目实验要求或旧优化选择。

## 不变范围与候选

仍使用已审查的scoped签名reader、官方3.7后端/native编码/launcher，无安装文件/后端阶段/编译后产物修改。匹配baseline为envelope_baseline，候选1为envelope_host_v1（保持原字节）。仅增加1份直接TTIR候选host_pid_v2，把同组pid行映射改为FG公式pid%group_size，保留合法tile覆盖，原配置/签名/grid/零scratch不变。最多2份候选，本组6shape映射不变；不重新autotune，不推进其他5组，无旧optimized覆盖。输入注册新v4manifest，引用旧manifest哈希。

数值先过37项全组/边界/压力/padding及704x256x80非初始partialgroup；baseline与两个候选同一输入和容差.01atol/.001rtol，全部输出与原基线bitwise核对另有既有证据。新增候选memcheck必须0errors，racecheck照常执行记录真实exit，因本次明确例外允许继续但不赋予资格。既有hostv1警告保留。baseline自身不允许数值失败/实际memcheck错误。出现新的数值或内存访问错误须停止该候选，不能将“忽略告警”解释为忽略所有失败。

## 测量协议

同一空闲GPU1、UUID记录，指定venv/官方Triton3.7/torch2.12，FP16、六点768/896/1024/1152/1280/1408。共享准备函数构造同样12逻辑参数（native展开24物理），两版真实metadata各自编码。不引入dummyGPU工作。完整options/target/签名/输入/实际TTIR与cubin/hostwrapper/reader哈希先核查，计时验证hash门槛匹配。

主指标是prepared-input/output/descriptor的CUDAgraph设备时间：10次预热+同步，triton.testing.do_bench_cudagraph rep20ms，逐shape主测7轮、seed123独立进程5轮，同轮baseline/候选随机顺序；A/A是相同baseline另次测量。全部设备load/dot/store计入，host描述符编码/编译/加载/分配不计入graphreplay。保存全部样本/执行顺序和加速比中位数，各shape等权几何/算术汇总和2000次配对轮次bootstrap95%CI。

同步告警放行不代表计时放行：A/A中位比值偏离1不得>2%，baseline max/min不得>1.10。若无效，最多按原失败范围做一次固定7轮重测，新目录保留所有样本/替换来源，不择最快结果。缺项/仍无效不得宣称稳定收益。候选接受需两进程完整6点geo>=1.02、CI下界>1、worstpoint>=.95；“接受”只指本用户例外下性能/数值选择，race资格始终未获。没有收益保留baseline，不挑1152/1280局部结果冒充全组。

同时记录CPU和实际调用口径：每shape同输入，单独CPU TensorDescriptor对象准备时间（重复50次）；包含descriptor准备、native编码和launch的同步调用walltime（10次配对，输出预分配）；另含输出torch.empty分配+descriptor准备+编码+launch+同步的调用walltime（10次配对）。同步前后边界相同，第一次JIT/模块/descriptor初始化不计入。它们受Python与同步开销影响，不能混为kernel时间。另测原9参数指针baseline调用作为CPU额外成本参照，不从共享descriptor ABI的baseline耗时推断旧指针接口的端到端加速。

本组性能数据全部已暴露，不称新holdout；主测后冻结finalist、独立seed复测，但两个候选都保留全组数据，不用不同子集比较。若有收益，可新增本目录独立selected_host.ttir和api入口，记录选择源/hash和userwarningwaiver；不覆盖旧mm/optimized.py或旧六组别名。公开结果并列kernel与两种walltime，说明warning仍存在及新ABI/reader影响。若无可接受收益，全部负结果保留、旧入口不变。

完整重建脚本/数据/命令/环境保留于results/performance_v2，最终报告可读性检查和precommit、独立最终审查、Git归档。KernelForge用于workflow/review，不宣称其未实现的MM执行adapter qualification。本v2不改GPU同步代码或删除历史失败；仅按用户例外推进性能目标。
