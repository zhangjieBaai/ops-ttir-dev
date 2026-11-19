# Host descriptor：执行集成可行，racecheck阻止进入优化阶段

已将host-descriptor方案实际编译、加载并运行。**可以把自己的TTIR改为host descriptor，并在官方Triton3.7.0原生backend/runtime上执行，但需要显式文件签名reader适配和新的host参数契约；不是只换旧optimized.ttir文件就能运行。** 全部旧baseline、配置映射和旧optimized选择保持原字节。

数值和memcheck流程已闭环，**racecheck仍有37条warning，正式性能优化门槛未通过**。因此本轮没有benchmark加速比、没有新的最优版本，也没有将该方向宣布为已验证可用。阶段A可行性与阶段B资格要分开；用户授权的条件“跑通后再真的优化”没有被数值PASS代替。

| 阶段/对象 | 执行结果 | 能支持的结论 |
|---|---|---|
| 原始host TTIR直接file入口 | GPU编译成功，launcher初始化断言失败 | 官方3.7原file-source签名读取与native descriptor展开不匹配 |
| 适配后FG原始host IR | 1152/1280数值通过 | native host编码/启动能力可用，非正式baseline |
| 同签名envelope baseline / host版本 | 各37项数值通过、scratch0 | host入口及匹配baseline可以跑通，不需要设备descriptor scratch |
| memcheck | 两版本各37项数值通过，工具0errors | 该覆盖范围没有memcheck错误 |
| racecheck | 数值通过，但37warning报告，退出99 | 同步资格失败，不能晋升或启动正式性能选优 |
| 不经reader的native源码对照 | 1152/1280数值通过，2warning报告、退出99 | 告警不只存在于reader适配路径；原因仍未确定 |
| 禁用acc多缓冲诊断 | 同原始可执行section，未启动 | 该提示本例没有改变同步机器码 |
| 单循环stage1诊断 | 数值通过，74warning报告、退出99 | 减少流水级没有清除告警；不是4stages性能候选 |
| 最终三路一致性 | 原baseline/envelope baseline/host 37项逐位相同 | 匹配接口未靠添加虚假GPU工作拖慢baseline，数值结果一致 |

## 签名适配具体做了什么

原始TTIR文件入口从IRSource得到物理MLIR参数：每个 `!tt.tensordesc<tensor<...xf16>>` 后已有2个i32 shape和2个i64 stride。nativeCudaLauncher却检查Python逻辑签名`tensordesc<fp16[...]>`并自己展开为CUDA tensor map及shape/stride，原形式不被识别，导致计数断言失败。

[bridge.py](bridge.py)调用原IRSource构造器解析文件，然后**在CompiledKernel构造前**仅对受支持、声明且hash匹配的文件恢复逻辑signature，将上述5个物理参数折叠为1个逻辑descriptor参数。标量/指针顺序不变；nativeCudaLauncher按**该kernel自己生成的tensordesc_meta**编码tensor map并展开。raw/adapted对同一输入的TTGIR、LLVM IR、PTX和cubin均逐字节一致，模块、源码hash、make_ir及backend阶段不变。没有在编译后改ck.src.signature、metadata、cubin、module或function，也没有阶段override/hook。

构造器只在受锁context中临时包装并finally恢复；未修改安装文件。校验涵盖正常/抛异常/解析失败的恢复、未声明路径拒绝、缺字段/错dtype/错相邻字段拒绝；native展开的类型顺序和数量与真实物理prototype完全相同。固定版本且单线程实验范围，未验证并发库级部署，不把这个局部ABI桥说成官方原file入口天然支持。

三份初始输入的来源与SHA256在 [prototype_inputs.json](results/prototype_inputs.json)：FGhostReference是只读实际产物副本；自己的envelope_host_v1直接从已有device-TMA TTIR移除device描述符创建，改为hostdesc入口；envelope_baseline保留原教程计算和指针访存，仅加入相同hostdesc参数。两者真正拥有相同逻辑及物理参数envelope，unused descriptor仍由native产生真实metadata并编码，没复制candidate metadata给baseline或插入假GPU工作。host端构造A/B/C官方TensorDescriptor，形状/stride由实际tensor取得，block_shape64x64/64x128/64x128；底层tensor与参数对象在启动及同步期间保活。scratch均为0，未注册allocator。

## 数值和同步证据

本组为BM64/BN128/BK64、group8、warps4、stages4、CTAs1，六点768/896/1024/1152/1280/1408；扩展37项含最小输入、K/tile尾部、行尾padding、zero/constant/alternating/large/small分布及704×256×80非初始三行partial-group。nativeFP16与禁TF32FP32参考，max维度<=272另用CPUFP64；rtol=.001、atol=.01，输出NaN sentinel、有限性、padding canary和输入不变检查。最终 [final_checks_v2.json](results/final_checks_v2.json) 对37项比较原baseline、envelope baseline、host版本，三路输出逐位相同，并核查完整解析options/target、相同实际ABI、zero scratch及原始安装RECORD。

[full_check](results/full_check/summary.json)、[memcheck_check](results/memcheck_check/summary.json)、[racecheck_check](results/racecheck_check/summary.json)的数值各为baseline37+host37通过。但工具状态独立：[memcheck.log](results/memcheck.log)0errors；[racecheck.log](results/racecheck.log)37条显示的hazard报告（单报告可能聚合多个底层hazard），0errors/37warnings，进程退出99。数值正确不清除警告，不把warning改标PASS。

为排除“只有签名适配才触发”的解释，新增 [source_controls.py](source_controls.py) native源码对照，不经过reader适配，直接`@triton.jit`执行同样host descriptor load/dot/store；1152/1280数值通过，native_source_racecheck.log仍有2条warning、退出99。它只是诊断对照，不是重新修改教程kernel后伪装直接TTIR候选；不能据此证明是编译器bug、工具误报或实际硬件race的哪一种，原因保持未知。

进一步的两个**同步诊断**由[make_sync_probes.py](make_sync_probes.py)确定性生成，使用独立v2输入registry，原registry/候选不覆盖。`tt.disallow_acc_multi_buffer`产生相同可执行机器码，没再启动；`tt.num_stages=1`只覆盖该循环，顶层编译选项仍4，但属于显式不同循环调度诊断、不参与排名，完整数值通过却仍74条warning，未修复资格。不是反复调参直到工具不报，也没有任何下游IR或binary编辑。

## 是否作为优化方向

**集成层面值得继续研究，但本次不能进入正式收益优化或替换optimized。** 优先未决问题是TMA/WGMMA路径racecheck告警的原因和清除条件；需原始hazard地址/生成同步的独立证据或明确工具支持说明，不能直接忽略。当前已经排除“仅因reader适配发生”和“降低循环stage必然能清除”的简单解释。

由于阶段A的必需同步检查失败，阶段B完全未运行：无kernel计时、host descriptor准备时间、包含准备的调用耗时或性能接受；这些状态均是NOT_RUN，不是0或通过。本轮不承诺host版会重现FG性能。若后续同步门槛通过，才按照[plan.md](plan.md)对统一参数契约的baseline/候选完整6点配对计时，另测host准备和调用成本，保持原旧实验独立。

## 环境、结果和复现

执行为指定venv官方Triton3.7.0/backend/runtime加本地signature reader适配，FP16/H100 SM90/GPU1，UUID `2256da5a-8ff3-e27c-c9f0-11578c10fe28`。compute-sanitizer2025.2.0、ptxas12.8.93，实际库及工具源码hash在final_checks_v2.json。FlagTree不参与本轮编译。GPU隔离检查针对阶段边界，不是连续独占保证；没有锁频或冷缓存承诺，且本轮未进行性能测量。

[README.md](README.md)列出脚本职责；[replay.sh](replay.sh)在新目录重新执行数值/工具/最终ABI检查并保留nonzero exit status，不自动进入性能。初始导入错误small.log保留，首次有效执行从small_v2开始；旧bridge/probe版本源文件快照保留，旧结果不重标为当前runner采集。最终原始registry、输入/产物hash和probe源码版本在各JSON核对。

全部原型和失败日志保留；独立KernelForge计划审查已通过，最终close审查记录位于 `/root/KernelForge-workspace/runs/mm-host-descriptor-feasibility-v1/advisors/`。该框架未注册MM执行adapter，不宣称CLI qualification；本结论也不授予hostTMA同步正确性或性能收益。

serial诊断记录的中间probe源码后来只清理了一条未使用的`import json`。现已按该记录的SHA256精确重建源码，见 [重建收据](results/serial_source_provenance.json)；历史serial JSON保持原字节，没有改成由当前probe采集，也没有重复执行GPU实验。
