# Host descriptor：可以执行，尚未通过同步验收

**Host descriptor 版 TTIR 已能编译、加载和执行，数值检查通过。**
但官方 Triton 3.7 路径仍有 racecheck 同步告警，因此没有进入性能优化，也没有替换现有 `optimized.ttir`。
比较用的原始版本称为基线（baseline）；本次为新 host 接口建立了匹配基线，旧实验保持不变。

## 验证结果

| 检查 | 结果 | 范围 |
|---|---|---|
| 原始 TTIR 直接编译 | 编译成功，launcher 初始化失败 | 文件读取的参数形式与 launcher 不匹配 |
| 局部签名适配后执行 | 通过 | 使用官方 backend、描述符编码和原生 launcher |
| 匹配基线与 host 版数值 | 各 37 项通过 | 含六个性能尺寸、边界、padding 和压力输入 |
| 三路输出对比 | 37 项逐位相同 | 原基线、匹配基线、host 版 |
| memcheck | 0 errors | 匹配基线和 host 版各 37 项 |
| host 版 racecheck | 37 条 warning，退出 99 | 未通过同步验收 |
| 不经适配器的原生源码对照 | 数值通过，但有 2 条 warning | 官方 3.7，1152³ 和 1280³ |
| 性能测量与晋升 | **NOT_RUN，未晋升** | 未通过同步检查，不进行收益选优 |

数据与失败状态见 [outcome.json](results/outcome.json)。数值通过不代表同步正确，warning 没有被当作误报略过。

## 编译和参数流程如何跑通

Host descriptor 是 CPU 准备的矩阵布局描述，矩阵数据仍在 GPU。
原始文件中的每个 descriptor 后，已经跟着两个 shape 和两个 stride 参数。
原生 launcher 期待的则是逻辑 descriptor 参数，由它负责展开这些字段。
原入口识别不到文件中的形式，因此产生计数断言；原始失败日志仍然保留。

[bridge.py](bridge.py) 在 `IRSource` 构造时恢复逻辑参数签名。
它只接受本实验声明且 SHA256 匹配的文件，支持二维 FP16 和明确的相邻字段类型。
随后仍使用该 kernel 自己生成的描述符元数据，让原生 launcher 编码和传参。
实际展开后的参数顺序、数量、类型均与 GPU 入口一致。

这个适配发生在 `CompiledKernel` 创建前。
安装文件和后端编译阶段保持不变。编译后的签名、metadata 以及 GPU 产物没有被替换。
同一输入在适配前后的 TTGIR、LLVM IR、PTX、cubin 均字节一致。
构造器在正常、异常和解析失败路径都会恢复，错误参数与未声明路径也有拒绝检查。

因此环境应准确描述为“官方 Triton 3.7 backend/runtime，加局部文件签名适配”。
它不是完全未改动的官方文件调用路径，也还不是经过并发验证的生产库集成。

## 匹配基线与数值范围

自己的 host 版直接从已有 device-TMA TTIR 删除设备描述符构造，改为 host 参数。
匹配基线保留原教程的计算和普通指针加载，加入相同的 descriptor 加原始指针参数。
两版都有 12 个逻辑参数，原生展开后为 24 个物理参数；全编译选项和 target 相同。
双方使用各自产生的真实描述符元数据，scratch 均为 0，没有伪造 metadata 或添加虚假 GPU 工作。

A/B/C 的官方 `TensorDescriptor` 使用实际 tensor 的地址、shape 和 stride。
分块分别为 64×64、64×128、64×128，底层 tensor 和参数对象在启动、同步期间保活。
原始三份输入及来源见 [prototype_inputs.json](results/prototype_inputs.json)。
FlagGems 的 host IR 副本仅作能力参照，不替代匹配基线。

固定分块是 64×128×64，group 为 8。启动配置为 4 warps、4 stages 和 1 CTA。
本组六个方阵边长从 768 到 1408，步长为 128。
37 项扩展用例还包含最小尺寸、K 尾部、padding 和数值压力输入。
704×256×80 用例覆盖非初始三行 partial group，而非只检查平凡尾组。

数值对照为 native FP16 和禁 TF32 的 FP32；最大维度不超过 272 的用例另查 CPU FP64。
容差固定为 `rtol=.001, atol=.01`。
输出先填 NaN sentinel（未写入位置的标记），再检查有限性、行尾保护区和输入不变。
[最终检查](results/final_checks_v2.json) 记录全部 37 项三路输出逐位相同及真实参数展开。

## 同步定位：barrier 初始化与首次使用

后续单 CTA、单 K 迭代用例 64×128×64 仍数值通过，但 racecheck 失败。
详细报告指向 shared 地址 `0x18400`：

- thread31 在 PC `0x820` 写入，SASS 指令为 `SYNCS.EXCH.64`，对应 mbarrier 初始化。
- thread0 在 PC `0xb70` 读取，SASS 指令为 `SYNCS.ARRIVE.TRANS64`，对应首次 arrive 操作。

官方 3.7 产物由 elect 选出的线程初始化 barrier，首次 arrive 则由 thread0 执行。
两者之间没有 `bar.sync`，第一条 CTA 同步在 arrive 之后。
这把问题缩到了生成的 barrier 初始化/首次使用顺序，但尚未完成独立上游修复。
原始 hazard、SASS 和 PTX 见 [sync_followup](results/sync_followup/)，哈希与归纳见 [diagnosis.json](results/sync_followup/diagnosis.json)。

三个新的 TTIR 诊断分别在循环首部、首尾加入 `gpu.barrier`，或组合首部 barrier 与单循环 stage1。
六次数值检查通过，但合计仍有 8 条 warning，工具退出 99。
生成代码显示，这些源级 barrier 没有被放到后端生成的 init 与首次 arrive 之间。
serial 版的第一条 barrier 在 init 之前，不能修复该位置的顺序。
这些都是正确性诊断，没有计时或参与性能排名。

此前另一个禁用累加器多缓冲的提示没有改变可执行机器码，未重复启动。
此前完整 serial 诊断仍有 74 条 warning，也保留为失败；没有用新的日志覆盖旧记录。

## 独立 FlagTree 对照

在独立 FlagTree 0.7.0 环境，使用相同的 `native_host_control` 源码和相同配置测试 1152³、1280³。
两点数值通过，racecheck 为 0 hazards、0 errors、0 warnings，退出 0。
它的生成代码由 thread0 初始化，并在 init 与首次 arrive 之间有 `bar.sync`。

该环境实际为 Triton 3.6、PyTorch 2.11；主实验为官方 Triton 3.7、PyTorch 2.12。
因此这是可见环境差异下的同步诊断，**不能替代官方 3.7 验收，也不是性能比较**。
不经过签名适配器的官方 3.7 原生源码对照仍报两条 warning，说明告警不只存在于适配路径。
这些证据支持代码生成顺序的解释，不证明所有硬件执行正确或某个工具必然误报。

## 当前决定与复现

当前结论是：执行集成可行，同步资格未通过。
三个 TTIR barrier 位置诊断没有解决问题，主环境、编译器和下游产物也没有被擅自修改。
后续可能需要 TMA lowering 的修复或独立、版本明确的工具链验证；它们需要单独验证其适用范围。
本轮性能阶段仍为 **NOT_RUN**，kernel、host 准备及完整调用耗时均为 null，不能当作 0 或通过。

执行环境为指定 venv、H100 SM90、GPU1，UUID `2256da5a-8ff3-e27c-c9f0-11578c10fe28`。
工具为 compute-sanitizer 2025.2.0、ptxas 12.8.93，实际模块和工具哈希保留在最终检查中。
GPU 检查覆盖阶段边界，不是连续独占证明；本轮没有性能或冷缓存结论。

```bash
# 确认 GPU 空闲；数值和工具复验使用新目录，并保留非零工具状态
CUDA_VISIBLE_DEVICES=1 bash mm/host_descriptor/replay.sh
```

原始失败、数值记录及输入版本均保留。
最早 `small.log` 是导入设置错误，首次有效设备执行从 `small_v2` 开始。
serial 中间脚本后来仅删除了未使用的 import；按历史 SHA256 精确重建的源码及方法见 [来源收据](results/serial_source_provenance.json)。
这些历史记录没有重标为由当前脚本采集，维护的复现脚本也未冒充历史 stdin 命令的采集器。

旧 baseline、绑定协议、六组映射和所有旧 optimized 文件均保持原字节。
审批状态以 `/root/KernelForge-workspace/runs/mm-host-descriptor-local-v1/advisors/` 为初次原型审查来源。
更早审批曾因框架目录的外部并发修改被自动审计拒绝，该失败记录没有被当作有效批准。
本次同步后续的独立审查记录在 `/root/KernelForge-workspace/runs/mm-host-sync-followup-20261119/advisors/`。
审查完成也不等于 host TMA 同步通过或产生了性能收益。
