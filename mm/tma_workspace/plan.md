# 设备TMA工作区分组 v1

目的：用户要求借鉴匹配配置的FGIR。固定workspace分组round2已完成且无可接受收益；按TTIR_EXPERIMENT_REQUIREMENTS第6节另建新工作区约定分组，比较原教程TTIR和已编译设备descriptor-TMA TTIR。不是修改旧约定或把旧结果重标为PASS。

仅目标64x128x64组6点（768/896/1024/1152/1280/1408）。原baseline.json、六组映射/编译配置/9参数签名/grid/common.launch和旧入口/别名均不变。新分组ir/baseline.ttir逐字节复制原baseline，ir/device_tma_v1.ttir逐字节复制round2编译诊断；手写TTIR、未改Pythonkernel或下游IR/binary。原r2诊断“未执行”的历史仍真实，新分组才执行此内容。

## 新的匹配运行约定

baseline和candidate都在相同进程context里注册相同的官方triton.set_allocator回调，退出恢复原allocator。回调只使用torch.empty分配uint8 CUDAbuffer（不做PyTorch compute），检查满足native传入alignment；actualstream为Torch当前CUDAstream。baseline metadata scratch0，candidate384bytes/CTA，均由Triton原生driver按真实metadata自行调用/分配；不手工传workspace指针，不改入口ABI，不替换binary/module/function/metadata。共享新的compile_native(path)函数，只有输入TTIR路径不同，options完全沿用4warps/4stages/1CTA/fusion，target官方3.7SM90。严格检查完整解析options与签名，差异仅真实资源和本组已声明scratch；baselinecubin须与原冻结baseline字节一致。

CUDAgraph计时包含candidate内构造TMA descriptor、fence、TMA访存、dot及store全部设备工作。runtime临时buffer分配在capture/host准备中，按本项目原kernel-only指标不计入时间；明确不能当作包含分配的端到端收益。不得预先在CPU编码descriptor来冒充device路径。

## 预算和验证

先验证baseline原生链一致性（复用未变来源、逐6点源码TTIR/后端产物/输出核对）以及新allocator两边同等生效/退出恢复。最多2功能候选：device_tma_v1，若首个可运行且有意义再试同一descriptor结构的FGpid公式device_tma_pid_v1。不更改配置、shape映射、数据或阈值。两版本非slot修复不得同名重写。

C0实际target/全options/签名/资源；C1每版本用round2已验证37用例（全部6性能点、压力、K/tile边界、padding和704x256x80非初始partialgroup），atol=.01 rtol=.001、nativeFP16/禁TF32FP32/小CPUFP64、sentinel/canary/input不变/有限性；失败不得计时。native _init_handles/launch不做手工替换，baseline和candidate共用common.launch。

初筛4点768/1024/1152/1280，3轮rep20ms随机对内配对；A/A为同baseline相同原生kernel的另一次计时。full6点7轮主测、seed123独立进程5轮复测，同输入/预分配输出/预热同步。每点原始样本、order、TTIR/cubin/runner/manifest/options/allocator源码与native driver hash全保存。本组已全暴露，不能把复测称新holdout。

每组6点等权配对中位几何均值+2000bootstrap条件CI；A/A偏差<=2%、baseline max/min<=1.10才可接受。两轮geo>=1.02、CI下界>1、最差shape>=.95，否则fallback/INCONCLUSIVE。最多对固定无效范围一次重测，不按加速比挑样本。最终native allocator入口的6点正确性与unsupported拒绝必须验证；memcheck/racecheck全部37用例或至少对应完整边界/压力集合，工具不支持TMA或报错须原样保留并阻止“最终验收通过”声明。

如果新分组有收益，提供独立tma_workspace入口和明确选择，不覆盖旧mm/optimized.py或optimized.ttir，也不声称旧固定工作区约定下收益。若失败则保留负结果。使用空闲GPU1，同卡不并行GPU任务，未锁频/无cacheflush声明，记录UUID/状态。完成precommit和独立close review、Git归档。此为本地适配，不授予KernelForgeCLIqualification。
