# INT4 解包、groupwise 反量化与 GEMM：首轮结果

**最终选择 `scalar_unsigned_v1`：冻结 7 点矩阵的配对加速比几何均值为
1.094808×，独立进程复测 1.095816×；最差点分别为 1.063898× / 1.068932×。**
这是本矩阵的 A 级性能证据，所有候选的数值、资源及原始样本均保留；不外推为
其他 dtype、量化格式、硬件或形状的性能，也不是对 A/B/C 信息消融的证据。

## 基线与唯一变量

来源为 FlagGems commit `ef8aea78fed8321c3e02a15bf31e847cd0579e4f` 的
`_weight_int4pack_mm_with_scales_and_zeros.py`。原始快照在
[upstream/source.py.snapshot](upstream/source.py.snapshot)，冻结提取版
[tutorial.py](tutorial.py) 保留设备函数的完整计算语句，仅去掉 `libentry`、主机包装与
FlagGems 依赖，将 `tle.program_id` 绑定到 upstream `tl.program_id`。
源码哈希、提取说明与每点配置在 [baseline.json](baseline.json)。

基线已经融合 INT4 nibble 解包、groupwise 反量化与 Tensor Core dot。
没有以显存完整反量化后再 GEMM 作为唯一基线。按预定的四组 tile 配置逐点筛选并冻结；
这是有限配置调优，不能称全面调优或最强库基线。FP16 输入/输出、FP32 反量化中间计算
及累积，group=128，连续 packed `[N,K/2]` uint8，低 nibble 是偶 K。
具体支持域与数值容差见 [optimization_plan.md](optimization_plan.md)。

优化只修改导出的 TTIR；输入、ABI、grid、完整后端 options、launch 和计时完全匹配。
冻结基线/候选都使用 `/root/venvs/ir-ops-dev/bin/python`、官方 Triton 3.7.0 和其
原生 `triton.compile(path)`→CompiledKernel→CUDA 链。源码与文件入口的 LLVM/PTX/cubin
在全部 7 点上字节一致，源码与基线输出逐位一致。最终 alias 与候选同 options 且输出
逐位一致，见 [最终 pipeline 核验](results/pipeline_final.json)。

设备是独占物理 GPU 0，H100 80GB HBM3，UUID
`GPU-cc248ebf-79d0-42d7-da65-99d1ef6f4ed1`；同卡无并行实验。
未锁时钟，未控制冷缓存。环境、驱动、工具链哈希及实际编译 metadata 随结果记录。

## 直接 TTIR 变换

[make_candidates.py](make_candidates.py) 以冻结 baseline 为唯一输入，确定性修改 SSA：

| 候选 | 结构变化 | 7 点初筛几何均值 | 状态 |
|---|---|---:|---|
| `shared_scale_v1` | 删除 high 分支 group/scale/zero 加载，直接使用 low 分支值 | 1.076263× | 完整主测 1.076005×，未独立复测，B 级 |
| `group_scalar_v1` | 在共享基础上用标量 `k0/group` 再广播，去掉恒真的 K mask | 1.082623× | 仅初筛，B 级 |
| `unsigned_v1` | 对有非负证明的索引除余改用 unsigned | 1.033726× | 仅初筛，B 级 |
| `scalar_unsigned_v1` | 合并 group 标量化、共享、unsigned | 1.096218× | 最终选择，A 级 |
| `cast_after_v1` | FP32 join/trans/reshape 后再转 FP16 | 0.999993× | 无收益；负结果保留，B 级 |

共享依赖 group 为偶数：同 byte 的偶/奇 K 位置属于同一 group；K 精确整除 128，
两者的有效 K mask 相同。标量化进一步依赖 BK 为 64 或 128，且每次循环起点与
group/BK 对齐，因此一个 tile 内 group 恒定。此事实不成立的 group、K 尾部或 BK
不在支持域内。unsigned 只用于非负 program id、正尺寸及非负 K 索引。
没有改反量化减法、乘法、FP16 舍入点或 dot 累积次序。

cast_after 假设没有在当前编译链中产生收益。7/7 shape 的 ELF `.text` 哈希与 baseline
相同，寄存器/shared 也相同；完整 PTX/cubin 哈希因位置/调试信息不同而不一致。
因此不能称“移动 cast 发现后端缺陷”，也不能把源码注释中的可能风险视为本轮已证实。
机器指令证据在 [analysis.json](results/analysis.json)。

## 数值与内存检查

- 第一进程：7 shape × random/extreme_nibble/zero_scale/zero/constant ×
  baseline 与 5 候选，共 210 次版本检查全部通过。
- 独立 seed=123：baseline 与最终候选共 70 次检查全部通过。
- 对照 FP32 ATen matmul：反量化权重先按合同舍入至 FP16，使用
  `atol=0.035, rtol=0.003`；首进程最大绝对误差 0.03125。
- 两个小 shape 额外对照 CPU FP64 matmul。所有候选还要求与基线逐位相同，
  检查输出有限、末尾 64 元素 canary 未改、全部输入未修改。
- 最终 baseline/候选在完整 7×5 分布上分别运行 memcheck/racecheck，
  各 70 次版本检查均通过：memcheck **0 errors**；racecheck
  **0 hazards，0 errors，0 warnings**。

原始证据：[数值](results/accuracy.json)、[独立数值](results/accuracy_repeat.json)、
[memcheck](results/memcheck.log)、[racecheck](results/racecheck.log)。
这些是明确有限分布与形状的检查，不是对支持域内所有输入的证明。

## 完整主测与独立复测

主指标为预分配输出的单 fused kernel CUDA graph replay，排除数据构造、参考、分配、
编译、加载和 Python launch。主测 7 轮/点，独立进程 seed=123 复测 5 轮/点；
每个 sample `rep=20ms`，随机配对顺序，采用每点配对加速比中位数。
初筛为 3 轮、10ms。每轮原始时间和顺序均保留，未只取最快样本。

| M,N,K | 冻结 BM,BN,BK | baseline μs | 最终候选 μs | 主测加速比 | 复测加速比 |
|---|---|---:|---:|---:|---:|
| 16,4096,4096 | 32,64,128 | 74.319 | 64.675 | 1.149121× | 1.147382× |
| 128,11008,4096 | 32,64,64 | 326.716 | 298.316 | 1.095187× | 1.095235× |
| 1,256,128 | 16,64,64 | 3.779 | 3.546 | 1.066774× | 1.068932× |
| 32,1024,1024 | 16,64,64 | 19.992 | 18.337 | 1.090165× | 1.089031× |
| 64,2048,4096 | 16,64,64 | 74.340 | 66.972 | 1.110139× | 1.110531× |
| 128,4096,4096 | 32,64,64 | 117.241 | 110.201 | 1.063898× | 1.071649× |
| 17,80,256 | 16,64,64 | 6.052 | 5.549 | 1.090610× | 1.089872× |

时间列为各版本样本中位数，不能用四舍五入时间之比替代配对加速比。
所有配置均为 4 warps、3 stages、1 CTA。全矩阵几何均值主/复测
**1.094808× / 1.095816×**，算术均值 **1.095128× / 1.096090×**。
A/A 主/复测几何均值 **1.000034× / 1.000069×**，各点接近 1。

固定七形状、仅重采样各形状配对样本的 bootstrap 95% 区间为
主测 `[1.094468,1.095178]`，复测 `[1.095595,1.096029]`。
这仅描述本轮 graph 样本内波动，不包含形状分布、跨时钟或跨日期不确定性；
两独立进程本身的差异更能反映这项局限。

选择依据是主测全矩阵几何均值，要求 ≥1.02 且最差点 ≥0.95；
[预复测选择](results/selection_pre_repeat.json) 在复测前冻结。
复测与 sanitizer 仅负责验收，没有重新按复测收益选候选。
没有逐点拼接最有利候选；例如 `(128,4096,4096)` 的 shared_scale 主测为 1.133123×，
确实快于最终全局候选的 1.063898×，这一局部不足保留。

配对原始数据：[初筛](results/screen.json)、[主测](results/main.json)、
[复测](results/repeat.json)，重算脚本 [analyze.py](analyze.py)。

## 资源证据与归因边界

| 分组 | baseline registers | 最终 registers | shared bytes | spill |
|---|---:|---:|---:|---:|
| BM32,BN64,BK128 | 250 | 210 | 40960→40960 | 0→0 |
| BM32,BN64,BK64 | 168 | 138 | 20480→20480 | 0→0 |
| BM16,BN64,BK64 | 140 | 106 | 16384→16384 | 0→0 |

TTIR 可见的重复 group/scale/zero 数据流在冻结基线中确实存在，直接删去后最终机器码、
寄存器与性能均变化。组合还包含 group 标量化、K mask 简化与 unsigned 索引，因此
不能将 1.095× 全部归于 scale 加载、寄存器或某个单项。没有 NCU 指标或逐项独立复测，
资源下降是 C 级静态事实；组合性能才是本矩阵 A 级证据。

文章讨论的偶/奇两路 dot 已存在于另一份 FlagGems 实现，本轮未重新实现或将其称为
新优化；未开展 INT4 独立反量化吞吐实验、residual epilogue、其他 packing、BF16/FP8
或其他硬件实验。当前改写可用 Triton 源码回写，但本轮没有 A/B/C 盲测，不能证明
“直接 TTIR 独占收益”或“看到 TTIR 比只看源码更好”。

最终映射和文件哈希见 [selection_final.json](results/selection_final.json)；
每组 `optimized.ttir` 是 `scalar_unsigned_v1.ttir` 字节副本。未修改 FlagGems 源码或注册。
搭建过程失败记录在 [setup_failures.md](results/setup_failures.md)。

新增的实际源码回写比较见 [source_port/report.md](source_port/report.md)，其中同轮比较原基线、TTIR优化版与正常Triton源码编译版。
