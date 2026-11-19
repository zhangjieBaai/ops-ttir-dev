# var_mean TTIR 优化实验（修正后的第二轮）

30 个 shape 的主测几何平均加速
**1.2928×**，配对轮次条件 bootstrap 95% 区间 [1.2916, 1.2941]；
独立进程复测 **1.2947×**，区间 [1.2937, 1.2966]。
这是预分配输出的 CUDA graph kernel 时间，不含 Python、分配、编译或加载。

本报告仅使用恢复 int64 行索引后的 r2 数据，契约为
`var_mean-row-f32-v3`，设备为物理 GPU5（H100，UUID `GPU-8b3add97-9c35-553e-695d-14cac5a9ea51`）。
30 个 shape 曾在已废弃的 int32 提取版本中暴露，属于确认和回归矩阵，不能称为全新 holdout。
旧轮数据完整保存在 [setup_attempts/i32_pid](setup_attempts/i32_pid/README.md)，不计入下列加速比。

## 冻结范围与基线

输入是连续 FP32 二维张量，对最后一维按行计算 mean 和中心化方差；correction 为 0 或 1。性能点为 M∈{1,17,128,1024,4096} 与
N∈{128,512,1040,4096,8192,16384} 的笛卡尔积，共30点，等权；性能用 correction=1。支持契约要求 N 是16的倍数且16≤N≤16384、1≤M≤4096，无别名。
全张量两阶段路径、FP16/BF16、非连续输入及 API 端到端性能不在本轮范围。

原始实现快照来自 `/root/FlagGems/src/flag_gems/ops/var_mean.py`。只去除主机缓存装饰器，将 `ext.program_id(0)` 展开为原 helper 的
`tl.program_id(0).to(tl.int64)`；[AST检查](results/runs/r2/extraction.json)证明其余 kernel 计算体一致。源码 SHA256 为
`e0346b973e4d63994c90a9699bd62a3cd2a9d88c8522df10502a5e5665fec820`。

原实现此路径没有 autotune。baseline 固定 BLOCK_N=1024，4 warps、3 stages、1 CTA，开启 FP fusion；两版使用完全相同的后端 options、签名、每行一个 CTA
的 grid、CompiledKernel launcher 和原生 Triton 文件入口。环境为官方 Triton3.7.0、独立 Python3.12.3 venv，PyTorch版本见
[pipeline](results/runs/r2/final_gpu5/pipeline.json)。统一设置 TRITON_DISABLE_LINE_INFO=1，避免文件解析 unknown loc 的调试行号差异；
最终30点源码/文件 baseline 的 LLVM、PTX、完整 cubin 逐字节一致，归一化 TTGIR 一致，两项输出逐位一致。

## 选中的 TTIR 改写

候选 `r2_short_hybrid_aligned_v1` 已按相同 SHA256 复制为 [optimized.ttir](ir/optimized.ttir)，哈希 `812e8d9eacc3a0bb33ea24efa729d48ac9b11b828d0c6c9ff31c7fb4a94efa9e`。baseline 哈希
`15298caf68470c10aa3d4e7dda94c5322d4c7c4091778622a6ed01faed06ef09`。

- N≤128/512/1024 时，分别采用128/512/1024宽的单 tile：一次加载，先归约 mean，再计算中心化平方和。
- N>1024 时保留两遍中心化算法，将动态循环 tile 从1024改为256。
- 在 TTIR 为 N 增加已由契约和 launcher 检查的16整除属性，使 int64 行地址计算可以利用该前提。

以上改动只发生在 TTIR；没有改 kernel Python 计算后重导出，没有编辑 TTGIR/LLVM/PTX/cubin，也没有改变两版的启动配置。baseline 与最终版均为
32/32 registers per thread、
0/0 spill、
16/16 bytes shared。

PTX 中可见最终版的向量化 load，而基线有多条标量 load。这是静态产物证据（C级），不能把全部性能收益归因于单项指令变化；没有 profiler 或完整最终矩阵单项消融。短行展开、循环宽度、整数前提共同构成最终组合。

## 候选与负结果

下表为同进程12点、3轮、rep=10ms初筛（B级），不是最终完整矩阵加速比。所有候选先通过 CPU FP64 数值检查。

| 候选 | 几何平均 | 最差点 | registers/thread | 处理 |
|---|---:|---:|---:|---|
| r2_short_hybrid_aligned_v1 | 1.3111× | 1.0071× | 32 | 进入18点确认 |
| r2_short_hybrid_v1 | 1.2253× | 0.9538× | 32 | 保留对照 |
| r2_single_dispatch_v1 | 1.4203× | 0.4829× | 213 | 严重退步，弃用 |
| r2_single_dispatch_v2 | 1.3667× | 0.4781× | 255 | 严重退步，弃用 |
| r2_tile256_aligned_v1 | 1.2869× | 1.0101× | 32 | 保留对照 |
| r2_tile256_v2 | 1.1892× | 0.9671× | 32 | 保留对照 |
| r2_tile4096_v2 | 0.9217× | 0.3341× | 96 | 严重退步，弃用 |
| r2_tile8192_v2 | 0.7225× | 0.1556× | 168 | 严重退步，弃用 |

其余18点确认几何平均
1.2833×、最差 1.0120×；确认后没有再修改候选。大 tile 单次加载版本的最坏退步说明，减少加载或提高局部吞吐并不足以保证所有行数上的
收益；高寄存器资源是观测到的静态线索，未做因果隔离。

## 最终配对计时

主测30点×7轮，seed42；独立进程复测30点×5轮，seed123；两者 rep=20ms，每轮随机 baseline/候选先后顺序，预热后 graph replay，同一物理 GPU5。
每点取配对加速比中位数再等权汇总，保存全部轮次、顺序、IR/cubin哈希及负载时钟/功耗/进程采样。两轮没有监测到其他 GPU5 进程。协作锁只协调使用同一锁的任务，采样不能排除所有瞬时外部干扰；没有主动锁频或主动清缓存，
不能称为冷缓存或端到端收益。区间仅是保存配对轮次下的条件不确定性，不覆盖选择偏差和所有系统误差。

主测算术平均
1.3152×；最小中位数 1.0074×，shape=(4096, 16384)；
最大 1.8101×，shape=(1, 16384)。独立复测最差 1.0078×。

| M | N | baseline µs | optimized µs | 主测配对加速 | 独立复测 |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 1.7326 | 1.5387 | 1.1263× | 1.1259× |
| 1 | 512 | 1.7233 | 1.5523 | 1.1102× | 1.1025× |
| 1 | 1040 | 2.2406 | 1.9339 | 1.1586× | 1.1680× |
| 1 | 4096 | 3.7333 | 2.5947 | 1.4428× | 1.4468× |
| 1 | 8192 | 6.0754 | 3.7159 | 1.6363× | 1.6400× |
| 1 | 16384 | 10.7940 | 5.9596 | 1.8101× | 1.8133× |
| 17 | 128 | 1.7599 | 1.5857 | 1.1196× | 1.1121× |
| 17 | 512 | 1.7731 | 1.6206 | 1.0919× | 1.0933× |
| 17 | 1040 | 2.3545 | 2.0431 | 1.1528× | 1.1502× |
| 17 | 4096 | 3.7785 | 2.6910 | 1.4088× | 1.4098× |
| 17 | 8192 | 6.1288 | 3.8134 | 1.6112× | 1.6146× |
| 17 | 16384 | 10.8617 | 6.0645 | 1.7929× | 1.7928× |
| 128 | 128 | 1.9488 | 1.7663 | 1.1051× | 1.1068× |
| 128 | 512 | 1.9727 | 1.8126 | 1.0826× | 1.0862× |
| 128 | 1040 | 2.5398 | 2.2515 | 1.1324× | 1.1335× |
| 128 | 4096 | 4.0115 | 2.9447 | 1.3629× | 1.3667× |
| 128 | 8192 | 6.3803 | 4.0580 | 1.5750× | 1.5776× |
| 128 | 16384 | 11.2373 | 6.2925 | 1.7857× | 1.7896× |
| 1024 | 128 | 2.8002 | 2.1445 | 1.3066× | 1.2930× |
| 1024 | 512 | 2.8575 | 2.2620 | 1.2616× | 1.2631× |
| 1024 | 1040 | 3.8313 | 3.1324 | 1.2197× | 1.2117× |
| 1024 | 4096 | 6.9908 | 6.2565 | 1.1152× | 1.1534× |
| 1024 | 8192 | 12.2300 | 10.6649 | 1.1465× | 1.1570× |
| 1024 | 16384 | 51.2797 | 47.3223 | 1.0837× | 1.0891× |
| 4096 | 128 | 6.6711 | 4.0854 | 1.6331× | 1.6284× |
| 4096 | 512 | 6.6952 | 4.2398 | 1.5787× | 1.5749× |
| 4096 | 1040 | 9.9099 | 6.7652 | 1.4648× | 1.4682× |
| 4096 | 4096 | 31.3670 | 27.9479 | 1.1223× | 1.1197× |
| 4096 | 8192 | 87.3737 | 86.4890 | 1.0102× | 1.0124× |
| 4096 | 16384 | 173.1538 | 171.8877 | 1.0074× | 1.0078× |

A/A控制（3个代表shape）几何比
0.999994，重复进程 0.999836；
已知双次启动的 baseline/双次序列几何比
0.499892，复测 0.500264，接近预期0.5。
两次控制进程对应 baseline 耗时比（repeat/main）为 `{'(1, 128)': 1.0000656952340463, '(128, 16384)': 1.0020960195091029, '(4096, 16384)': 1.000248502489594}`，仅描述会话内这3点的跨进程漂移，不是全域噪声界。

## 数值、内存与同步检查

最终 baseline/optimized 每个 seed 各564项数值比较，两进程各1128条记录；包含30性能shape、18边界N、M=1/17、correction0/1，
以及normal、constant、offset10000、lowvar、zero、alternating、first-element outlier。每次 launch 前重置输出NaN，检查两端各16元素 canary。
CPU FP64 `torch.var_mean` 是主参考，variance rtol=2e-4/atol=2e-6，mean rtol=2e-5/atol=2e-5。以下是测试集绝对误差，判断使用上述相对+绝对容差，
不能把它们解释为全部输入误差上界。

| 输出误差对CPU FP64 | baseline | optimized |
|---|---:|---:|
| variance最大绝对误差 | 8 | 80 |
| mean最大绝对误差 | 0.00191865811 | 0.00154981427 |

native GPU FP32 ATen误差也逐项保存，但大偏移输入本身可能偏离FP64；不通过放松FP64门槛适配它。最终 memcheck/racecheck 各覆盖12个边界/压力用例×2
correction×2版本，共48项检查；均为0 errors，racecheck为0 hazards。该子集不能证明所有支持输入都经过sanitizer。

## 复现与证据

从 `/root/ops-ttir-dev` 执行：

```bash
CUDA_VISIBLE_DEVICES=5 bash var_mean/test_accuracy.sh
CUDA_VISIBLE_DEVICES=5 bash var_mean/test_performance.sh
```

脚本生成新结果目录，拒绝覆盖已有JSON。候选可由 make_candidates.py → make_dispatch.py → make_hybrid.py 重建；重建时核对已有记录而不覆盖旧证据。
所有结果和重建脚本已归档到Git；ignored artifacts可由同一原生编译入口再生，哈希在原始JSON中。

- [源文件/编译链证据](results/runs/r2/final_gpu5/pipeline.json)
- [主测全部配对样本](results/runs/r2/final_gpu5/full.json)、[独立复测](results/runs/r2/final_gpu5/repeat.json)
- [正确性](results/runs/r2/final_gpu5/accuracy.json)、[第二seed](results/runs/r2/final_gpu5/repeat_accuracy.json)
- [memcheck](results/runs/r2/final_gpu5/memcheck.log)、[racecheck](results/runs/r2/final_gpu5/racecheck.log)
- [候选选择与最终哈希](results/runs/r2/selection.json)、[验收记录](results/runs/r2/outcome.json)

KernelForge当前没有var_mean执行适配器：本实验使用其流程、证据和只读独立审查机制，不声明KernelForge正式算子集成资格，本实验未修改FlagGems算子源码。
已废弃的提取、正则生成失败和GPU7干扰记录均保留；不得拿这些数据替代本报告的修正基线结果。
