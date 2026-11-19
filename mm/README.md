# MM：六组 autotune baseline

当前使用官方31个方阵点逐shape autotune实测选出的 **6份不同baseline TTIR**。已冻结选择，不在正常验证或性能测试中重新调参。

TTIR首轮优化已完成六组各8个候选的尝试：仅128×32×64组保留无符号索引改写，所属3点主测/独立复测几何平均约1.0510×/1.0507×；其他五组继续baseline。大组部分A/A控制无效，因此没有31点整体可靠加速结论。完整结果、负结果与限制见 [report_round1.md](report_round1.md)，冻结选择见 [round1_selection.json](results/round1_selection.json)。`ir/<family>/optimized.ttir` 对应最终选择，baseline副本均明确标记fallback。

针对1152³/1280³匹配配置的第二轮IR借鉴见 [report_round2.md](report_round2.md)：对64×128×64组尝试8个功能候选，完整6点主测/复测未获可接受收益，保持原选择。官方TTIR可编译设备TMA诊断，但其每CTA384字节global scratch超出当前冻结工作区约定，未执行或计时。

另有用户授权的 [FlagGems Hopper / FlagTree 与教程对比](comparison/report.md)：同FlagTree环境配对主比较与原upstream环境参照分别报告，不属于纯TTIR改写收益实验，也不替换上述冻结基线。

优化版实际入口为 `optimized.load_kernels()` / `optimized.matmul()`，完整重放执行 `CUDA_VISIBLE_DEVICES=6 bash mm/test_optimized.sh`。原基线验证入口和下面的六份baseline保持不变；[optimization_plan.md](optimization_plan.md)是原基线阶段的冻结契约，用户授权后的优化计划见 [optimization_round1.md](optimization_round1.md)。

- [baseline.json](baseline.json)：31个shape到6组配置的映射、每组TTIR/编译选项/来源hash。
- [common.py](common.py)：`config_for_shape` 选择配置组，`compile_ir(family_id)` 编译该组TTIR，`launch` 使用该组grid和签名；`matmul(a,b,kernels,manifest)` 提供同样的自动dispatch。
- [export_baseline.py](export_baseline.py)：逐31点核查源码生成的TTIR/cubin与已冻结组一致，不重新autotune。
- [freeze_baselines.py](freeze_baselines.py)：从保存的autotune证据确定性重建六组；已有组只检查，不隐式覆盖。

| 配置组（BM×BN×BK） | warps / stages | baseline文件 | shape数 |
|---|---|---|---:|
| 32×64×32 | 2 / 5 | [baseline.ttir](ir/bm32_bn64_bk32_g8_w2_s5/baseline.ttir) | 1 |
| 128×32×64 | 4 / 4 | [baseline.ttir](ir/bm128_bn32_bk64_g8_w4_s4/baseline.ttir) | 3 |
| 64×128×64 | 4 / 4 | [baseline.ttir](ir/bm64_bn128_bk64_g8_w4_s4/baseline.ttir) | 6 |
| 128×256×64 | 8 / 3 | [baseline.ttir](ir/bm128_bn256_bk64_g8_w8_s3/baseline.ttir) | 18 |
| 64×256×32 | 4 / 4 | [baseline.ttir](ir/bm64_bn256_bk32_g8_w4_s4/baseline.ttir) | 1 |
| 128×128×32 | 4 / 4 | [baseline.ttir](ir/bm128_bn128_bk32_g8_w4_s4/baseline.ttir) | 2 |

所有组GROUP_M=8、CTAs=1，无activation；M/N/K和行stride为运行时参数，tile大小为编译期常量。31个性能shape为 `M=N=K=256…4096，步长128`，与官方教程一致。自动dispatch只接受冻结的31点，其他shape不会自动套用配置。配置选择来自 [autotune统计报告](autotune_ttir_report.md) 的实际记录，可能随重新测量波动；本轮使用其固定映射，不重复选优。

源码是官方Triton v3.7.0 tutorial 03提取函数，计算体不变。来源、快照SHA256及本次未远程独立核验的限制见 [upstream/provenance.json](upstream/provenance.json)。同组baseline与未来候选均须使用相同原生TTIR编译、加载、启动和计时流程。

```bash
# 在仓库根目录运行；先确认设备空闲
CUDA_VISIBLE_DEVICES=6 bash mm/test_accuracy.sh
CUDA_VISIBLE_DEVICES=6 bash mm/test_performance.sh
```

性能入口验证全部31点源码/TTIR的完整options、target、LLVM/PTX/cubin字节一致、输出逐位一致，然后验证每组准确性，计时31点×5轮（rep=20ms CUDA graph），用seed=123独立进程复测，最后对6组分别执行memcheck/racecheck。每次写新的results/runs目录。计时不含编译、加载、分配和Python launch；未锁定/核实锁频，未主动清缓存。设备进程前后检查不等于系统级预约或连续独占证明。

本轮FP16、SM90，M/N/K与行stride均为正16倍数、列stride=1、指针对齐、无别名；支持行尾padding。每组另外执行边界与压力正确性用例，不把额外用例当作自动性能dispatch范围。容差和完整协议见 [optimization_plan.md](optimization_plan.md)。当前实验报告见 [baseline_report.md](baseline_report.md)。

旧单配置文件 `ir/baseline.ttir` 保留原字节，**仅为历史基线，不参与当前dispatch**。旧manifest/代码/契约/报告存于setup_attempts/v3，旧原始结果保留；当前6份活跃baseline位于上表的子目录。
