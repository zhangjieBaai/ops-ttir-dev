# 诊断：指标与反汇编怎么读

本文件只写**怎么读**和**读到什么意味着什么**。工具命令见各算子的 `profile_*.py`。

---

## NCU 指标的正确读法

### `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`

**字面含义**：平均每个 SM 的 Tensor 管线处于忙状态的周期占比。

**不是**：FLOPS 利用率。

该指标以 elapsed cycles 为分母；活跃周期不表示每周期完成满额运算。FLOP/s 还依赖指令形态、dtype、有效计算量、时钟、设备峰值定义与计时边界。**不存在通用的 0.8 换算系数。**

**用途**：观察 Tensor 管线的活动程度，结合实际指令、有效 `2MNK / time`、同精度设备峰值、waves 和调度指标形成假设。

**不能**：直接用 `1 / active` 给优化设硬上界。只有另行证明“不可缩短的管线工作时长、吞吐/时钟和工作量固定”才可建立这种模型，计数器本身不提供这些前提。原文的“恒定 0.8”“1/active 上界”和“约 3% 无法消除”缺少本库可追溯模型，已撤回；旧数值不能用于候选淘汰。

### `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio`

**含义**：按发射指令归一化的相应 warp 停顿周期比值；它不是百分比，可能大于 1。确切采集方式与指标后缀需查本机 NCU 版本的说明。

**做法**：同时看主要 stall 项、issue-active 与 eligible warps。仅当各项来自同一采集、同一实例聚合与分母、且原因集合完整互斥时，求和归一化才有描述意义；不要将缺项集合或不同采样口径拼成“总时间占比”。

**不能**：把百分比当作"可直接削除的时间"。这是采样/比值，不是时间账本。

### `smsp__warps_eligible.avg.per_cycle_active`

**含义**：每个调度器每周期平均有多少 warp 可立即发射。

**读法**：值低说明每调度器活跃周期内可发射 warp 少。原因可能是数据依赖、同步或执行资源不可用，须结合 stall 与 issue 指标判断；不能仅凭它认定“在等数据”。比较 resident 与 eligible 时必须使用相同的 SM/SMSP 聚合范围。

**与 occupancy 的区别**：occupancy 是"驻留多少"，eligible 是"能发射多少"。前者高、后者低说明隐藏延迟可能不足，尚未确定是哪条依赖。

### `launch__occupancy_limit_*`

**含义**：各资源分别允许的 CTA/SM 数。**取最小值才是实际限制者。**

例如若某次采集得到 `limit_registers=1`、`limit_shared_mem=1`、`limit_warps=8`、`limit_blocks=32`，则这组资源给出的**理论上限**为 1 CTA/SM，register 与 shared 同时约束。它不等于实际驻留数量；网格大小、尾波和运行时调度还会影响 achieved occupancy。

**操作含义**：要知道"降哪一项才有效"，必须看全部 limit，不能只看 regs。

指标解释依据 [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) 的 Metrics Guide、Scheduler Statistics 与 Warp State Statistics。工具版本和原始采集仍是具体数值的适用边界。

---

## 反汇编怎么定位问题

### 区分 `.text` 与完整 cubin

`.text._<kernel_name>` 的 SHA256 是可比较单元。完整 cubin 含调试信息，`loc` 差异会改变其哈希。

命令：

```bash
cuobjdump --dump-elf <cubin>   # 或从 .text 段取哈希
```

### 静态指令统计

`baseline_audit.md` 的做法——在相同后端下比较：

| 静态 SASS | 旧 llc baseline | 原生 Triton baseline |
|---|---:|---:|
| FADD | 239 | 174 |
| FMUL | 110 | 45 |
| FFMA | 0 | 65 |
| 总数 | 856 | 792 |

**能得出（C 级）**：这两份产物的浮点指令结构不同（旧版 0 条 FFMA），与后端融合设置差异相符。

**不能得出**：动态执行指令数、性能归因。静态计数是结构证据，不是时间证据。

**注意**：实验证明"在旧 llc 命令加 `-fp-contract=fast` 也能得到 65 条 FFMA，但机器码哈希仍不同"——说明差异不只在 FMA，还有地址生成、寄存器分配、调度。

### 分支级定位

`llir_comparison.md:37` 的方法：反汇编后按分支统计特定指令。

例：发现 2 条 STL / 2 条 LDL **只出现在** 8192<N≤16384 的双 tile 路径内。

**结论价值（C 级）**：若已检查全部实际执行路径且没有其他 local 指令，可排除小 N 的直接 spill 访存开销；仍不能排除统一 80-reg 资源额度的影响。

这个区分很重要：局部指令不存在 ≠ 局部不受全局资源限制。

---

## 归因纪律

### 一、现象与机制分开

| 可以写 | 不能写 |
|---|---|
| "long-scoreboard 从 5.67 升到 18.90" | "volatile 导致慢 3 倍" |
| "occupancy 从 33% 升到 65%，但 N=15872 仍更慢" | "提高 occupancy 无效" |
| "静态 FFMA 从 0 变 65" | "FMA 融合带来 4.5%" |

机制需要**针对性干预**来证明。没有干预就是相关性。

### 二、组合收益不可拆给单项

layer_norm 首轮 1.293× 来自结构修改（tile 分派）+ 算术修改的组合，但：

> 未对每个分支宽度或单项算术做完整消融，不能拆分总收益。——`report.md:75`

`arithmetic` 单项只有 1.009×，`tile_dispatch` 单独 1.288×，两者组合 1.293×——**加法关系不成立**（1.288×1.009≈1.300，接近但不等）。

### 三、资源变化不是因果证明

> 资源变化不是未经消融的性能因果证明。——`TTIR_EXPERIMENT_REQUIREMENTS.md` §8

寄存器从 60 降到 32，性能却可能不变或更差（见 [failure-patterns F4](failure-patterns.md#f4-occupancy-上升但性能下降)）。

---

## 典型诊断路径

拿到"某个候选变慢"时：

1. **先确认测量有效**：C1 通过？哈希匹配？配对样本完整？
2. **看寄存器/spill**：是否跨了驻留档位？是否有新 spill？
3. **看反汇编**：目标指令真的减少/增加了吗？还是被调度器还原了？
4. **看 stall 分布**：总停顿变化在哪一项？occupancy 与 eligible 是否背离？
5. **看逐 shape**：是全面变慢还是某个区间？几何平均是否掩盖了单点崩？

**第 5 步最容易被跳过。** `r2_loop1024` 几何 1.2570 看着不错，最差却是 0.7836。

## 性能不确定性与无效样本

报告配对比值时同时给逐 shape 耗时、全部轮次、最差点和不确定性。单次 graph 内的 replay 不是独立实验；本库核查脚本以保存的 baseline/candidate 配对轮次为重采样单位，在每个固定 shape 内 bootstrap 后汇总几何均值。该区间只反映已记录轮次的波动，不覆盖选择偏差、跨进程差异、温度/时钟漂移或外部 GPU 干扰。

独立进程复测用于检验结果稳定性；跨进程比较两个不同候选时，还需同会话 A/A 控制和交错进程顺序。历史数据没有的检查保持“未测”，不能因区间很窄而补写 PASS。

已有反例：`layer_norm/results/round2/measurement_validity.json` 明确弃用 `full_benchmark.json`，原因是 baseline 在轮次内明显漂移。原文件保留，另有 `full_stable.json`；后者不能抹去前者，也不能在缺少独立复测时升级为最终选择。GPU 占用快照只覆盖采样时刻，不是持续独占证明。


## 第二轮如何把诊断变成候选

1. 第一轮parent为80寄存器且有spill，提出减少分组活跃状态的假设。
2. 重载候选达到40寄存器、约65% achieved occupancy，但N=15872的NCU耗时108.416µs，parent为89.504µs；long-scoreboard/issue比值也更高。它推翻了“occupancy升高就够了”，没有单独证明volatile的性能因果。
3. 检查PTX看到了volatile load，追加普通ca load及不同分组；最终组合为48寄存器、无spill，中等N的NCU耗时与吞吐改善，大N仍有代价。
4. 依靠完整配对benchmark和独立进程复测选择，而不是选NCU最快点。最终A证据是几何1.4397×/1.4395×，NCU只是C级机制线索。

完整过程及失败证据见[第二轮复盘](layernorm-round2.md)。核对原始指标的聚合：本轮整卡DRAM吞吐用`dram__bytes.sum.per_second`，不是`avg`；单位从CSV读取，不自行假定都是byte/s或百分比。
