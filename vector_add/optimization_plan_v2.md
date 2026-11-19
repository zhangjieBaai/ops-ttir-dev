# Vector add 第二轮 TTIR 优化计划与 NCU 诊断

状态：**NCU 诊断已完成；下列新 TTIR 候选尚未生成、编译或执行。**
第一轮 `optimized.ttir` 继续等于 baseline，本计划不改变历史结果或选择。

## 结论与优先级

下一轮优先研究 **TTIR 内按长度分派：小尺寸保留 baseline，中等尺寸试用
split512，再单独验证与 streaming store 的组合**。目标是保留中等尺寸的局部
收益，同时消除 split512 在小尺寸的回退。大尺寸已接近实测 DRAM 带宽上限，
继续压寄存器或泛化缓存策略的优先级低。

不能承诺整体超过 2%。本次退出 profiler 后的三轮复核中，split512 在
2^20/2^21 为 1.0500x/1.0944x，`.cs` 在 2^22 为 1.1883x。
即使把这三点收益无损组合、其余十三点严格持平，16 点等权几何均值也只有
约 1.0197x。这个算式是**乐观的选点模型，不是组合实测结果**；分支开销、
编译布局变化和其他尺寸上的收益/回退都尚未知。

## 本次诊断方法与证据边界

- 环境：官方 Triton 3.7.0、指定 venv、H100 80GB HBM3 SM90、NCU 2025.2.0。
  GPU 7，UUID `GPU-b024aea2-2ffa-7211-b551-8e5575887423`；132 SM，NCU 报告
  L2 容量 52,428,800 bytes（50 MiB）。运行前与各进程前后保存设备快照，
  本任务没有并发提交其他 GPU 工作；快照不构成全时段独占证明。
- 重新跑过本算子的 pipeline 核查：全部官方 16 个尺寸加四个边界，源码/文件
  baseline 的 LLVM、PTX、完整 cubin 一致。冻结源、baseline、options、launcher
  均未修改。每个 profile 启动前后对真实输入做逐位数值核对。
- 32-node CUDA graph、同一输入与预分配输出；8 次整图预热。用 profiler
  start/stop 只采最后一张图的第 17 个 kernel，跳过 JIT、初始化与验证。
- 16 份 kernel-replay profile：7 个 baseline 长度、缓存/分组候选对照，以及
  3 份清缓存对照；另有 4 份 application-replay profile 复核缓存敏感点。
  前置 pilot 一份单独保留。所有 `.ncu-rep`、CSV、命令、日志、IR/cubin 哈希
  和输入/设备绑定都已保存。
- 主矩阵收集 SpeedOfLight、LaunchStats、Occupancy、MemoryWorkloadAnalysis、
  SchedulerStats、WarpStateStats、InstructionStats；没有使用 `--set full`。
  所有对照显式 `--clock-control none`；缓存因子分别为 none/all。
- `--cache-control none` 仅表示工具不主动清缓存，**不保证每次 replay 都有
  与自然执行相同的缓存状态**。普通 kernel replay 的保存/恢复及多 pass 会
  扰动缓存。按 NVIDIA 指南，用 application replay 在每一 pass 重跑预热，
  但注入、串行化、计数器收集仍会改变执行环境。
- NCU 单点计数器属于 **C 级机制线索 / CORRELATION_ONLY**，不作为加速比。
  没有采集 CPU 时间线，不能由这些数据判断 Python launch 或应用端瓶颈。
  本任务关注的主指标仍是预分配、重复输入的 graph kernel 时间。
- CSV 会自选 K/M/G/T 单位；`analyze_ncu.py` 读取每列实际单位再归一化。
  下面的 MB/TB 是十进制单位，MiB 是二进制单位。

原始来源：

- [pipeline](results/runs/ncu_setup_20261119/pipeline.json)
- [16 份 kernel profiles 索引](results/runs/ncu_matrix_20261119/index.json)
- [4 份 application profiles 索引](results/runs/ncu_application_20261119/index.json)
- [归一化汇总、原始 metric 名/值/单位/行号和静态反汇编](results/runs/ncu_analysis_20261119_v3/analysis.json)
- [退出 profiler 后的三轮配对复核](results/runs/ncu_setup_20261119/unprofiled_check.json)
- [NVIDIA Cache Control / Replay 说明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#cache-control)

## 原始计数器摘录与解释

以下首先给出 raw.csv 第三行的读数，归一化后保留显示精度。路径为
`results/runs/ncu_matrix_20261119/baseline_n<N>_cache_none/raw.csv`。
Compute 使用 `sm__throughput.avg.pct_of_peak_sustained_elapsed`，DRAM 使用
`gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`；二者都不是 FLOP/s。

| N | CTA 数 | NCU 时间 µs | Compute % | DRAM % | Achieved occupancy % | L2 总命中率 % |
|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 4 | 2.336 | 0.040 | 0.033 | 5.96 | 98.08 |
| 65,536 | 64 | 2.688 | 0.615 | 0.491 | 6.02 | 97.90 |
| 1,048,576 | 1,024 | 3.936 | 5.212 | 1.270 | 33.01 | 91.50 |
| 2,097,152 | 2,048 | 7.360 | 5.093 | 50.076 | 64.47 | 82.04 |
| 4,194,304 | 4,096 | 19.296 | 3.759 | 75.850 | 80.49 | 35.19 |
| 8,388,608 | 8,192 | 35.584 | 3.957 | 84.213 | 77.38 | 34.11 |
| 134,217,728 | 131,072 | 520.544 | 4.277 | 92.312 | 83.13 | 34.57 |

### 小尺寸：网格不足与依赖延迟，不能靠压寄存器补足 CTA

N=4096 只有 4 个 CTA，最多占用 4 个 SM；N=65536 也只有 64 个 CTA。
NCU 的理论 occupancy 为 100%，26 registers/thread（分配粒度下实际 32），
register 和 warp limit 均允许 16 CTA/SM。编译 metadata 为 0 shared、0 spill；
NCU 额外报告的 driver-reserved 1024 bytes/CTA 不应算成算法共享内存。

因此低 achieved occupancy 与不足一波的网格一致，不支持“寄存器太多”的
解释。当前固定 grid 的 TTIR 主线不能增加 CTA 数；仅把一个 CTA 的逻辑张量
拆成 512+512 也不会把 4 CTA 变成 8 CTA。

split512 在这个点的 eligible warps/scheduler 从 0.0813 下降到 0.0504，
long-scoreboard/issue 从 7.1486 升到 14.6111。两版动态 warp 指令数都是
656，资源也相同。SASS/PTX 可见 baseline 先发出四组 load，split512 先完成
前 512 的 load/add/store，再处理后半段。这支持“分阶段表达削弱小网格的
访存重叠”假设；这些 stall 比值不是百分比，也不是可直接消除的时间。
普通配对复核在 N=4096 测得 split512 0.8750x，方向一致。

### 中等尺寸：缓存行为随表达变化，不只是寄存器或指令条数

下面均来自 application replay、cache-control=none，避免用 kernel replay
的缓存状态代替每 pass 的应用预热。每点只有一份 profile，仍是 C 级证据。
对应目录为 `results/runs/ncu_application_20261119/<name>_n<N>/raw.csv`。

| N / 版本 | NCU 时间 µs | DRAM read MB | DRAM write MB | L2 总命中 % | TEX read-sector 命中 % |
|---|---:|---:|---:|---:|---:|
| 2^21 baseline | 7.296 | 4.102656 | 3.661056 | 73.95 | 43.64 |
| 2^21 split512 | 6.208 | 1.838336 | 1.523712 | 87.60 | 75.47 |
| 2^22 baseline | 18.880 | 33.214976 | 13.180160 | 34.97 | 0.71 |
| 2^22 store `.cs` | 17.248 | 29.217280 | 12.670976 | 40.99 | 8.21 |

TEX read-sector 命中率由
`lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum / lts__t_sectors_srcunit_tex_op_read.sum`
计算；它与包含写入等其他请求的 `lts__t_sector_hit_rate.pct` 不同。
2^21 两版 read/hit sectors 为 524288/228820 与 524288/395660；
2^22 为 1048576/7448 与 1048576/86084。

2^22 的三份数组总大小是 48 MiB，已接近报告的 50 MiB L2 容量。
`.cs` 被编译为 `STG.E.EF.128`，baseline 为 `STG.E.128`；两版其余静态
opcode 数量、26 registers、0 spill 相同。application replay 中 `.cs`
减少了约 12.0% 的 DRAM 读取，与缓存局部收益方向一致。
这支持写入淘汰策略影响下一次读取的假设；尚未隔离出确切的 L1/L2 淘汰、
集合冲突和各项写回行为，不能仅凭接近容量就宣称已证明容量冲突机制。

反例同样保留：普通 kernel replay、cache=none 时两版 NCU 都是 19.296µs，
没有重现普通计时收益；cache=all 时 baseline/`.cs` 的读取均约 33.56 MB，
NCU 时间为 15.488/15.648µs，也没有显示 `.cs` 的优势。**这些 profiler
时间不能计算为正式性能结论**；这里只说明缓存状态和 replay 方式会影响观察。
实际自然执行计时必须在退出 NCU 后进行。

`evict_first` 也生成 `STG.E.EF.128`，但有额外 policy 构造指令；2^22 的
动态 warp 指令数为 770048，baseline/`.cs` 为 671744。因已有 `.cs`
表达更直接且第一轮未输给它，不优先继续扩大 eviction-policy 搜索。

### 大尺寸：DRAM 传输主导，优先保护吞吐

N=2^27 的 `dram__bytes.sum.per_second` 是 3.094453 Tbyte/s，DRAM 为
92.311858%，读取约 1.073777 GB、写入约 537.021440 MB。相对本算子每个
元素一次加法和 12 字节逻辑读写，其算术强度只有 1/12 FLOP/byte。
资源相同的 `.cs` 在本次 profile 为 3.080422 TB/s，未显示带宽改善。
NCU 单点和普通计时均支持将此区间视为 DRAM 传输主导。

这里不把 `1/0.923` 当作严格可用加速上限；计数器峰值定义、时钟、写回
边界与执行重叠都不足以建立那样的硬界。也不因 occupancy 已高就推断
所有 latency 都被隐藏。当前没有依据优先引入 shared、TMA、复杂流水或
更大 unroll；它们不会减少这个独立输出算子必需的逻辑读写。

## 退出 NCU 后复核

原 `benchmark.py`、同一冻结基线、完整 16 点、3 轮、rep=20ms、seed=941，
所有 A/A 中位数均在原 [0.97,1.03] 范围内。

| 候选 | 16 点几何均值 | 95% 配对轮次区间 | 局部观察 | 最差点 |
|---|---:|---|---|---|
| `.cs` | 1.01031x | [1.00848,1.01123] | 2^22: 1.18833x | 32768: 0.98714x |
| split512 | 0.95821x | [0.95740,0.95955] | 2^20: 1.0500x；2^21: 1.09445x | 32768: 0.86326x |

这是 **B 级诊断复核**，轮数未达到最终晋升标准；不会替换第一轮 A 级主测/
独立复测。split512 的未分派版本仍整体回退，下一轮只能作为局部候选组件。

## 第二轮实验约束

1. 主线沿用第一轮唯一 `ir/baseline.ttir`，FP32、N 为正的 16 倍数、N<=2^27、
   16-byte aligned、contiguous、无别名；ABI、grid=ceil(N/1024)、1024 elements/CTA、
   4 warps 和完整 parsed options 不变。只直接编辑新的 TTIR，禁止内联 PTX、
   下游 IR 或 binary 改写。旧候选、source、baseline、selection 不覆盖。
2. 尺寸分派写在同一个 TTIR 的 `scf.if` 内，按运行时 N 选择，所有 CTA
   仍负责原先的 [pid*1024, pid*1024+1024)。每个分支都必须保持尾部 mask、
   数值行为与输出覆盖；标量条件是 CTA-uniform 的，仍需实测分支开销。
3. 这次只交付诊断和计划。开始第二轮实现时，新增不可变的协议 manifest，
   包含本文件、执行脚本、测试列表、候选生成器和父 TTIR 的 SHA256；先提交
   协议，再生成候选。旧 v1 manifest/证据继续保留。
4. 第一轮 16 个官方点和旧 tuning/validation 已全部暴露。它们作为固定回归和
   主指标矩阵使用，不再称作未见 holdout；换 seed 不会恢复独立性。

## 候选顺序、预算和可证伪条件

所有候选都从 baseline 或注明的已验证组件生成；下表名字只是预留，尚无对应文件。
首批最多 6 个，再允许最多 2 个由观察明确支持的修正版，合计最多 8 个。

| 次序 / 候选 | 直接 TTIR 变换 | 主要假设与成功信号 | 否证 / 停止条件 |
|---|---|---|---|
| 1 `r2_guard_split` | 2^20<=N<=2^21 使用原 split512，其他 N 原 baseline；仍逐 lane mask | 中等区间保留局部收益，小尺寸退出分阶段路径 | 小尺寸或大尺寸回退超过 2%，或组合资源上升导致原收益消失；先定位分派成本 |
| 2 `r2_guard_split_cs` | 在 1 的基础上，仅上区间 N>2^21 使用 `.cs`，其他分支保持 1 | 用少量连续区间结合两种机制，不能逐 benchmark 点挑 kernel | 相对 1 没有可复现改进，或未见长度回退；不得用 N==2^22 的专用分支补成绩 |
| 3 `r2_split_reorder` | 在中等分支中试前半 load 后提前发出后半 load、再完成前半 store；不使用 volatile | 在保留独立 512 表达的同时增加 load-use 重叠；检查 SASS 是否真改变顺序 | 后端编译为相同 `.text` 即停止该形式；寄存器/活跃请求增加而自然计时不改善则保留负结果 |
| 4 `r2_guard_reorder_cs` | 仅当 3 有效，替换 2 的中等组件 | 验证调度和缓存策略交互，和各单项同进程比较 | 不把单项收益相乘当作组合预测；不能继承父版本 PASS |
| 5 `r2_fulltile`，低优先级 | N%1024==0 的 uniform fast path 去掉 load/store mask，其他情况原 masked 路径 | 若 TTIR 文件入口能消去多余 mask 初始化/地址谓词，短 kernel 或可少做工作 | 历史 LLIR 路线已有无收益背景；仅一个 TTIR 探针，`.text` 相同或小尺寸无改善即停，不再穷举 |
| 6 组合消融，按需 | 只组合上述已实测有效的独立变换 | 比较改动最少版本与组合，报告相互影响 | 无有效单项则不生成；最多两个后续修正版须先写新假设 |

初始阈值只来源于本次已测行为分区，未宣称为最优。只允许在 tuning 中
比较一组有解释的邻近阈值，不开展逐长度查表。各分支共享编译后的统一资源
预算，必须记录 registers/spills/shared、实际 SASS 和分支外回归。

## 验证、计时和接受规则

- 正确性沿用 v1：两个 seed、全部官方长度和原 36 个用例、signed-zero/overflow/
  NaN 等压力分布、canary、输入不变与不支持参数拒绝。新增每个分派边界的
  N=threshold-16/threshold/threshold+16，以及非 1024 倍数，先正确性再计时。
- 持续 tuning 使用旧 8 点及诊断点；finalist 内容固定后，再首次测下列新的
  非幂次长度性能：`[12288,49152,196608,786432,1572864,3145728,3670016,
  4718592,6291456,12582912,25165824,100663296]`。这些属于辅助泛化集，
  不混入官方 16 点均值；靠近已测尺寸，不能声称统计上完全独立。
  查看该集后再改候选，就必须如实标记其已参与调优。
- 初筛 3 轮；finalist 完整官方 16 点 9 轮，独立进程 7 轮、不同 seed。
  沿用原 CUDA graph、预分配输出、rep=20ms、baseline/候选随机交错、逐点
  A/A、全样本保存、配对中位数与 5000 次轮次 bootstrap。不得用 NCU 时间替代。
- 晋升仍要求主测/复测整体几何均值均>=1.02、两次区间下界>1、原 validation
  子集>=1.01、每点中位数>=0.95，A/A 均有效。辅助泛化集逐点报告并解释
  回退；不因它平均较快而放宽官方主矩阵条件。最终 memcheck/racecheck 必须通过。
- 新变换若声称改变缓存，选一个自然计时有效点做 application replay；
  不必为所有形状重复 NCU。若声称 load-use 改善，先检查 SASS，再补少量
  scheduler/stall 数据；资源或计数器改善不能单独晋升。
- 编译/正确性预算 20 分钟，初筛 GPU 时间预算 10 分钟，诊断最多 4 个新增
  application profiles；完整矩阵和独立验收另计。若 3 个不同、有效候选均未
  比前一版本改善，先复盘，不继续堆同类 cache/调度参数。无法达到 2% 时
  保留 baseline 和局部收益结论，不降低门槛或删除慢点。

## 本轮不优先做的方向

无新证据时不重复泛化 `.cg`、load evict-last、硬压 maxnreg 或扩大 unroll；
第一轮已有无收益/退步，且大尺寸带宽与本次无 spill 的事实不支持它们优先。
改变 BLOCK_SIZE、grid、warps 才能更直接研究小尺寸并行度，但它们需要
另建匹配配置的 baseline，属于独立实验组，不计入本计划的固定启动主线。

## 复现本次诊断

先检查空闲设备，显式设 `CUDA_VISIBLE_DEVICES=7`。以下输出目录必须是新路径：

```bash
py=/root/venvs/ir-ops-dev/bin/python
CUDA_VISIBLE_DEVICES=7 "$py" vector_add/run_ncu.py --output /root/tmp/vector-add-ncu-new
CUDA_VISIBLE_DEVICES=7 "$py" vector_add/run_ncu_application.py --output /root/tmp/vector-add-ncu-app-new
```

原始 profile 的完整参数在各 `manifest.json`；关键设置为：

```text
ncu --profile-from-start off --kernel-name add_kernel --graph-profiling node
    --launch-skip 16 --launch-count 1 --clock-control none
    --replay-mode kernel --cache-control none|all --section ...
ncu --profile-from-start off --kernel-name add_kernel --graph-profiling node
    --launch-skip 16 --launch-count 1 --clock-control none
    --replay-mode application --app-replay-mode strict --cache-control none --metrics ...
```

KernelForge 提供 diagnose-kernel/benchmark-evidence/review-agent 方法与评审；
执行仍使用本项目原生 Triton 路径，未新增 KernelForge vector_add 执行适配器。
