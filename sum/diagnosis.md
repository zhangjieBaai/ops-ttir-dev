# Sum 瓶颈诊断与下一轮优化方向

最有价值的两个方向是：**在当前 TTIR 契约内测试完整 CTA 的无掩码向量加载；另建契约研究小输入单 kernel 路径**。
中等输入值得重新选择分块和 warp 数。大输入已经接近 HBM 带宽上限，继续单独缩窄整数、压寄存器或改加法树，优先级较低。
这些是下一轮待验证假设，本次没有生成新候选，也没有改变 baseline 或 optimized 选择。

## 采集范围

2026-11-19，北京时间。官方 Triton 3.7.0，原 FP32 连续全量 sum，两阶段 TTIR 和原生编译/启动路径不变。
本次 GPU5：`GPU-8b3add97-9c35-553e-695d-14cac5a9ea51`，H100 80GB，132 SM，L2 实测属性为 52,428,800 B（50MiB）。
前一轮主测在 GPU6，本次绝对耗时不与其混合汇总。先重新运行全部30点 baseline pipeline 检查，再使用已验证且哈希相同的候选。

测量分三类：

- 无 profiler：6 个代表 shape，分别测 stage1、stage2、完整两阶段图；每项5轮，每轮 rep=20ms，随机测量顺序，事先预热。
- NCU 2025.2：6点 baseline 的 SOL/launch/occupancy 初查；32MiB、256MiB 输入的 baseline/i32 深查；非对齐64MiB附近输入的 baseline/cg 对照及 sector 计数补查。
- Nsight Systems 2025.3：n=1024，单次 graph replay 内含100次完整 sum、200个 kernel 节点。

使用设备 UUID 锁和约1秒一次进程/时钟/温度/功耗采样，所有绑定记录均未观察到其他 GPU5 compute 进程。
采样不能排除采样间隙内的短暂活动；没有主动锁频。NCU 也明确提示了未锁频。
所有诊断计数和时间分解为 C 级范围证据；新优化的收益仍是 D 级假设。
已有候选的无整体收益结论沿用首轮完整配对证据，不能由 profiler 时间重新排名。

## 无 profiler 的时间构成

原始值来自 [stage_timing.json](results/runs/diagnosis_20261119/stage_timing.json)。单位 µs。

| n | 输入大小 | B / stage1 CTA数 | stage1 | stage2 | 完整两阶段 | stage2单独时间/完整时间 |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 4KiB | 32 / 32 | 1.2614 | 1.2600 | 2.5366 | 49.7% |
| 65,536 | 256KiB | 256 / 256 | 1.5956 | 1.3344 | 2.9353 | 45.5% |
| 1,048,576 | 4MiB | 1024 / 1024 | 2.0649 | 1.3820 | 3.5766 | 38.6% |
| 8,388,608 | 32MiB | 4096 / 2048 | 5.9397 | 1.4171 | 7.5317 | 18.8% |
| 16,777,216 | 64MiB | 4096 / 4096 | 22.8921 | 1.5836 | 24.7115 | 6.4% |
| 67,108,864 | 256MiB | 8192 / 8192 | 85.2430 | 1.9326 | 88.1743 | 2.2% |

stage2 只启动一个 CTA，数据仅为 stage1 的部分和；该阶段长期处于约1–2µs量级。
这些是分别捕获的三个图，缓存、调度和计时边界存在差别，不能要求两个单独时间精确相加，也不能把比例当作严格 Amdahl 上界。
它们用于判断值得优化哪个阶段。

小输入下，第二次 kernel 的固定成本与第一次相当。大输入下，即使第二阶段明显变快，对完整调用的影响也很小。
最大点按输入字节/第一阶段时间计算的有效读带宽为 **3.149TB/s**，32MiB点为 **5.649TB/s**。
后者不能称为 HBM 带宽，它包含缓存服务。输入从32MiB增至64MiB时，stage1时间约增加3.85倍，同时跨越这张卡的50MiB L2容量。
这是缓存容量边界的有力线索，但本次没有做清缓存/轮换缓冲区消融，不声称容量是全部耗时差异的唯一原因。

Nsight Systems 的[原始摘要](results/runs/diagnosis_20261119/timeline_stats_cuda_gpu_kern_sum.csv)显示：

| n=1024，图内100次调用 | 观测 |
|---|---:|
| stage1 平均 kernel 活跃时间 | 1.0925µs |
| stage2 平均 kernel 活跃时间 | 1.1936µs |
| 200节点总活跃时间 | 228.606µs |
| 首节点开始至末节点结束 | 246.622µs，即2.4662µs/调用 |
| 相邻节点间隙中位数 | 0.096µs |

图内没有每次 Python launch 的空隙，因此这里应称为**两次短 kernel 的固定设备成本与计算量不足**，
不能把当前 graph 指标直接解释成 Python 开销主导。timeline 的节点跟踪有开销，其耗时不能替代正常配对 benchmark。

## NCU 原始计数与解释

所有命令在仓库根目录执行，隔离 `CUDA_VISIBLE_DEVICES=5`，Python 为 `/root/venvs/ir-ops-dev/bin/python`。
下面是深查实际采集命令；其他命令与文件见 `results/runs/diagnosis_20261119/commands.json`。

```bash
CUDA_VISIBLE_DEVICES=5 ncu --profile-from-start off \
  --replay-mode application --cache-control none --clock-control none \
  --section SpeedOfLight --section LaunchStats --section Occupancy \
  --section MemoryWorkloadAnalysis --section WarpStateStats --section SchedulerStats \
  -o sum/results/runs/diagnosis_20261119/deep_app.ncu-rep \
  /root/venvs/ir-ops-dev/bin/python sum/diagnose.py --mode ncu \
  --sizes 8388608 67108864 --names baseline r1_i32 --per-process-output \
  --output sum/results/runs/diagnosis_20261119/deep_binding.json
```

原始 [deep_app.csv](results/runs/diagnosis_20261119/deep_app.csv) 中，stage1 的关键行如下。
ID4是256MiB baseline，ID6是同输入 i32；百分数为对应 NCU 指标的原始值。

| Metric Name | ID4 baseline | ID6 i32 | 单位 |
|---|---:|---:|---|
| launch__registers_per_thread | 56 | 32 | register/thread |
| sm__warps_active.avg.pct_of_peak_sustained_active | 54.84 | 94.22 | % |
| gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed | 92.32 | 91.76 | % |
| dram__bytes.sum.per_second | 3092639001101.73 | 3073592417061.61 | byte/s |
| sm__throughput.avg.pct_of_peak_sustained_elapsed | 11.88 | 7.55 | % |
| gpu__time_duration.sum | 87136 | 87776 | ns |
| smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio | 71.69 | 177.53 | inst（NCU原始单位） |
| smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio | 4.37 | 7.96 | inst（NCU原始单位） |
| smsp__average_warps_issue_stalled_wait_per_issue_active.ratio | 1.47 | 1.51 | inst（NCU原始单位） |

**解释：**占用率增加很大，但 HBM 带宽几乎不变；与首轮两次完整配对得到的约1.000×结果一致。
说明“寄存器数高就是主瓶颈”的假设在这个最大点不受支持。低 SM 算术吞吐也不意味着应该优先优化加法：归约每4字节约一次加法，本身算术强度低。
long-scoreboard 是主要等待线索，指向等待内存相关操作；这些 stall ratio 不是耗时百分比，不能直接相加或把177.53读成177.53%。
achieved occupancy 是活跃周期口径，也不是全设备所有时间都在有效计算的百分比。

92%左右的 DRAM 利用率与无 profiler 的约3.15TB/s相互支持：大输入已接近当前设备读带宽能力。
这不是“完全不可优化”，但保留相同字节数和两个 kernel 时，继续大幅提速的空间较小。
不把 NCU 的87.136/87.776µs比值当作新的功能加速比。

小输入的 [sol.csv](results/runs/diagnosis_20261119/sol.csv) ID0：
`launch__grid_size=32`、SM数132、`launch__waves_per_multiprocessor=0.02`、SM吞吐0.22%、DRAM吞吐0.01%、achieved occupancy6.02%。
这支持 n=1024 的工作量不足以占满设备；NCU规则给出的“预计加速百分比”不是实测收益，不用于本报告选择方向。

### 非对齐输入与 .cg 退步

`n=16,777,215` 的第一阶段 B=4096。静态[PTX记录](results/runs/diagnosis_20261119/ptx_observations.json)显示：

- 此 shape 的 baseline 是32条标量 `ld.global`；M为奇数，掩码边界缺乏足够整除信息。
- 相邻 `n=16,777,216` 的 baseline 是8条 `ld.global.v4.b32`。
- `.cg` 仅改变加载策略；该奇数点仍为32条标量加载，寄存器仍为62，没有解决标量化。

实际 tail 命令使用 `--sizes 16777215 --names baseline r1_cg`，采集 SOL、LaunchStats、Occupancy、MemoryWorkloadAnalysis。
原始 [tail_app.csv](results/runs/diagnosis_20261119/tail_app.csv) 的 stage1 行：

| Metric Name | ID0 baseline | ID2 cg | 单位 |
|---|---:|---:|---|
| l1tex__t_sector_hit_rate.pct | 74.96 | 0.01 | % |
| gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed | 81.14 | 51.77 | % |
| launch__registers_per_thread | 62 | 62 | register/thread |
| gpu__time_duration.sum | 25952 | 39168 | ns |

为了区分“多读HBM字节”与“更多缓存层请求”，额外只采集3个计数：

```bash
CUDA_VISIBLE_DEVICES=5 ncu --profile-from-start off --replay-mode application \
  --cache-control none --clock-control none \
  --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,lts__t_sectors_srcunit_tex_op_read.sum,dram__bytes_read.sum \
  -o sum/results/runs/diagnosis_20261119/tail_sectors.ncu-rep \
  /root/venvs/ir-ops-dev/bin/python sum/diagnose.py --mode ncu \
  --sizes 16777215 --names baseline r1_cg --per-process-output \
  --output sum/results/runs/diagnosis_20261119/sectors_binding.json
```

原始 [tail_sectors.csv](results/runs/diagnosis_20261119/tail_sectors.csv)：

| Metric Name | ID0 baseline | ID2 cg | 单位 |
|---|---:|---:|---|
| l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum | 8388608 | 8388608 | sector |
| lts__t_sectors_srcunit_tex_op_read.sum | 2097352 | 2814020 | sector |
| dram__bytes_read.sum | 67120640 | 67123968 | byte |

L2读sector请求增加约34.2%，HBM实际读取字节只变化约0.005%。不能把L1 miss次数直接等同于HBM流量，
也不能声称L2流量增加4倍。标量加载下 `.ca` 的 L1 复用被 `.cg` 取消，是退步的重要机制线索；
请求数量和请求延迟的独立贡献没有进一步拆开。
首轮无 profiler 配对中该点 `.cg` 为0.623685×，已有负结果与这里的计数相符。

### 局部加法树和 barrier

最大分块的PTX中，第一次shuffle之前有63次局部FP32加法。
[依赖分析](results/runs/diagnosis_20261119/dependency_observations.json)显示 baseline 最长虚拟PTX加法链为63，
`r1_fold`/`r1_i32_fold` 已缩短至18；加法总数不变。这是虚拟PTX依赖路径，不是实测SASS流水线延迟。
组合版仍没有整体收益，因此“改成平衡树”已经被部分测试，不能作为完全没试过的新主方向。
当前两条 `bar.sync` 也值得保留为观察项，但大shape的等待以long-scoreboard为主，
小shape n=1024 的两个kernel本来就没有 `bar.sync`。没有依据把全范围慢的主因归到 barrier。

## 下一轮候选优先级

| 优先级 | 方向 | 适用范围 | 当前冻结契约是否允许 | 最小验证与否定条件 |
|---|---|---|---|---|
| P1，严格TTIR首选 | 完整CTA无掩码加载，只有最后CTA走原masked路径 | M不满足向量掩码整除属性，尤其较大奇数输入 | 允许，保持签名/grid/workspace/options | 检查完整CTA出现v4加载；比较原`.ca` baseline，在奇数/相邻整除点配对；若分支/资源增加抵消收益则否定 |
| P1，整体收益潜力最大 | 小输入单CTA、单kernel归约 | 从n≤1024到≤16384逐级验证边界，阈值待测 | 不允许沿用原契约；须另建匹配baseline分组 | 测一个kernel与原完整两阶段；精度、尾部、边界都过，再冻结阈值；不能只测新kernel本体而漏额外初始化 |
| P2 | 分别选择stage1/stage2的B、warp数，替代一律sqrt(n)、4warps | L2内的小中输入，特别是4MiB附近 | 需新实验契约/分组；两边匹配配置与调参证据 | 小集合测试B=256/512/1024/2048、warps=1/2/4，比较完整两阶段；若stage1收益被stage2增大抵消则否定 |
| P3 | 大输入固定数量CTA、多tile循环，或更适合带宽的分块 | 超过L2容量的大输入 | 改grid/分块需新契约 | 保持相同读取字节，验证是否还能提高总带宽；当前约92%DRAM利用率，预期收益应保守 |
| 低 | 更激进i32、单独压寄存器、继续小改加法树或`.cg/.cs` | 已测试范围 | 部分允许 | 已有无收益/退步证据，不优先重复同类变体 |

**P1 的TTIR改写思路：**计算 `block_end=(pid+1)*B`，使用CTA一致的 `scf.if block_end<=M`。
完整CTA分支做无掩码 `tt.load`，尾部保留原mask/other=0，然后合流到原归约与store。
它不改变参与归约的元素集合，不增加第二次完整读取，也不用改变grid。
不能为未知M直接删除mask或伪造divisibility属性。
分支可能提高整个kernel的寄存器额度，宽/窄分支也可能被后端重新合并；必须看最终PTX/SASS，先做单项消融再组合i32。

这个候选的回报也可能较小：原baseline `.ca` 已通过L1部分补偿标量加载，
首轮相邻两个约64MiB点的总耗时差只有约1.4%。不能把 `.cg` 的严重退步当成相对原baseline可获得的大幅收益。

单kernel小输入路线可能更有回报，但它改变当前固定两阶段契约，不能把收益标为“第一轮只改TTIR且保持启动方式”。
若研究该方向，仍可从TTIR实现，必须明确新的签名/grid/输出约定以及匹配的两阶段参考。
不优先采用朴素 atomic_add：它需要输出初始化和正确同步约定，额外初始化设备工作必须计时，
而且浮点次序变化也需要重新验证，不能假设少一次kernel就一定少一次完整调用成本。

## 证据限制与复现

初始 `sol.ncu-rep` 使用kernel replay；后续 `deep_app`/`tail_app`使用application replay，并在每次重放运行100次预热。
缓存指标随重放策略和采集pass变化：例如32MiB点初查DRAM吞吐74.34%，后续深查baseline为29.07%、i32为2.90%；
深查相应L2命中也为69.70%与97.47%。这些差异不能当作i32的缓存优化收益，更不能当作精确匹配graph缓存状态的对照。
本报告只将它们视为缓存口径敏感性的观测，缓存边界判断以正常graph时间、实测L2容量及大输入稳定高DRAM带宽共同支持。

`--cache-control none`只是禁止工具主动清理缓存，不保证每个pass的缓存内容与真实graph完全相同。
[NVIDIA 2025.2 Profiling Guide](https://docs.nvidia.com/nsight-compute/2025.2/ProfilingGuide/index.html#cache-control)
说明kernel replay不能保存/恢复硬件cache；[application replay](https://docs.nvidia.com/nsight-compute/2025.2/ProfilingGuide/index.html#application-replay)
可以重跑前置活动，代价是整个进程的重复执行。没有锁频、没有重复独立NCU采样，这些计数用于诊断而非精确延迟排名。

本轮没有功能修改，不需要把已有全矩阵精度/性能证据重新跑成“新候选”；driver核对了原精度/编译链的TTIR和cubin哈希，
也在每个诊断输入上检查FP64参考。保留了所有工具日志、replay进程绑定、`.ncu-rep`、`.nsys-rep`、CSV、SQLite和采样。
工具身份见 [tools.json](results/runs/diagnosis_20261119/tools.json)。可复算至新文件：
`CUDA_VISIBLE_DEVICES=5 /root/venvs/ir-ops-dev/bin/python sum/analyze_diagnosis.py --output /root/tmp/sum-analysis-new.json`。
分析脚本只读取本轮证据，不使用GPU，拒绝覆盖已有结果。
目录中的 `timeline_export.log` 是已有SQLite被重复导出时的拒绝日志，不是timeline采集失败；`analysis.log`保留了首次解析非数值字段的失败，修正后的结果在`analysis_v2.log`。

使用的流程技能：[diagnose-kernel](/root/KernelForge/skills/diagnose-kernel/SKILL.md)、
[framework-workflow](/root/KernelForge/skills/framework-workflow/SKILL.md)、
[benchmark-evidence](/root/KernelForge/skills/benchmark-evidence/SKILL.md)、
[review-agent](/root/KernelForge/skills/review-agent/SKILL.md)。本报告是下一轮实验依据，不是新的收益验收。
