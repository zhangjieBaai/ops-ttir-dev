# 证据索引

每个数字 → 源文件 → 等级。**引用任何结论前先在这里核对。**

除另有说明，路径相对 `/root/ops-ttir-dev/layer_norm/`。性能等级不等于机制因果强度；静态事实用 C，正确性/工具结果按实际覆盖单列。

---

## LayerNorm 首轮

| 结论 | 数字 | 等级 | 源 |
|---|---|---:|---|
| 最终加速比（完整 30 shape、7 轮） | 算术 1.314× / 几何 1.293× | A | `report.md:5`、`results/full_benchmark.json` |
| 独立进程 seed=123 复测 | 算术 1.316× / 几何 1.294× | A | `report.md:5`、`results/independent_repeat.json` |
| canonical（仅删 loc） | 初筛约 1.000×（6 点、3 轮、rep=30ms） | B | `results/screening.json` |
| canonical `.text` 与 baseline 相同 | 完整 cubin 可不同 | C | `results/code_sections.json` |
| arithmetic（倒数复用 + rsqrt） | 初筛 1.009× | B | `report.md:70` |
| single_tile | 初筛 1.196×，64 regs | B | `report.md:71` |
| tile_dispatch_v2 | 初筛 1.358×，80 regs / 2 spill | B | `report.md:72` |
| tile_dispatch_math_v2 | 初筛 1.362×，80 regs / 2 spill | B | `report.md:73` |
| 正确性覆盖 | 6 版本 × 142 用例 = 852 次 | PASS（正确性） | `results/accuracy_full.json` |
| memcheck/racecheck | baseline/最终版各 17 用例，0 errors | PASS（工具覆盖） | `results/memcheck.log`、`results/racecheck.log` |
| 初筛 6 点 vs 完整 30 点差异 | 1.362× → 1.293× | B 初筛 / A 最终 | `results/screening.json`、`results/full_benchmark.json` |

`tile_dispatch_v2` 的完整矩阵几何均值 1.2880× 为 B（30 点、7 轮，未独立复测该候选）；只有最终 `tile_dispatch_math_v2` 有对应独立复测。首轮完整矩阵 rep=20ms，独立复测 5 轮 rep=10ms，不能写成两次完全相同的计时预算。

---

## LayerNorm 第二轮

下表是 **B** 级初筛（6 点 / 3 轮 / rep=10ms）；第二轮最终 **A** 级完整矩阵与独立复测见本节末尾。

| 候选 | 初筛 geo | 最差 shape | 寄存器 / spill | 源 |
|---|---:|---:|---|---|
| optimized（v1） | 1.3611 | 1.1898 | 80 / 2 | `results/round2/screen_batch1.json` |
| r2_delay | 1.4017 | 1.2021 | 80 / 4 | 同上 |
| r2_loop1024 | 1.2570 | **0.7836** | 32 / 0 | 同上 |
| r2_loop2048 | 1.2233 | **0.7508** | 32 / 0 | 同上 |
| r2_loop4096 | 1.2040 | 0.9812 | 40 / 0 | 同上 |
| r2_fine1024 | 1.3419 | 1.1385 | 80 / 0 | 同上 |
| r2_fine1024_reload | 1.4060 | 1.0124 | 40 / 0 | 同上 |
| r2_affine_v2 | 1.3689 | 1.2050 | 80 / 2 | `results/round2/screen_batch2.json` |
| r2_delay_affine_v2 | 1.2008 | 0.9600 | 94 / 0 | 同上 |
| r2_fine2048_reload | 1.4445 | 1.1076 | 48 / 0 | 同上 |
| r2_fine2048_delay | 1.4039 | 1.1902 | 80 / 0 | 同上 |
| r2_fold2048 | 1.2493 | 1.0301 | 39 / 0 | 同上 |
| **r2_fine2048_ca** | **1.4901** | 1.1439 | 48 / 0 | `results/round2/screen_ca.json` |

寄存器数据源：`results/round2/resources.json`；`compile_v2.json` 也有第一批资源，`compile_batch2.json` 只记录编译状态，不能用它支撑第二批寄存器数字。

### 第二轮最终选择（A 级）

`ir/optimized.ttir` 已更新为 `r2_mixed_stream`，SHA256 `6516d6ee0fce3d559cb288ab700f3000e9448f4756ef895a904ae434c166d9db`。

| 结论 | 数字 | 等级 | 源 |
|---|---|---:|---|
| 完整30点、7轮 | 算术1.4995× / 几何1.4397× | A | `results/round2/full_stable.json` |
| 同轮parent | 算术1.3171× / 几何1.2952× | A | 同上，label optimized 是历史parent |
| 独立进程seed123、5轮 | 算术1.4991× / 几何1.4395× | A | `results/round2/independent_repeat.json` |
| r2_fine2048_stream 完整30点、7轮 | 几何1.4322×；未独立复测此候选 | B，完整矩阵 | `results/round2/full_stable.json` |
| 相对parent | 22/30点更快，最大回退3.63% | A | `report_round2.md` |
| 最终资源 | 48 registers/thread，0 spill | C | `results/round2/resources.json` |
| 同一原生baseline下LLIR参考 | 约1.412×，新TTIR约1.440× | B，单进程3轮补充观测 | `results/round2/llir_comparison.json` |
| TTIR/LLIR优化版耗时几何比 | 0.9809 | B，单进程3轮补充观测 | 同上 |
| 新分支边界与sanitizer | 各120例/版本，memcheck 0 errors；racecheck 0 hazards/0 errors/0 warnings | PASS（记录覆盖） | `results/round2/boundary_accuracy.json`、`memcheck_accuracy.json`、`racecheck_accuracy.json`、对应工具日志 |

初次 `full_benchmark.json` 的N=1024有显著基线漂移，整轮中止并排除，不能用于最终收益。最终 `full_stable.json` 和独立复测均增加所有版本相同的untimed graph稳定阶段（rep=100ms）。仍是预分配CUDA graph时间，未锁时钟，不称冷缓存或端到端收益。

主测 rep=20ms，独立复测 rep=10ms。最差主测 shape 为 (4096,14336)，对 baseline 配对加速比中位数 1.1667×；parent 最差为 (4096,5120) / 1.1766×。相对 baseline 全部中位数改善不等于相对 parent 逐点改善。

选择记录：[`selection.json`](../layer_norm/results/round2/selection.json)，主测 SHA256 `baefd9852f18f9cf85ef5e3ec6b78f16ccd267b261441988c9f643d2f5e69b4c`。`provenance.json` 明确旧 `optimized` 标签指第一轮 parent，后续别名晋升不能改变历史样本身份。最终版数值验证哈希见 `accuracy_finalists.json` 与 `boundary_accuracy.json`；这些是已有记录的核对，本次知识探针没有重跑 LayerNorm 完整性能或 sanitizer。

### 第二轮经验复盘入口

[有效与无效经验](layernorm-round2.md)按最终组合、局部干预、未采用候选和流程失败分别登记。以下条目避免把最终组合收益误作单项收益：

| 结论 | 证据等级/边界 | 原始来源 |
|---|---|---|
| 小N改1024分组：N=2560为16.526→15.264µs | B，同会话完整矩阵；2048版未独立复测 | `results/round2/full_stable.json` 的 stream/mixed_stream |
| 延迟store+affine为94寄存器且最差0.9600× | B初筛/C资源；不是普遍禁止组合 | `screen_batch2.json`、`resources.json` |
| reshape/reduce折叠为39寄存器，初筛几何1.2493× | B初筛/C资源；并非所有宽窄解耦均失败 | 同上 |
| `.cs`的独立收益 | D，未有可接受的完整消融；不能从弃用测量抽取 | `measurement_validity.json`、`full_benchmark.json` |
| 验证别名KeyError发生在计时前 | C，测试入口失败，不是kernel失败 | `independent_repeat.log`、`accuracy_finalists.json` |

表中第二轮简写文件均位于 `results/round2/`。

### 第二轮最终候选 NCU（C级机制线索）

来源：[`ncu_finalists_summary.json`](../layer_norm/results/round2/ncu_finalists_summary.json)，原始 `.ncu-rep` 与CSV在同目录；base clock、cache-control=all。`optimized`标签为第一轮parent。

| N | parent / mixed NCU µs | parent / mixed 实际occupancy | parent / mixed DRAM Tbyte/s |
|---:|---:|---:|---:|
| 4096 | 31.360 / 24.576 | 35.4% / 56.1% | 1.613 / 1.905 |
| 8704 | 62.752 / 50.144 | 34.2% / 56.6% | 2.011 / 2.417 |
| 15872 | 89.664 / 97.696 | 33.4% / 54.7% | 2.723 / 2.494 |

吞吐来自`dram__bytes.sum.per_second`；不能误用按partition平均的`dram__bytes.avg.per_second`冒充整卡吞吐。NCU耗时不能代替常规graph benchmark，occupancy变化不证明全部收益的因果。

### 第二轮 NCU（`results/round2/ncu_batch1_summary.json`）

4 个 shape（1024/4096/8704/15872）、3 个 kernel：

| N | optimized | r2_fine1024_reload | r2_delay |
|---:|---:|---:|---:|
| 1024 | 18.112 µs | **13.664** | 17.920 |
| 4096 | 31.488 | **27.552** | 29.472 |
| 8704 | 63.872 | **53.408** | 57.056 |
| 15872 | **89.504** | 108.416 | 90.048 |

**这是单次 NCU 采集，不是 benchmark**，只用于机制解释。注意 15872 上 `r2_fine1024_reload` 反而最慢（108 µs）——与 F5 一致。

---

## 失败候选

| 候选 | 阶段 | 原因 | 源 |
|---|---|---|---|
| tile_dispatch | C0 | 分支张量宽度与零常量不匹配 | `results/failed_candidates.json` |
| tile_dispatch_math | 静态登记，未单独编译/执行 | 相同缺陷；不是第二次实测编译失败 | 同上 |
| r2_affine | compile | `PassManager::run failed` | `results/round2/compile_v2.json` |
| r2_delay_affine | compile | 同上 | 同上 |

---

## 跨层比较与 baseline 审计

| 结论 | 数字 | 等级 | 源 |
|---|---|---:|---|
| `.text` SHA256（教程原生 = 当前 baseline） | `4fb552d5…b52996` | C | `baseline_audit.md:8` |
| `.text` SHA256（旧 llc 重编译） | `813cd28b…9b70c3` | C | `baseline_audit.md:9` |
| 计算 IR 逐字节一致，差异仅调试元数据 | — | C | `baseline_audit.md:7` |
| 同协议四版对照 | 见下 | B | `llir_comparison.md:13-16`、`results/llir_comparison/same_protocol.json` |

同协议（30 shape、3 轮、rep=10ms、CUDA graph、GPU 7）：

| 指标 | 算术 | 几何 |
|---|---:|---:|
| LLIR base / LLIR opt | 1.5235× | 1.4759× |
| TTIR base / TTIR opt | 1.3173× | 1.2947× |
| LLIR base / TTIR base | 1.0454× | 1.0452× |
| TTIR opt / LLIR opt | 1.0961× | 1.0907× |

分解：`1.476/1.295 ≈ 1.045 × 1.091`。

资源对照（`llir_comparison.md:26-31`）：

| 版本 | regs | spill | CTA/SM | occupancy |
|---|---:|---:|---:|---:|
| LLIR baseline | 60 | 0 | 4 | 50.0% |
| TTIR baseline | 60 | 0 | 4 | 50.0% |
| LLIR optimized | 48 | 0 | 5 | 62.5% |
| TTIR optimized | 80 | 2 | 3 | 37.5% |

CTA/SM 由 `cuOccupancyMaxActiveBlocksPerMultiprocessor` 实测，非手算。

该 API 查询给定资源与 block size 下的**理论最大活跃 CTA 数**，不是 profiler 观察到的 achieved occupancy。上表是第一轮版本，不代表第二轮资源。

静态 SASS（`baseline_audit.md:16-21`）：

| | 旧 llc | 原生 Triton |
|---|---:|---:|
| FADD | 239 | 174 |
| FMUL | 110 | 45 |
| FFMA | 0 | 65 |
| 总数 | 856 | 792 |

---

## 环境与工具链

| 项 | 值 | 源 |
|---|---|---|
| Python | 3.12.3，`/root/venvs/ir-ops-dev/bin/python` | `results/baseline_pipeline.json` |
| Triton | 3.7.0（upstream，非 FlagTree） | 同上 |
| Torch | 2.12.0+cu130 | `results/round2/screen_batch1.json` |
| 设备 | H100 80GB HBM3，SM90 | 同上 |
| 编译 options | `num_warps=8, num_ctas=1, num_stages=3, enable_fp_fusion=True` | `common.py`，完整解析值见 pipeline JSON |

---

## MM 基线证据

路径相对本仓库根目录。入口为 [`mm/baseline_report.md`](../mm/baseline_report.md)，原始记录在 `mm/results/runs/baseline_20261118T124859Z_Vh0qLn/`。

| 事实 | 已有证据与范围 | 不支持的推论 |
|---|---|---|
| 源码/TTIR 同链 | `pipeline.json`：LLVM/PTX/cubin 字节一致，TTGIR parser 归一化一致 | 没有优化候选或加速比 |
| 正确性与 sanitizer | 两进程各 36 例；memcheck/racecheck 各 5 个 padding 边界例 | FP64 只覆盖小用例；不代表全部 shape CPU 验证 |
| 数值容差 | `atol=1e-2, rtol=1e-3`，部分 FP32/FP64 对照绝对误差 0.25 仍通过 | 不能写“全部绝对误差小于 1e-2” |
| 基线计时 | 15 shape × 5 轮，独立进程重复，GPU6 | 单一固定配置，不是教程逐 shape autotune 最优 |
| 资源负结果 | `autotune_failures.json`：5 个 Infinity 配置诊断为 shared-memory OutOfResources | Infinity 本身不能证明算术错误或具体失败原因 |
| 干扰运行 | GPU7 的 `baseline_20261118T124700Z_gO2bUH/performance_status.json` 被拒绝 | 不与正式 GPU6 数据混合 |

这些是已有记录的核对，本次没有新跑 MM 性能。配置和来源限制见 `mm/baseline.json`、`mm/upstream/provenance.json`，不能从 LayerNorm 的 tile/warp 经验推出 MM 最优配置。

## KernelForge 知识核查

本轮运行目录：[`ttir-knowledge-20261118`](../../KernelForge-workspace/runs/ttir-knowledge-20261118/)。代码为 [`verify_claims.py`](verify_claims.py)，冻结 venv、官方 Triton 3.7.0、SM90、物理 GPU0（UUID `GPU-cc248ebf-79d0-42d7-da65-99d1ef6f4ed1`）。独立 FP32 探针只验证能力与数值现象，**不增加 LayerNorm 候选，不产生新算子性能收益**。

| 核查 | 结论 | 证据 |
|---|---|---|
| KernelForge 任务导航 | LayerNorm 未注册执行适配器；使用证据/诊断/审查方法 | `task_context.json` |
| baseline pipeline（C） | 官方源码/TTIR baseline 的 LLVM/PTX/cubin 字节一致；此时 optimized 是首轮 parent | `pipeline.json`；之后晋升的 pipeline 见 `layer_norm/results/round2/pipeline_after.json` |
| 张量 negf（C） | 解析后在 ConvertTritonToTritonGPU 合法化失败 | `probes-v4/tensor_negf.ttir`、`probes-v4.log` |
| `-0 - x` 与 `+0 - x`（C） | 有限值/Inf/equal-NaN 比较通过；仅前者通过零符号检查 | `probes-v4/audit.json`、两份 `*_subf.*` 产物 |
| 内联汇编（C） | `tt.elementwise_inline_asm` 的 `neg.f32` 编译、执行并通过零符号检查 | `probes-v4/inline_asm.*` |
| 双 load（C） | 默认/默认在 PTX 中合并为 1 条；默认/ca 和默认/volatile 各保留 2 条，后者含 `ld.volatile.global` | `probes-v4/{same,ca,volatile}_load.*` |
| 原始样本复算 | 8 份 JSON 的每条样本比值、配对顺序、轮数、中位数核对通过；按输入哈希保留派生区间 | `probes-v4/audit.json:historical_samples` |
| 官方语义来源 | Triton v3.7.0 op 定义、NVIDIA NCU/PTX 文档，网页快照及哈希保留 | `sources/manifest.json` |

每份输入 TTIR、逐阶段产物和脚本的 SHA256 在 `probes-v4/audit.json`；输入名为 `<probe>.ttir`，编译器回显与其他阶段为 `<probe>.compiled.<ext>`。已逐一核对输入和产物哈希。早期 `probes/audit.json` 漏提取 `ld.volatile.global`；`probes`、`probes-final`、`probes-v3` 还曾将编译器回显 TTIR 写回输入文件，导致原始输入哈希与保留文件不符。这些有缺陷的中间记录保留供追溯，最终引用以分开保存输入/产物的 `probes-v4` 为准；复跑不混为新的独立性能证据。

以下是对**历史配对轮次**做 2000 次 bootstrap 的派生结果（seed=42；每个固定 shape 内重采样配对轮次，再汇总几何均值）：

| 版本/原始测量 | geo | 条件 bootstrap 95% 区间 | 配对样本数 |
|---|---:|---|---:|
| 首轮 math_v2 主测 | 1.292535 | [1.292377, 1.292776] | 210 |
| 首轮 math_v2 独立复测 | 1.293553 | [1.293388, 1.294154] | 150 |
| 第二轮 mixed_stream 主测 | 1.439667 | [1.439499, 1.439900] | 210 |
| 第二轮 mixed_stream 独立复测 | 1.439502 | [1.439267, 1.440065] | 150 |

这是固定矩阵、已有轮次条件下的不确定性；区间很窄**不能**证明无 GPU 干扰、无时钟漂移、跨进程系统误差受控或机制因果成立。历史未测的 A/A 控制仍未测，详见 [不确定性说明](diagnosis.md#性能不确定性与无效样本)。原始 JSON 不改写，新派生分析不提升它们的原有有效性或等级。

---

## 引用纪律

1. 引用 A 级性能须同时说明矩阵和计时边界；B 级写明“初筛”或“完整矩阵、未独立复测”，不能统称为初筛。
2. 引用 C 级静态/能力事实可描述已验证现象，不能推导性能收益；因果解释另看消融证据。
3. 任何跨层/跨口径数字，先确认 baseline 相同。
4. 数字与本表冲突时，**以原始 JSON 为准**，并修正本表。
