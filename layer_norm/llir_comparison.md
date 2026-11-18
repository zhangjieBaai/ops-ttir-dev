# 为什么本轮 TTIR 加速比低于 LLIR

> 后续核查：两轮计算 IR 相同，旧教程原生 cubin 与当前 baseline 的机器指令相同；差异来自旧 LLIR 实验的独立 llc 重编译。统一当前 baseline 后，LLIR/TTIR 的几何平均加速为 1.412×/1.295×。详见 [baseline 一致性核查](baseline_audit.md)。下面各自 baseline 比值不应当作纯优化效果对照。

结论：本轮 TTIR 只做了范围特化、展开和简单算术变换，没有迁移 LLIR 成熟版的寄存器/规约/同步/输出调度优化；同时两轮 baseline 编译结果不同。不是 TTIR 优化能力上限的比较。

## 同 GPU、同计时协议的四版本对照

GPU 7，H100，FP16，M=4096，N=1024..15872 step512；相同输入、预分配输出、随机四版本执行顺序，CUDA graph replay，每点 3 轮、rep=10ms。保留既有二进制，不改变两轮 kernel。计时前四版在全部 30 个 shape 上验证 Y/Mean/Rstd，共 120 次通过。未锁时钟、未控制冷热缓存，结果不等于冷缓存或应用端到端性能。

| 指标 | 算术平均 | 几何平均 |
|---|---:|---:|
| LLIR baseline / LLIR optimized | 1.5235× | 1.4759× |
| TTIR baseline / TTIR optimized | 1.3173× | 1.2947× |
| LLIR baseline / TTIR baseline | 1.0454× | 1.0452× |
| TTIR optimized / LLIR optimized | 1.0961× | 1.0907× |

因此几何平均加速比的差距可以近似分解为：`1.476/1.295 ≈ 1.045 × 1.091`。其中旧 baseline 慢约 4.5%，当前 TTIR 优化版慢约 9.1%。每项取配对比值中位数后汇总，等式存在微小统计偏差。

旧报告的 1.413×/1.400× 使用包含分配的 do_bench；本轮 1.314×/1.293× 使用预分配 CUDA graph。两者不能直接做因果比较。相同额外开销一般会把加速比拉向 1，并不能用“新计时不含分配”解释加速比降低；计时更换同时改变了重复访问和缓存行为。这里的同协议复测排除了该口径差异后，差距仍然存在。

## 主要技术原因

### 1. 当前 TTIR 版的寄存器代价方向相反

| 版本 | registers/thread | spills | CUDA driver 最大 active CTA/SM | 理论 occupancy |
|---|---:|---:|---:|---:|
| LLIR baseline | 60 | 0 | 4 | 50.0% |
| TTIR baseline | 60 | 0 | 4 | 50.0% |
| LLIR optimized | 48 | 0 | 5 | 62.5% |
| TTIR optimized | 80 | 2 | 3 | 37.5% |

CTA/SM 不是仅按寄存器手算，而是对实际 CUDA function、256 threads、32 B dynamic shared 调用 cuOccupancyMaxActiveBlocksPerMultiprocessor 得到。它是理论驻留能力，不代表已经测得的实际 occupancy。

LLIR 版压缩 mean/variance 累加器，并按 8 个 FP16 一组加载、计算、输出，控制值存活期。当前 TTIR 展开后保留更多值，后端给整个单一 kernel 分配 80 个寄存器。即使小 N 分支本身不用全部寄存器，也受同一个 kernel 的资源上限约束。没有做单因素消融，不能把所有耗时差都归因于 occupancy。

反汇编进一步定位：2 条 STL 和 2 条 LDL 仅在 8192<N≤16384 的双 tile 路径内，其他分支没有这些 local-memory 访问。因此不能把 N=3072 等小/中尺寸差距直接归因于 spill；那些分支仍受整体 80-register 分配影响。

### 2. TTIR 保留了后端默认规约/同步，LLIR 直接改了实现

当前 TTIR 每条 N 分支都包含 5 个 CTA barrier，LLIR 成熟版每条执行路径为 2 个。LLIR 使用分布式跨 warp 规约以及 mean/variance 独立 shared 缓冲；当前 TTIR 的 tt.reduce 仍交给后端默认 lowering。本轮主要改变张量宽度和循环，没有实现这些优化。全 binary 的 barrier 静态总数是多条分支之和，不能当成一次 launch 的 barrier 数。

### 3. N 特化粒度较粗

当前 TTIR 用 1024/2048/4096/8192 单 tile 和 ≤16384 双 tile。N=2560、3072 走 4096 宽，N=8704 也完整展开第二个 8192 宽 masked tile，只靠 mask 保证访问合法，范围内无效工作不能全被静态清掉。

LLIR 版除精确 N=1024/4096/8192 路径外，使用 10240/12288/14336 等更细范围、部分展开和更小的累加/输出分组，能消除更多确定无效的组。LLIR 还设置了输出 X 的 .lu 和 Y 的 .cs 缓存策略；当前 TTIR 未迁移。已有 LLIR 历史记录中，最后一轮缓存/缓冲组合平均收益约 1.012×，不足以单独解释全部差距。

### 4. 当前 baseline 更强，分母效应缩小了加速比

两个 baseline 源自同一教程，但 LLIR 走离线 llc/ptxas，TTIR 走 Triton 后端，机器指令哈希不同。旧 baseline 静态 SASS 有 856 条指令，当前为 792 条；当前包含 65 条 FFMA，旧 baseline 没有 FFMA，相关分离 FADD/FMUL 数也更多。这里是静态指令结构证据，不等于动态执行指令数或独立归因。相同计时下，当前 baseline 平均更快约 4.5%，本来就会使相对加速比降低。

## 差距落在哪些尺寸

| N | LLIR opt µs | TTIR opt µs | TTIR/LLIR 优化版耗时 | LLIR 自身加速比 | TTIR 自身加速比 |
|---:|---:|---:|---:|---:|---:|
| 1024 | 11.881 | 12.865 | 1.084× | 2.837× | 2.419× |
| 2048 | 13.296 | 16.642 | 1.252× | 2.603× | 1.916× |
| 2560 | 17.277 | 23.892 | 1.383× | 2.152× | 1.442× |
| 3072 | 21.169 | 29.844 | 1.410× | 1.834× | 1.208× |
| 4096 | 26.627 | 31.367 | 1.178× | 1.522× | 1.214× |
| 8192 | 48.080 | 49.192 | 1.023× | 1.310× | 1.247× |
| 8704 | 51.816 | 59.089 | 1.140× | 1.426× | 1.198× |
| 12288 | 71.867 | 73.080 | 1.017× | 1.268× | 1.209× |
| 15872 | 93.931 | 92.007 | 0.979× | 1.207× | 1.199× |

差距最大的是中等 N 和刚进入双 tile 的区间；N=15872 当前 TTIR 优化版反而略快。大尺寸更受内存供给约束，LLIR 额外减少指令和同步的优势可能被带宽限制削弱。旧 LLIR 的 NCU 数据支持“大 N 带宽/加载依赖突出”，但本次未对 TTIR 新跑 NCU，不能把该解释当作已经完成的逐项性能归因。

## 后续优先级

1. 先在 TTIR 内控制展开后的中间值数量和存活期，减少累加/输出分组规模；验证能否把 80 registers 降下来、消除双 tile spill。不能只硬限 maxnreg，把压力转移到更多 spill。
2. 针对 2560/3072 和 8704/9216 等差距大的尺寸细分范围或进行部分展开，并保留完整矩阵防止偏科。
3. 研究通过 TTIR 的分组/规约表达影响 lowering，检查最终是否真正缩短规约和 barrier 路径。若需要手工 warp/shared 布局，应作为 TTGIR/后端优化另开实验，不暗中混入纯 TTIR 结果。
4. 在同 baseline、同后端、同计时下逐项消融，再比较 TTIR 与 LLIR。两轮成熟度不同：本轮 TTIR 只有 5 个初始候选，LLIR 已包含多轮资源和规约优化，不能据此推断 TTIR 天生更差。

## 原始证据与复现

- `results/llir_comparison/same_protocol.json`：全部 120 次正确性结果、30×3 个四版本计时块、顺序、二进制哈希和资源输出。
- `results/llir_comparison/binary_inspection.json`：CUDA occupancy API 查询、分支级 barrier/local 指令统计。
- `results/llir_comparison/*.sass`：四版实际反汇编。
- LLIR 方法参考 `/root/ir-ops-dev/layer_norm/ir_opt/optimization_methods.md`，此轮不修改旧实验。

```bash
CUDA_VISIBLE_DEVICES=7 /root/venvs/ir-ops-dev/bin/python layer_norm/compare_llir.py \
  --output layer_norm/results/llir_comparison/new_run.json
CUDA_VISIBLE_DEVICES=7 /root/venvs/ir-ops-dev/bin/python layer_norm/inspect_llir_comparison.py
```

本次只增加分析、对照测量和证据，没有修改 baseline、optimized.ttir 或其生成器。
