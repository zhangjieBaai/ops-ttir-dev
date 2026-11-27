# TTIR 模型：能控制和不能控制的东西

改 TTIR 前必须先判断：你想拿的收益，是否可能由 TTIR 这一层承载。判断错了，候选会编译通过、精度通过、性能不变——浪费一整轮。

## TTIR 在流水线中的位置

```
tutorial.py ──make_ttir──▶ TTIR ──make_ttgir──▶ TTGIR ──make_llir──▶ LLIR ──▶ PTX ──▶ cubin
                            ▲                      │
                       你只能改这里          布局/共享内存/barrier 在这里产生
```

关键事实（TTIR_EXPERIMENT_REQUIREMENTS.md §4）：Triton 3.7.0 的**文件入口从 `make_ttgir` 继续，不会再次执行 `make_ttir`**。所以 `triton.compile("x.ttir")` 跳过 Python 前端，直接以你给的 TTIR 为输入。

这里的“能用 TTIR 表达”不表示“只有 TTIR 能表达”。公开 Triton 语法也支持本轮大部分算法变换；逐指令属性、区域/SSA 接口与编译截点的具体差异，见 [TTIR 与 Triton 语法的能力对照](ttir-vs-triton.md)。

## TTIR 里有什么

以 layer_norm baseline 为例（`ir/baseline.ttir`，210 行）：

| 构造 | 例子 | 你在改的就是它 |
|---|---|---|
| `tt.func` + 属性 | `%X: !tt.ptr<f16> {tt.divisibility = 16 : i32}` | 签名契约，**不改** |
| 张量常量 | `arith.constant dense<0.0> : tensor<8192xf16>` | 掩码的 other 值、初值累加器 |
| 索引构造 | `tt.make_range`、`tt.splat`、`tt.addptr` | 地址与掩码计算 |
| 访存 | `tt.load %ptr, %mask, %other`、`tt.store` | 宽度、掩码、`cacheModifier` |
| 循环 | `scf.for %off = ... iter_args(...)` | 展开、分组、tile 宽度 |
| 分支 | `scf.if %fits { ... } else { ... }` | 按 N 特化的分派 |
| 算术 | `arith.addf/mulf/divf`、`math.sqrt/rsqrt` | 表达式形式、倒数复用、FMA 显式化 |
| 规约 | `"tt.reduce"(%t) <{axis=0}> ({...})` | 规约的**表达方式**（分组/折叠） |
| 形状变换 | `tt.reshape %t : tensor<8192xf32> -> tensor<4x2048xf32>` | 折叠累加器 |

## 表达力边界（最重要的一节）

### TTIR 能控制的

**1. 张量宽度 → 影响每线程元素数与寄存器压力**

这是 TTIR 最强的一根杠杆，也是最容易被忽视的映射关系：

```
平均逻辑元素数 ≈ 张量宽度 / (num_warps × 32)
```

对 NVIDIA 上均匀分布的一维 blocked layout，这是估算值。实际映射要看 TTGIR 的 `sizePerThread`、`threadsPerWarp`、`warpsPerCTA` 与复制；布局变换、标量、地址、重排、存活重叠和 spill 都会影响最终资源。它不是寄存器分配公式。

layer_norm：`BLOCK_SIZE=8192`、8 warps = 256 线程 → 平均每线程 32 个逻辑元素。一个同时存活、未被折叠的 FP32 累加器可能因此需要约 32 个寄存器，但**不能据此断言 kernel 的寄存器数**。

把宽度换成 1024 → 平均每线程 4 个元素。这是减少逻辑累加状态的一种 TTIR 表达；是否达到 LLIR 方案的资源和性能效果，需要编译与计时确认。

资源事实为C级，来源为`results/round2/compile_v2.json`与`resources.json`；各候选性能另看证据索引：

| 候选 | 张量宽度 | 每线程元素 | 寄存器 | spill |
|---|---:|---:|---:|---:|
| baseline | 8192 | 32 | 60 | 0 |
| r2_loop1024 | 1024 | 4 | **32** | 0 |
| r2_loop2048 | 2048 | 8 | **32** | 0 |
| r2_fine1024_reload | 1024 | 4 | **40** | 0 |
| tile_dispatch_math_v2 | 分支混合 | 最多 32 | **80** | **2** |
| r2_mixed_stream | 1024/2048分组 | 每组最多8（非总存活量） | **48** | **0** |

**2. 循环结构与展开**

`scf.for` 可以换成展开的直线代码，或用 `scf.if` 按运行时尺寸分派。layer_norm 的 `tile_dispatch` 就是三层嵌套 `scf.if`（1024/2048/4096/8192）+ 双 tile 展开。

**3. 算术表达式形式**

- 倒数复用：`divf %mean, %N` + `divf %var, %N` → 共用一个 `divf 1.0, %N` 再两次 `mulf`
- `math.sqrt` + `arith.divf` → `math.rsqrt`
- 显式 `arith.mulf`/`arith.subf` 让后端能融成 FMA（也可显式构造，但注意下文的 negf 陷阱）

**4. 加载的缓存属性**

`tt.load` 可带 `cacheModifier`。r2 实验用 `cacheModifier=ca` 做了对照（`r2_fine2048_ca`，B 级 1.4901×，本库最高初筛值）。

**5. 规约的表达方式**

`tt.reduce` 的**输入形状**可以改。`r2_fold2048` 用 `tt.reshape ... -> tensor<4x2048xf32>` 再 `tt.reduce axis=0`，把 8192 宽加载折叠成 2048 宽累加器——**加载宽、累加窄**。（B 级 1.2493×，不如直接窄加载，但提供了第三条路。）

### 直接控制、间接影响与本轮限制

| 目标 | 普通 TTIR 的边界 | 验证办法 |
|---|---|---|
| 指定 warp 内 shuffle 规约 | `tt.reduce` 不直接指定 shuffle 指令 | 检查 TTGIR/LLVM/PTX；形状变化也可能改变 lowering |
| 显式分配/划分 shared memory | 本轮普通 `tt.*`/`scf.*` 改写不直接指定 GPU shared 布局 | 检查 TTGIR 布局与 shared metadata |
| 精确增删某个 barrier | 普通 TTIR 不逐条指定 lowering 插入的 barrier | 改算法/规约形状可能间接改变数量，须查产物 |
| 控制物理寄存器分配 | 后端和 ptxas 决定 | 改存活值后检查实际 registers/spill |
| 内联 PTX | **支持** `tt.elementwise_inline_asm`；但 LayerNorm 第二轮契约明确禁用 | 独立探针已编译、执行；不能混入本轮候选 |

**边界结论（C 级）**：普通 TTIR 不能逐条指定下游同步指令，但改变算法、规约形状或控制流可能间接减少同步。LLIR LayerNorm 的 5→2 barrier 尚未在本轮 TTIR 候选复现；这不构成“TTIR 永远做不到”的证明。内联汇编会引入 ISA、布局与通信语义约束，能力存在也不等于跨后端可移植。

官方 op 定义见 [Triton v3.7.0 TritonOps.td](https://github.com/triton-lang/triton/blob/v3.7.0/include/triton/Dialect/Triton/IR/TritonOps.td)。复现实验与完整哈希见 [证据索引](evidence-index.md#kernelforge-知识核查)。原“文件入口不接受内联 PTX”结论已被该探针反证。

## 三个非直觉的机制

### 1. 分派分支共享同一份资源预算

**这是 TTIR 特化最容易踩的坑。**

本轮常规 CUDA kernel 的 launch 使用统一的每线程寄存器资源额度。各分支的存活区间和寄存器复用由全局分配决定，**不能精确等同于各分支单独编译寄存器数的最大值**。`llir_comparison.md:35` 原文：

> 即使小 N 分支本身不用全部寄存器，也受同一个 kernel 的资源上限约束。

证据：`tile_dispatch_math_v2` 有 1024/2048/4096/8192 窄分支，但整体仍是 80 regs / 2 spills；单独窄化的 `r2_loop1024` 只有 32 regs。宽分支可能抬高整体额度；要量化某个分支的贡献，仍需移除/窄化该分支的受控消融。

资源档位后果（`llir_comparison.md`）：

| 版本 | 寄存器 | spill | 理论 CTA/SM | 理论 occupancy |
|---|---:|---:|---:|---:|
| TTIR baseline | 60 | 0 | 4 | 50.0% |
| LLIR optimized | 48 | 0 | **5** | **62.5%** |
| TTIR optimized（v1） | 80 | 2 | **3** | **37.5%** |

**特化版理论 occupancy 从 50% 降到 37.5%，仍观测到约 1.29×。** 这说明 occupancy 不能单独预测性能；减少无效工作是机制解释，尚未把各项贡献完整消融。

**操作含义：** 加特化分支时，先问"最肥的分支会把整个 kernel 拉到哪个档位"。如果会跨档下跌，考虑把最肥的分支也窄化，或者干脆不做宽分支。

### 2. 本例去 loc 未改变机器指令，但改变调试信息

`canonical` 候选只删 `#loc`/`loc(...)`，文件从 210 行降到 93 行。结果：**ELF `.text` 哈希与 baseline 完全相同**（C 级静态证据），6 点初筛计时约 1.000×（B 级；不是完整矩阵或独立复测）。

含义：
- `loc` 只带调试位置，不参与计算
- 删掉它可让 TTIR 更易读、可 diff；保留冻结原件与映射，也可只对比较副本去 loc
- 但它本身零收益，不要算进优化成绩

注意：`canonical` 的**完整 cubin 哈希不同**（调试信息差异），只有 `.text` 相同。报告时必须写清是"`.text` 一致"而非"cubin 字节一致"（要求见 `TTIR_EXPERIMENT_REQUIREMENTS.md` §5）。

### 3. 窄 tile 不是免费午餐

窄化减少寄存器，但增加循环迭代次数。代价因 N 而异：

| 候选 | 宽度 | 初筛 geo | 最差 shape | 寄存器 |
|---|---:|---:|---:|---:|
| r2_loop1024 | 1024 | 1.2570 | **0.7836** | 32 |
| r2_loop2048 | 2048 | 1.2233 | **0.7508** | 32 |
| r2_fine1024_reload | 1024 | 1.4060 | 1.0124 | 40 |
| r2_fine2048_reload | 2048 | 1.4445 | 1.1076 | 48 |

`r2_loop*` 在本次大 N 初筛上回退到 0.75–0.78×；`r2_fine*_reload` 对 baseline 的初筛中位数均大于 1。但两族同时改变分派、展开和 load 行为，不能把差异全部归因于分派，也不能推断它们在所有 shape 上优于 parent。

**经验（B 级）**：窄化要检查大 N 的循环代价；范围分派是一条候选路线，不是所有算子的必要条件。

## 第一轮与 LLIR 的同协议对照

以下是第一轮历史对照（B 级），不是当前 `optimized.ttir` 的结果。第二轮已晋升 `r2_mixed_stream`，新结果见 [证据索引](evidence-index.md#layernorm-第二轮)。同协议下（`llir_comparison.md`，30 shape、3 轮、rep=10ms、CUDA graph）：

| 指标 | 算术 | 几何 |
|---|---:|---:|
| LLIR baseline / LLIR optimized | 1.5235× | 1.4759× |
| TTIR baseline / TTIR optimized | 1.3173× | 1.2947× |
| LLIR baseline / TTIR baseline | 1.0454× | 1.0452× |
| TTIR optimized / LLIR optimized | 1.0961× | 1.0907× |

分解：`1.476 / 1.295 ≈ 1.045 × 1.091`。

两部分：
1. **两个 baseline 的机器码不同**（1.045×）。同为 Triton 3.7 导出的 LLVM IR，但旧 LLIR 走独立 `llc`，TTIR 走 Triton 内置后端。`.text` 哈希不同（`baseline_audit.md`）。
2. 这组样本中 `TTIR opt / LLIR opt = 1.0907`，即 TTIR 版耗时约高 9.1%（B 级同协议观测）。寄存器/规约/同步/调度差异是待消融的解释，不能据此给 TTIR 的表达力定上限。

**规矩：** 跨层比较必须用同一 baseline 重算，且注明后端差异。不能拿 1.476× 对 1.295× 直接下结论。

## 迁移到 MM 时重新建立映射

MM 的冻结 TTIR 含 `tt.dot`：`tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>`。它能表达分块计算和逻辑累加形状，但具体 Tensor Core 指令、operand layout、流水与 shared 资源仍要查下游产物。不要直接套用 LayerNorm 的一维“宽度/线程数”估算，更不能由 `inputPrecision=tf32` 的字面属性断言 FP16 输入走了 TF32 算术。

当前 MM 配置来自代表点 autotune 后冻结，warps=4、stages=4；本库只有其基线验证，没有优化效果（C 级静态事实及数值/性能记录见 [MM 证据](evidence-index.md#mm-基线证据)）。新的 dtype、dot 精度或编译 options 必须另建契约与匹配 baseline。LayerNorm 的收益与瓶颈排序不能直接迁移。

## 快速判断清单

拿到一个想做的优化，逐条问：

1. 它要求直接指定 warp 通信、shared 布局或 barrier 吗？→ 先分清显式控制与算法改写的间接影响，再检查契约是否允许 ISA 相关手段
2. 它能表达成"改张量宽度"吗？→ 可作为候选；检查布局、活跃状态、循环与分派成本
3. 它会加特化分支吗？→ 会，检查编译后的整体资源档位，必要时消融宽分支
4. 它只改算术表达式吗？→ 先验证误差与特殊值；LayerNorm 单项初筛约 1% 不能外推到其他算子
5. 它只在某个 N 区间有效吗？→ 必须证明不伤其他区间，否则需要分派


## 第二轮已验证的组合与反例

最终组合通过完整矩阵与独立复测（A，几何约1.440×），资源从parent的80寄存器/2 spill降到48/0（C）。但低到32寄存器的动态窄循环在大N初筛回退，achieved occupancy约55%的最终版在最大N的NCU计时也仍慢于parent。不要把逻辑状态、资源指标、机制解释和计时结论混为一件事。

组合的有效经验和未采用方向见[第二轮复盘](layernorm-round2.md)；TTIR `.ca`属性保留额外load是当前编译器/程序中的观察，不是通用禁止CSE语义。该组合也没有实现LLIR的两次barrier路径，不能由平均耗时接近推断指令实现相同。
