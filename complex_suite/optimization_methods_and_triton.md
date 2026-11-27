**后续实测更新：源码回写对照已完成。五类方案在本轮配置下均复现了优化TTIR的可执行指令与性能，详见[源码与TTIR三路实测报告](source_port_comparison.md)。下文“尚未验证”的表述记录的是回写实验开始前的静态核查状态。**

本报告回答两个问题：上一轮五类算子具体优化了什么，以及这些方法能否用普通 Triton 源码表达。这里的 Triton 指官方 **3.7.0** 的 `@triton.jit` 和公开 `triton.language` API，不包括修改编译器、内部 IR builder、Gluon 或厂商扩展。

**结论：五类最终采用方案的算法、类型和数据流变换，都有公开 Triton 源码表达方式；本轮没有证据表明任何一个成功方案只能在 TTIR 中实现。** Attention 的操作次序也能在源码中调整，但其性能尤其依赖编译器是否保留相应结构和调度。我们实测的是直接修改 TTIR 后的结果，尚未把这些方案完整回写成源码候选并验证相同收益。

因此，需要分别看三件事：

| 判断层次 | 本轮能给出的结论 |
|---|---|
| 优化思路能否用 Triton 表达 | 可以；依据冻结候选和本地 3.7.0 公开 API 的静态核查，属于 C 级能力判断 |
| 源码是否生成与手改 TTIR 相同的中间结构或机器码 | 未验证；不能由 API 存在直接推出 |
| 源码版本是否复现已测加速比 | 未验证；表中性能全部属于 TTIR 版本 |

下文代码都是**结构说明片段**，省略完整签名、地址构造或部分边界处理；没有作为源码优化候选编译、计时，不是可直接替换生产 kernel 的交付实现。

**实测结果与方法对应如下。** 加速比为各 shape 的配对时间比中位数，再对冻结矩阵取几何平均；A 级只表示该矩阵有完整主测和独立进程复测，不表示已经证明单项机制的全部因果贡献。

| 算子及最终版本 | 实际优化内容 | 主测 / 独立复测 | Triton 源码表达 |
|---|---|---:|---|
| INT4 反量化＋GEMM，`scalar_unsigned_v1`，7点 | low/high 共享 scale/zero；tile 内 group 标量化；非负索引除余转 unsigned | 1.0948× / **1.0958×** | 共享变量、标量索引与广播、`tl.uint32` 和 `//`、`%` |
| Grouped/biased MoE，`membership`，8点 | 将 `[E,G]` 专家/组关系矩阵及归约，换成 group mask 广播与 reshape | 1.1502× / **1.1502×** | `tl.broadcast_to`、`tl.reshape` |
| Residual RMSNorm＋INT8，`stream_v1`，8点 | amax 因式分解；sum/max 联合归约；输出 `.cs` cache hint | 1.0548× / **1.0521×** | 普通表达式、`tl.reduce` 自定义组合函数、`tl.store(cache_modifier=".cs")` |
| Attention dK/dV，`r1_dv_late`，D64六点 | 将概率转 FP16 和 dV dot 移到 dK dot 之后 | 1.1241× / **1.1214×** | 调整 `.to(tl.float16)` 和 `tl.dot` 的源码次序；下游保留情况未验证 |
| 大 N partial Top-K，`ordered_carry_v3`，6点 | 有序 uint32 key 代替浮点选择状态；编码移出循环；循环传递并更新 key | 1.0827× / **1.0816×** | `bitcast`、位运算、uint32 `tl.max`、普通 `for` 循环变量 |

Attention 共测12点，D128未获得稳定收益，保留 baseline；不能将表中 D64 数值当作全部12点的收益。Norm 最差点复测为 **0.9859×**，约1.4%回退，也保留在报告中。各任务测量边界不同，不计算跨算子总加速比。数值来源分别见 [INT4](../int4_gemm/results/analysis.json)、[MoE](../moe_routing/selection.json)、[Norm](../norm_quant/selection.json)、[Attention](../attention_dkdv/selection.json) 和 [Top-K](../partial_topk/selection.json)。

**INT4：优化的是反量化过程中的重复数据流和索引计算。**

基线已经把 packed uint8 读取、两个 INT4 解包、groupwise 反量化和矩阵乘融合在一个 kernel 中。这次不是通过“先独立反量化，再 GEMM”换成融合来制造收益。

原实现分别为 low/high nibble 计算 K 索引、group、scale 和 zero。实际合同中 group=128，同一个 byte 对应的偶/奇 K 总在同一 group，因此候选删除了 high 分支的重复 group/scale/zero 表达式，直接使用 low 分支的值。进一步利用 `BLOCK_K=64 或128`、循环起点对齐、K 为128倍数，将 tile 内逐元素 group 除法改成一次标量 `k0/group_size` 后广播。最后对已证明非负、分母为正且不溢出的索引除余使用 unsigned 语义。

用 Triton 可以表达同样的结构，例如在完整 K tile 的前提下：

```python
# 位于 @triton.jit kernel 的 K 循环中；沿用原 N 尾部 mask。
group = k0.to(tl.uint32) // qGroupSize.to(tl.uint32)
scale = tl.load(S + group * stride_sg + offs_n * stride_sn,
                mask=offs_n < N, other=1.0).to(tl.float32)
zero = tl.load(Z + group * stride_zg + offs_n * stride_zn,
               mask=offs_n < N, other=0.0).to(tl.float32)

low = ((packed & 0xF).to(tl.float32) - zero[None, :]) * scale[None, :]
high = ((packed >> 4).to(tl.float32) - zero[None, :]) * scale[None, :]
low = low.to(tl.float16)
high = high.to(tl.float16)
# 沿用原 join/trans/reshape/dot 及其累积顺序。
```

其他 program/tile 索引也可在有值域证明后改成 `tl.uint32`；这不等于可以对任意带负索引、负 stride 或可能溢出的表达式机械替换类型。N/M 尾部掩码仍需保留，不能把本轮完整 K tile 的条件推广到任意 K。

这项优化在 TTIR 上删除了重复加载表达式，但不能仅凭 IR 行数断言 DRAM 读取字节减半；实际访存还受编译器合并和缓存影响。另一个试验将 FP16 cast 从 join/reshape 前移到后面，7点的可执行 `.text` 与基线相同，没有收益。这说明源码/IR 形式不同，不一定改变最终代码。[候选生成器](../int4_gemm/make_candidates.py) 和 [完整报告](../int4_gemm/report_round1.md) 保存了变换、前提和负结果。

**源码可表达性结论：可以，不需要 TTIR 独有操作。** 是否生成相同布局和相同加速比，需要独立源码复现；上述更紧凑的示意加载形状也不保证与本轮手改 TTIR 的形状完全相同。

**MoE：优化的是 group mask 向 expert mask 的传播。**

本轮完整 routing 从 logits 开始，包含 tanh 表达式实现的 sigmoid、带 bias 的 top2 group score、选组、expert Top-K、未加 bias 的原始权重提取、归一化及 weights/indices 输出。它不包含 router 线性层、token dispatch 或 expert 计算。

基线为每个 expert 与每个 group 构造关系，形成 `[E,G]` 布尔中间矩阵，再归约得到每个 expert 是否属于已选组。合同中 G=8，专家按连续、等长组排列，所以这段计算等价于把每个组的布尔状态重复 `E/G` 次：

```python
# group_selected: [G] bool；专家连续、等长分组。
expert_selected = tl.broadcast_to(
    group_selected[:, None], (G, E // G)
).reshape((E,))
selection_scores = tl.where(expert_selected, biased_scores, -float("inf"))
# 后续选择、原始分数加载和归一化保持原合同。
```

候选保持两个 kernel 和工作区接口不变，只替换第二阶段的 membership 数据流。E256/E512 的第二阶段 shared memory 分别由1024/2048B降到16B，寄存器从32降到25/28，属于 C 级资源事实；不能把全部性能收益分摊给其中某一个数字。

同时试过把 expert 的 max 和 tie-index min 合并为 tuple argmax。它能用 `tl.reduce((scores, ids), ..., combine_fn)` 表达，但完整主测几何平均只有约0.9975×，大 batch 约0.909×。尽管静态 barrier 数下降，整体仍可能变慢。它与 membership 的组合也没有超过单独 membership。[候选生成器](../moe_routing/make_candidates.py) 和 [完整报告](../moe_routing/report_round1.md) 保留了证据。

**源码可表达性结论：可以，公开 broadcast/reshape 足够。** 这里读取选中原始权重的是索引 `tl.load`，不是依赖 `tt.gather` 的隐藏布局属性；本轮也没有用 `efficient_layout`。

**Norm＋动态 INT8：优化的是两个统计阶段的依赖和输出方式。**

定义 `z=x+residual`、`u=z*w`、`rho=rsqrt(mean(z²)+eps)`、`y=rho*u`。基线先归约 sum(z²)，算出 rho 与 y，然后再对 y 做 amax。候选利用有限值、rho>0 下的数学关系：

\[
\max_i |\rho u_i|=\rho\max_i |u_i|.
\]

这样 sum(z²) 和 max(abs(u)) 可以从同一批输入分别产生，不必让第二个统计量的数据依赖于 rho。随后将两个统计量写成一个多输出 reduction，最后合成量化 scale。最终版本还对输出使用 `.cs` cache hint。

对应的公开 Triton 写法是：

```python
@triton.jit
def add_max(sum_l, max_l, sum_r, max_r):
    return sum_l + sum_r, tl.maximum(max_l, max_r)


# 位于 kernel 内；z/u 的 padding lane 为0。
sum_z2, max_u = tl.reduce((z * z, tl.abs(u)), axis=0, combine_fn=add_max)
rho = tl.rsqrt(sum_z2 / N + 1.0e-5)
amax = rho * max_u
scale = tl.where(amax > 0, amax / 127.0, 1.0)
y = u * rho
q_float = tl.div_rn(y, scale)
# 沿用 clamp、nearest-even 舍入、INT8 cast；得到 q。
tl.store(Z_ptr + offsets, z, mask=mask, cache_modifier=".cs")
tl.store(Q_ptr + offsets, q, mask=mask, cache_modifier=".cs")
tl.store(S_ptr + row, scale, cache_modifier=".cs")
```

最终 `stream_v1` **没有**采用在量化比值中消去 rho 的 `joint_cancel_v1`；它保持 `y/scale` 的计算路径，只重新组织统计量并添加 cache hint。不能把未采用的候选写成最终实现。

上述恒等式不是严格的浮点逐位恒等式。合同在实验前允许 Q 相对 FP64 参考最多差1个量化码，另检查 scale、重构误差、Z 精确值和 Q 范围。因此这是满足预定容差合同的优化，不能宣称与基线的 INT8 输出逐位等价。`.cs` 是公开 API 参数，但其实际效果具有后端和架构相关性，不保证其他硬件同样获益。[候选生成器](../norm_quant/make_candidates.py) 和 [完整报告](../norm_quant/report_round1.md) 提供了组合及消融记录。

**源码可表达性结论：可以。** tuple reduction 在公开 `tl.reduce` 中直接受支持，不需要手工构造私有 reduction region。减少统计依赖是结构事实；减少同步、降低寄存器或更快执行仍需检查实际编译产物。

**Attention dK/dV：优化的是循环内独立消费路径的先后次序。**

基线是官方 fused-attention backward 的 dK/dV 段提取，保留 QK 分数重算、概率恢复、dP/dS、dK/dV 四个 dot 与共同循环累积。候选将 `pT→FP16` 的 cast 和 dV dot 从 dP/dS/dK 之前移到 dK dot 之后；因果模式的两段循环都作相同调整。

源码可以直接写出这种次序：

```python
# pT 已恢复并应用 causal mask；qT、do、v、Di 的语义沿用基线。
dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
dsT = (pT * (dpT - Di[None, :])).to(tl.float16)
dk += tl.dot(dsT, tl.trans(qT))

ppT = pT.to(tl.float16)
dv += tl.dot(ppT, do)
```

这不改变每个 accumulator 自身的累加次序，也不改变其 dot 输入的舍入点。本轮 TTIR 候选与基线输出逐位相同。变换改变了中间值、转换和 dot 的消费时序，不能笼统称为“所有值的生命周期都缩短了”。

D64 六点主/复测约1.1241×/1.1214×。D128非因果整体接近1×，因果约0.937×，因此保留基线；其中因果 spill 从182降到160仍然变慢。这也是资源改善不能单独证明加速的反例。[候选生成器](../attention_dkdv/make_candidates.py) 和 [完整报告](../attention_dkdv/report_round1.md) 记录了真实范围。

**源码可表达性结论：能表达预期计算次序；尚未验证源码路径是否产生同样的有利编译结果。** 两个独立计算在 Python 文件里的前后位置，不构成最终 GPU 调度的强制约束。直接改 TTIR 的优势是精确指定进入后续 lowering 的实验输入，而不是保证物理指令顺序；后续 passes 仍可重排。这里不能把“从源码复现尚未验证”写成“源码无法实现”。

性能只包括已给定 LSE/delta、预缩放 K 后的 dK/dV 子算子，不含 forward、这些准备过程、dQ 或完整训练 step。固定官方配置未逐 shape autotune。

**大 N partial Top-K：优化的是选择状态的表示，而不只是少做一次归约。**

基线已经采用两阶段 partial selection：每1024个元素保留K个候选，再从候选中合并全局 Top-K。候选没有改变 kernel 数量、chunk 大小、工作区或启动配置。

这条路线经历了三个有区分度的结果：

| 变换 | 观测 | 解释边界 |
|---|---:|---|
| max＋min-index 改为联合 value/index argmax | 完整主测约0.810×，B级 | 联合归约更少不代表后端更高效 |
| 浮点 score 转有序 uint32，仍在每轮编码 | 完整主测约0.980×，B级 | 整数 max 的潜在收益会被重复编码等代价抵消 |
| 循环外编码一次，循环中持有和更新 uint32 key | 最终复测1.0816×，A级 | 同时改变循环状态表示和每轮工作量；不能只归因于一条指令 |

对有限 FP32，可把符号/位型映射为按数值单调递增的 uint32 key；先将 ±0 归一化，确保零值 ties 仍按最小原始 index 选择。循环里对 key 求 max，再对相同 key 的原始 index 求 min。选中位置的 key 改为 `0x007fffff`，即 -Inf 的编码，低于所有有限输入的 key。

这些操作都能使用公开 Triton 语法：

```python
# scores: FP32；ids: 原始全局INT32 index；padding为-inf/INT_MAX。
scores = tl.where(scores == 0.0, 0.0, scores)
bits = scores.to(tl.uint32, bitcast=True)
negative = (bits & 0x80000000) != 0
flip = tl.where(negative, 0xFFFFFFFF, 0x80000000).to(tl.uint32)
keys = bits ^ flip                       # 在循环外编码一次

for step in range(K):
    best_key = tl.max(keys, axis=0)
    best_id = tl.min(tl.where(keys == best_key, ids, 2147483647), axis=0)
    undo = tl.where((best_key & 0x80000000) != 0,
                    0x80000000, 0xFFFFFFFF).to(tl.uint32)
    best_value = (best_key ^ undo).to(tl.float32, bitcast=True)
    tl.store(out_values + step, best_value)
    tl.store(out_indices + step, best_id)
    selected = ids == best_id
    keys = tl.where(selected, tl.full((), 0x007FFFFF, tl.uint32), keys)
    ids = tl.where(selected, 2147483647, ids)
```

这个循环从进入到退出都持有 uint32 key，符合常规 Triton 循环变量的类型要求；不需要 before/after 类型不一致的特殊 `scf.while` 接口。示意里的 `where` 编码与实际 TTIR 的位移/XOR 编码在合同内等价，不代表两者导出的 IR 已做过逐字节比对。

实际候选 PTX 出现 `redux.sync.max.u32`，这是 C 级静态证据。源码写 `tl.max(uint32)` 能表达 unsigned max 语义，却不能在所有配置和后端上保证一定选用该条 NVIDIA 指令。本轮没有用内联 PTX 强行指定它。

合同为值数值精确、indices严格精确，**±0等价，不要求保留零符号位**；覆盖尾块不足K、ties、subnormal和最低有限值，不覆盖 NaN/Inf。源码表达还必须保留这些边界，不能仅用随机正数证明等价。[有序 key 生成器](../partial_topk/make_ordered.py)、[循环状态生成器](../partial_topk/make_carry.py) 和 [完整报告](../partial_topk/report_round1.md) 保存了证据。

**源码可表达性结论：可以。** 改变 loop-carried SSA 类型本身不意味着超出源码能力；这里可以通过在循环前初始化 uint32 变量、循环中保持该类型来表达。

**本轮 TTIR 的价值，主要体现在观察和控制编译输入。**

TTIR 直接展示经过前端处理后的广播、cast、加载、归约、dot 和循环参数，让人能定位重复数据流、检查实际中间类型，并将某一处结构改动作为确定的输入交给后续 lowering。这个实验入口避免了“先改源码，结果在前端又被规范化成另一种形态”带来的部分不确定性，但不绕过全部后端优化。

本轮没有使用逐指令 fast-math、不溢出属性、`efficient_layout`、不对称区域接口或手工修改 TTGIR/LLVM/PTX/cubin 来取得最终收益。因而目前支持的表述是：**TTIR 帮助实现并验证了若干可解释的结构优化；这些结构有源码表达方式。** 本轮没有 A/B/C 盲化搜索对照，不能证明看 TTIR 比只看源码更容易找到方案，也不能把已有加速比解释为“TTIR 相对 Triton 的独占收益”。

若后续要确认源码复现，应为每个最终候选建立单独源码实验组，在相同输入合同、tile/warp/stage、ABI和计时边界下，先核对源码导出的 TTIR 是否保留目标变换，再检查机器码、数值和配对性能。Attention 应优先检查操作次序是否保留；Top-K 应检查循环状态是否仍为有序 uint32；Norm 必须沿用已声明的量化容差。该验证尚未在本报告中执行。

**本报告的证据口径与核查入口如下。**

性能沿用上一轮官方 Triton3.7.0、H100 的冻结矩阵：主测7轮、独立进程复测5轮、每轮20ms、预分配 CUDA graph 设备时间。源码基线与 TTIR 文件入口的 LLVM/PTX/cubin 一致性、数值、memcheck/racecheck 已有记录；这些检查证明被测 TTIR 实验链路，并不证明本文源码示意已通过同样验证。INT4/MoE 做过有限配置调优，Norm 是新融合研究基线，Top-K/Attention 的配置搜索范围有限，均不代表已战胜最强现有实现。

| 源码能力判断 | 本地官方3.7.0的核查位置 |
|---|---|
| `uint32` 除余生成 unsigned 运算 | [semantic.py](/root/venvs/ir-ops-dev/lib/python3.12/site-packages/triton/language/semantic.py:316) 中的 `floordiv`、`mod` |
| `broadcast_to`、reshape | [core.py](/root/venvs/ir-ops-dev/lib/python3.12/site-packages/triton/language/core.py:1722)、[reshape](/root/venvs/ir-ops-dev/lib/python3.12/site-packages/triton/language/core.py:1937) |
| 多输入/多输出 reduction | [core.py 的 reduce](/root/venvs/ir-ops-dev/lib/python3.12/site-packages/triton/language/core.py:2662)，接受 tuple 输入及 `@triton.jit` combine 函数 |
| `.cs` 输出 cache hint | [core.py 的 store](/root/venvs/ir-ops-dev/lib/python3.12/site-packages/triton/language/core.py:2223) |
| bitcast、公开类型转换 | [core.py 的 tensor.to](/root/venvs/ir-ops-dev/lib/python3.12/site-packages/triton/language/core.py:1120)、[cast](/root/venvs/ir-ops-dev/lib/python3.12/site-packages/triton/language/core.py:1997) |
| 常规 for 循环与循环传值 | [code_generator.py 的 visit_For](/root/venvs/ir-ops-dev/lib/python3.12/site-packages/triton/compiler/code_generator.py:1161) |

以上以本地固定版本源码为依据，没有将未编译示意当作编译探针或性能证据；所核查源码及候选生成器的哈希见 [静态核查记录](triton_expression_audit.json)。完整环境、结果、负候选和复现入口见 [实验总入口](README.md)、[性能汇总](report.md)；表达力边界另见 [TTIR 与 Triton 语法对照](../knowledge/ttir-vs-triton.md)。本次新增的是方法报告与静态核查记录，没有改动冻结 baseline、优化候选或历史性能样本。
