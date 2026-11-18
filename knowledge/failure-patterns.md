# 失败模式与陷阱

每条区分实测症状、原始诊断、人工登记与机制假设。**证伪的路线保留**，并限制到已测条件；未做过的对照不写成失败。

---

## F1. 分支内张量宽度与外部零常量不匹配

**症状**：C0 阶段编译失败，kernel 从未执行。

**人工失败登记**（`results/failed_candidates.json`，不是编译器原始诊断）：

```
Branch-local tensor width changed without resizing outer tensor zero constants:
tensor<1024xf16> vs tensor<8192xf16>. Never executed.
```

**根因**：`dispatch()` 把外层 `%cst`（`tensor<8192xf16>`）的宽度改成 1024，但分支里的 `tt.load` other 值仍引用外层那个 8192 宽的常量。

**修复**：在每个分支**内部**重新声明同宽度的零常量：

```mlir
%tile_zero16 = arith.constant dense<0.0> : tensor<1024xf16>
%tile_zero32 = arith.constant dense<0.0> : tensor<1024xf32>
```

并用 `TOKEN.sub` 把分支内对 `%cst`/`%cst_0` 的引用重定向到分支局部常量。

**涉及候选**：`tile_dispatch`；`tile_dispatch_math` 登记为相同缺陷，但未单独编译或执行。修复后为 `*_v2`。区分静态发现、实际编译失败与设备执行失败。

**教训**：改张量宽度时，**所有**引用该宽度的常量都要同步改，且作用域要对——分支局部而不是全局。

---

## F2. `arith.negf` 未通过文件入口 lowering

**症状**：`PassManager::run failed`。

**涉及候选**：`r2_affine`、`r2_delay_affine`（`results/round2/compile_v2.json`）。

**定位（C 级）**：解析成功，`ConvertTritonToTritonGPU` 未能合法化张量 `arith.negf`。`compile_v2.log` 的确切诊断为 `failed to legalize operation 'arith.negf' that was explicitly marked illegal`。独立 `tensor<256xf32>` 探针复现；没有据此判定标量、其他入口、后端或版本一律不支持。

**修复**：改用 `arith.subf` 表达 **`-0.0 - x`**（`make_round2.py:affine_v2()`）。不要写反操作数，也不要换成 `+0.0 - x`：后者对 `x=+0.0` 的结果符号不同。独立探针检查了正负零、有限值、Inf 和 NaN；NaN 仅作 equal-NaN 比较，没有证明 payload、sNaN 或所有舍入模式等价。

**结果**：修复版 `r2_affine_v2` 编译通过，初筛 1.3689。

**教训**：TTIR **不是完整 MLIR**——有些合法 MLIR op 这个入口不认。写新 op 前先在最小候选上试编译。

---

## F3. volatile load 造成 long-scoreboard 爆炸

**症状**：正确性通过，小 N 明显改善，**大 N 严重回退**。

**数据**（`optimization_plan_v2.md:34`，NCU）：

| 指标 | 重载版 | parent |
|---|---:|---:|
| 实际 occupancy | 65%–69% | 33%–35% |
| long-scoreboard / issue | **18.90** | 5.67 |

occupancy 翻倍了，但 N=15872 仍更慢。

**机制假设（D 级）**：volatile 限制某些编译器复用和重排，可能改变缓存访问与延迟隐藏；已观察到 `ld.volatile.global`。不能据此断言每次走 DRAM或无法隐藏延迟。PTX 对 volatile 的语义并不是绕过所有缓存，且它不是线程同步原语。

**后续验证**：`r2_fine2048_ca` 改用 `.ca` 普通 load，并得到更高的独立初筛值；尚缺同会话机制消融，不能将全部变化归因于 volatile。`.ca` 只给缓存提示，不保证所有阶段禁止 CSE，须查实际产物。

**教训**：约束编译器优化前先确定真实语义需求；用编译产物和同会话计时验证代价。occupancy 上升不等于更快，见 F4。

---

## F4. occupancy 上升但性能下降

**症状**：寄存器降低、occupancy 提高，但计时更慢。

**证据**：
- F3 的 occupancy 33%→65%，大 N 仍慢
- `llir_comparison.md`：TTIR optimized 的 occupancy 37.5% 低于 baseline 的 50%，却仍有 1.29× 收益

**解释边界**：理论 occupancy 是资源允许的驻留上限，achieved occupancy 是采集到的活跃 warp 比例，两者都不等于 eligible warps。访存、依赖、同步或循环开销可能主导性能，具体原因还需诊断。

**正确读法**：occupancy 是**解释变量**，不是**目标函数**。必须配合 stall 分布一起看（见 [diagnosis](diagnosis.md)）。

**操作含义**：不要用"寄存器下降"或"occupancy 上升"当作收益。必须实测计时。

---

## F5. 单点窄 tile 在大 N 崩

**症状**：小 N 大幅改善，大 N 回退到 0.75×。

**数据**（`results/round2/screen_batch1.json`）：

| 候选 | 初筛 geo | 最差 shape |
|---|---:|---:|
| r2_loop1024 | 1.2570 | **0.7836** |
| r2_loop2048 | 1.2233 | **0.7508** |
| r2_fine1024_reload | 1.4060 | 1.0124 |

**机制假设（D 级）**：窄 tile 增加循环迭代次数，可能抵消寄存器收益；当前比较还改变其他 lowering 行为，没有单项消融。

**可试路线**：范围分派（[techniques T2](techniques.md#t2-按尺寸分派-tile-宽度最强杠杆)），对每个 N 区间验证合适宽度；不是所有算子的必要条件。

**报告含义**：几何平均 1.2570 会掩盖 0.78× 的单点崩。**必须报告最差 shape**。

---

## F6. 初筛子集夸大收益

**症状**：初筛高，完整矩阵低。

**数据**：首轮初筛 6 点约 1.362×，完整 30 点为 1.293×（`optimization_plan.md:44`）。

**根因**：6 点里 1024/4096 权重过高，而这两个尺寸恰是特化最受益的。

**规矩**：初筛（B 级）只能用于选拔，不能作为最终结论。完整矩阵 + 独立复测（A 级）才算。

---

## F7. 跨层/跨口径比较得出错误归因

**症状**：拿 TTIR 的 1.295× 直接对 LLIR 的 1.476× 说"TTIR 层不如 LLIR"。

**根因**（`baseline_audit.md`）：两个 baseline 的机器码本来就不一样，因为走了不同后端（独立 `llc` vs Triton 内置）。`.text` 哈希不同。

**正确做法**：用同一 baseline 重算。同协议下分解为 `1.476/1.295 ≈ 1.045 × 1.091`，其中 1.045 是 baseline 差异，1.091 才是优化版差距。

**规矩**：跨层比较必须（a）同一 baseline 重算，（b）注明后端差异，（c）不把后端差异算成优化能力差异。

---

## F8. 把 `.text` 一致写成 "cubin 字节一致"

**症状**：报告结论过强。

**事实**：`canonical` 与 baseline 的 `.text` SHA256 完全相同，但**完整 cubin 因调试信息不同而不一致**。

**规矩**（`TTIR_EXPERIMENT_REQUIREMENTS.md` §5）：只能证明 `.text` 一致时，必须明确记录区别，不得写为"cubin 字节一致"。

---

## F9. 调试/位置信息被误认为计算差异

**症状**：两个 TTIR 看起来不同，怀疑计算变了。

**事实**：`baseline.ttir` 的 LLIR 与旧实验 `baseline.llir` 的**全部计算代码逐字节一致**，区别仅为末尾调试元数据的文件名/路径/行号（`baseline_audit.md:7`）。

**方法**：对比较副本去除位置或调试元数据，并检查完整 diff；保留原件和哈希。`#loc`/`loc(...)` 适用于 MLIR，LLVM 的 `!dbg`/metadata 需分别处理，不能只剥 MLIR loc 就宣布 LLVM 计算等价。TTGIR 可用相同 parser round trip 核查局部 SSA 重命名。

---

## F10. 用 `maxnreg` 硬压寄存器

**潜在症状（D 级）**：寄存器下降但 spill 上升，性能可能下降。本库没有 maxnreg 对照实验，不登记为已测失败。

**来源**：`llir_comparison.md:71` 的建议，不是 maxnreg 消融结果。

**机制假设**：硬限寄存器可能引起 spill、重计算或不同调度，也可能改善 occupancy；收益方向需要实测。改变编译选项不属于当前“只改 TTIR、同 options”的实验，若开展需另建匹配 baseline 的分组。

**正确做法**：从 TTIR 侧减少存活值（窄 tile、减少展开宽度、延迟定义），让后端自然分配得更少。

## F11. 文件入口支持某机制，不等于本轮允许使用

旧结论“TTIR 文件入口不接受内联 PTX”已反证（C 级）：官方 Triton 3.7.0 的 `tt.elementwise_inline_asm` 独立探针可编译、执行并正确取负。LayerNorm 第二轮的“不使用内联 PTX”是冻结实验约束，仍然有效。新增算子应单独声明是否允许 ISA 相关机制；不能把政策限制写成编译器能力限制。

探针、逐阶段产物和官方 op 定义见 [核查证据](evidence-index.md#kernelforge-知识核查)。

## F12. 把 autotune 的 Infinity 当成算术失败

**观测（C 级诊断）**：MM 基线代表点的 16 配置 autotune 中，5 个配置返回 Infinity。单独编译/加载诊断确认 shared-memory OutOfResources，见 `mm/results/runs/baseline_20261118T124859Z_Vh0qLn/autotune_failures.json`。

**处理**：保留原始选择记录，记录具体异常与资源限制，区分编译、加载、执行和数值失败；不静默重调参或替换 baseline。Infinity 单独不提供具体原因，也不能证明所有失败配置数值错误。选择后的固定配置只对其测试矩阵有效，不代表逐 shape autotune 最优。


## F13. 合理的变换组合后，性能反而更差

**观测（B级初筛/C级资源）**：`r2_delay`几何1.4017×、80寄存器/4 spill；`r2_delay_affine_v2`几何1.2008×、最差0.9600×、94寄存器/0 spill。独立affine修复版为1.3689×。两个组合不在同一计时进程，不能精确拆分每项耗时，但本轮没有依据接受组合为优胜者。

**含义**：消除spill也不足以证明更快；改写可能改变活跃状态、调度或资源限制，具体因果需消融。不要把父版本PASS或单项收益直接继承给组合。

来源：[初筛1](../layer_norm/results/round2/screen_batch1.json)、[初筛2](../layer_norm/results/round2/screen_batch2.json)、[资源](../layer_norm/results/round2/resources.json)。

## F14. 把平均更快写成逐点无回退

**A级观测**：最终混合分组版对baseline全部30点加速，却只在22/30点优于parent，最大回退3.63%。N=13312的parent约77.775µs，新版约80.598µs。

**处理**：明确baseline、parent与应用分布各自的参照；提前声明是否要求逐点无回退。本轮目标为整体矩阵平均逼近LLIR，没有逐点无回退承诺。来源：[主测](../layer_norm/results/round2/full_stable.json)、[报告](../layer_norm/report_round2.md)。

## F15. 不稳定计时不能靠挑样本补救

**观测**：初次完整测试同一baseline在N=1024约31–68µs漂移，原因未知。该轮被中止并整轮排除，保留原始partial数据；不是只删除慢样本或改成取最小值。

**处理**：排查GPU状态，对所有版本应用相同untimed稳定阶段，再重跑完整矩阵和独立复测。此处采用graph warmup的rep=100ms，不代表预热能保证无干扰或已证明原始漂移原因。

来源：[无效性登记](../layer_norm/results/round2/measurement_validity.json)、[弃用样本](../layer_norm/results/round2/full_benchmark.json)、[有效主测](../layer_norm/results/round2/full_stable.json)。NCU计时、常规计时、弃用计时必须分开归档。

## F16. 用可更新别名关联历史正确性

**观测（运行脚本失败）**：第一次独立复测在GPU计时前因验证记录缺少`tile_dispatch_math_v2`名称抛出`KeyError`。这不是kernel编译、数值或性能失败。

**处理**：先核对不可变parent候选与当时`optimized.ttir`的内容哈希一致，再在新验证文件中登记别名来源，重新跑计时。后续`optimized.ttir`晋升不能改变旧记录含义。

来源：[失败日志](../layer_norm/results/round2/independent_repeat.log)、[正确性别名记录](../layer_norm/results/round2/accuracy_finalists.json)、[成功复测](../layer_norm/results/round2/independent_repeat_v2.log)。长期标识采用不可变候选名+TTIR哈希，文件别名只作便捷入口。
