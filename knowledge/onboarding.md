# 新算子开工清单

按顺序做：先声明契约，再冻结 baseline 和验证 pipeline，最后写候选。

---

## 0. 先读

- [TTIR_EXPERIMENT_REQUIREMENTS.md](../TTIR_EXPERIMENT_REQUIREMENTS.md) — 协议全文
- 本库 [ttir-model.md](ttir-model.md) 的「表达力边界」— 判断你的目标是否 TTIR 可表达

---

## 1. 环境核查（不能跳过）

```bash
/root/venvs/ir-ops-dev/bin/python - <<'PY'
import sys, torch, triton
from triton._C import libtriton
print('python:', sys.executable)
print('prefix:', sys.prefix)
print('torch:', torch.__version__)
print('triton:', triton.__version__, triton.__file__)
print('libtriton:', libtriton.__file__)
PY
```

**必须确认**：
- `sys.executable` 是 venv 路径
- Triton 是 3.7.0 **upstream**
- `libtriton.so` 在 venv 内，不是系统或 FlagTree 的

`triton.__version__` 单独不够，必须核对实际模块和分发来源；定制编译器可能使用同名 Python 模块，不能据名称推断二进制身份。复用 `layer_norm/environment.py` 的检查，并记录 RECORD 检查的来源限制与实际动态库路径。

环境核查后、导出前，先完成下文第 4 节的契约表。导出依赖该组 dtype、对齐、constexpr、options 与 layout，不能导出后再按结果修改契约。

---

## 2. 冻结 baseline

1. 保存上游教程源码 + SHA256 到 `upstream/`
2. 用 `(M, N)` 代表点调用 `@triton.jit`，取 `compiled.asm['ttir']`
3. 存为 `ir/baseline.ttir`，记录源码哈希与 TTIR 哈希
4. **不得隐式覆盖**——要换 baseline 必须新实验标识

**必须检查导出 IR 的真实前提**，从函数签名读：

```mlir
tt.func public @kernel(%X: !tt.ptr<f16> {tt.divisibility = 16 : i32}, ...)
```

对上面的**指针参数**，`tt.divisibility=16` 表示地址满足 16 字节对齐，不代表元素个数为 16 的倍数；对 `i32` 的 N/stride 参数，它表示整数值能被 16 整除。LayerNorm 的签名同时对 N、stride 加了此属性，因此本组要求它们是正的 16 的倍数。列连续、X/Y 同 stride、无别名和索引范围是另外的契约前提，不能从指针属性推出。

**别把这一条漏了**——用不符合属性的尺寸测出来的正确性结论无效。

---

## 3. Pipeline 一致性核查

`verify_pipeline.py` 必须证明：

- 教程源码导出的 TTIR 与冻结文件字节一致
- 源码路径与文件路径用同一 Triton、同 target、同完整 options
- 源码 baseline 与冻结 TTIR baseline 的 LLVM IR、PTX、cubin 字节一致（当前 LayerNorm/MM 已做到）；只有 cubin 是 ELF，LLVM/PTX 文本没有 ELF `.text` 段
- 若完整 cubin 因调试信息不同，只能另行记录 `.text` 一致及原因；不能将此例外推广到任意 LLVM/PTX 差异
- TTGIR 经 parser round trip 归一化后一致
- 相同输入下全部输出逐位一致

**文件入口从 `make_ttgir` 继续，不重新执行 `make_ttir`。** 不要宣称文件路径跑过 Python 前端。

---

## 4. 确定契约

在导出 baseline 前先在 `optimization_plan.md` 冻结。若必须改变契约，创建新实验分组、版本和匹配 baseline，不能改旧记录后沿用 PASS：

| 项 | 例（layer_norm） |
|---|---|
| dtype | FP16 输入/输出，FP32 累加 |
| 尺寸前提 | 正 N，N 与 stride 均为 16 的倍数 |
| 对齐 | 指针 16 字节对齐 |
| 别名 | 不支持 X/Y 别名 |
| 编译 options | warps=8, CTAs=1, stages=3, fp_fusion=True |
| 容差 | Y < 1e-2；Mean rtol 2e-5；Rstd rtol 2e-4 |
| 测试矩阵 | 30 性能 shape + 压力 + 边界 |

**容差不能抄 layer_norm 的。** 每个算子根据自己的数值范围定。

---

## 5. 生成第一批候选

在写生成器**之前**，先回答四个问题：

1. **主杠杆是什么？** 减少无效工作可试 tile 分派（[T2](techniques.md#t2-按尺寸分派-tile-宽度最强杠杆)）；降寄存器可试窄 tile（[T3](techniques.md#t3-窄-tile-循环省寄存器但会在大-n-崩)），并检查循环/加载代价；这些是候选方向，不是普适最佳方案。
2. **会不会加特化分支？** 会 → 查看编译后的整体资源档位，必要时消融宽分支（[T11](techniques.md#t11-控制特化分支的寄存器预算)）。
3. **目标需要直接控制下游指令吗？** 区分 TTIR 改算法的间接影响、受契约限制的内联汇编和直接手改下游 IR；本轮禁止后两种，不能把这种限制推广为所有 TTIR 的能力边界。
4. **失败时怎么区分"路线错"和"候选有 bug"？** 先写下来。

**建议第一批不超过 5–8 个候选**，可包括：
- 一个 `canonical`（只删 loc）作路径对照
- 每个主要假设一个单项候选
- 与假设相关的负对照；不为了凑数量重跑已确定且条件未变的失败

---

## 6. 验证顺序（不可颠倒）

```
契约 → baseline/pipeline → TTIR 解析 → 全后端编译 → C1 正确性（全用例）→ 初筛计时 → 完整矩阵 → 独立复测 → sanitizer
```

- **正确性失败不进入计时**
- 每版运行前重置输出 sentinel，防止上版结果掩盖漏写
- 记录输入 TTIR 哈希，计时前校验匹配
- 最终候选跑 memcheck/racecheck；失败必须阻止接受

---

## 7. 报告时必须回答的

对照 `TTIR_EXPERIMENT_REQUIREMENTS.md` §10.7：

1. 唯一改变是否为 TTIR？
2. 两版是否同一环境、同一后端？
3. baseline 是否与教程原生产物一致？
4. 加载/启动/计时是否相同？
5. 数值是否通过？
6. **收益在哪些 shape 有效？**（必须给逐 shape 与最差回退）
7. 有什么限制？

---

## 常见第一轮错误

| 错误 | 后果 | 参见 |
|---|---|---|
| 用不符合 `tt.divisibility` 的尺寸测 | 结论无效 | 第 2 步 |
| 改张量宽度但没同步分支内常量 | C0 编译失败 | [F1](failure-patterns.md#f1-分支内张量宽度与外部零常量不匹配) |
| 窄化后不检查大 N 成本 | 本例曾回退至约 0.75× | [F5](failure-patterns.md#f5-单点窄-tile-在大-n-崩) |
| 用初筛数字当完整结论 | 本例 1.362× 降至 1.293× | [F6](failure-patterns.md#f6-初筛子集夸大收益) |
| 只看几何平均不看最差 | 掩盖单点崩 | [diagnosis](diagnosis.md#典型诊断路径) |
| 未验证就断言 `maxnreg` 有收益 | 可能转成 spill；本库未做该对照 | [F10](failure-patterns.md#f10-用-maxnreg-硬压寄存器) |
| 把资源下降当收益 | 结论错误 | [F4](failure-patterns.md#f4-occupancy-上升但性能下降) |


## 复用第二轮经验时多问三件事

- 我的改动是减少load，还是缩短值存活期？两者可能冲突，应把它们作为不同候选，而非默认“load越少越好”。
- 我看到的是编译失败、数值失败、性能回退，还是计时/验证入口失败？先分类，再决定修复候选还是重跑测量；不可把别名KeyError算作kernel负结果。
- 目标是相对baseline、parent还是应用分布？最终组合对baseline全部点加速，不代表对parent逐点无回退。

参考[LayerNorm第二轮有效与无效经验](layernorm-round2.md)。初筛只能选方向，最终收益仍需新算子自己的矩阵、完整正确性和独立复测支持。
