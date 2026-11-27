# TTIR 比 Triton 语法多出的控制

**在本项目固定的官方 Triton 3.7.0 中，TTIR 的主要增量是逐指令语义属性、直接指定区域/SSA 结构，以及选择前端优化后的编译入口。现有 LayerNorm 的主要算法变换，大多也能用 Triton Python 表达。** 不应把“这次通过改 TTIR 实现”写成“只有 TTIR 能实现”。

本文的“语法”指常规 `@triton.jit` 配合公开 `triton.language` API；“无法表达”指没有直接指定该 IR 属性或结构的公开语法，不是证明不存在等价算法。自定义 builtin、内部 IR builder、编译器修改和产物 override 不在此范围。`tl.inline_asm_elementwise` 是公开 API，必须计入；它能实现一些同类机器级行为，但不是给现有 `arith.*` 指令附加 MLIR 属性的接口。Gluon/TLE、TTGIR 和厂商扩展也不能混作这组 upstream 结论。

除另有说明，以下结论都是 **C 级：固定版本的源码核查与编译探针**。没有数值验收或性能收益结论。

## 1. 逐指令浮点语义属性

TTIR 可为某一条运算写出 MLIR fast-math flags，而相邻运算保持原语义：

```mlir
%a = arith.addf %x, %y fastmath<reassoc,nnan,ninf,nsz> : tensor<256xf32>
%b = arith.addf %u, %v : tensor<256xf32>
```

常规 Triton 的 `x + y` 经 `semantic.add()` 调用 `create_fadd()`，没有逐条传入上述 flags 的公开参数。kernel 级 `enable_fp_fusion`、`tl.fma`、`tl.div_rn` 等各有自己的语义，不能替代任意位置的 `reassoc/nnan/ninf/nsz` 属性组合。这里的差异是**语义注解的粒度**，不是 Triton 完全不能控制浮点行为。

| flag | 允许编译器采用的假设/变换 | 使用前提 |
|---|---|---|
| `reassoc` | 允许相应的浮点重结合 | 必须重新评估舍入误差，不能当作严格等价改写 |
| `nnan` / `ninf` | 按相应运算不涉及 NaN / Inf 的承诺优化 | 只知道输入有限仍可能不够，运算结果也可能溢出 |
| `nsz` | 不要求区分正负零 | 需要确认符号零不属于算子承诺 |

**当前实测限制**：`per_op_fastmath` 探针编译到 cubin 成功，flags 在 TTGIR 中保留，但导出的 LLVM IR 是普通 `fadd float`，没有这些 flags。因此本例只证明“TTIR/TTGIR 能表示且整链可编译”，不能证明“这些承诺已传到 LLVM、开启了重结合或带来收益”。其他 op/flag 组合需分别核对。

这些属性会放宽语义约束，不能在既有契约要求 NaN/Inf/符号零行为时随意添加。MLIR 语义依据见 [Arith dialect](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithaddf-arithaddfop)；具体 lowering 以固定版本产物为准。

## 2. 逐指令整数不溢出承诺

TTIR 可给整数加法附加 `nsw` / `nuw`：

```mlir
%next = arith.addi %index, %step overflow<nsw,nuw> : tensor<256xi32>
```

它们分别承诺没有有符号/无符号溢出，**不是饱和加法，也不是运行时溢出检查**。违反相应承诺会产生 poison；不能为了生成更短的索引代码就直接添加。若两项都写出，就必须同时成立。

常规 Triton 整数 `+` 没有逐条声明这两个属性的公开参数。`tl.int32` / `tl.uint32` 选择 dtype，不等于声明“不发生溢出”；`tl.assume` 表达条件，可能帮助推理，但不直接指定某条 `arith.addi` 的 overflow 属性。调试用 `sanitize_overflow` 也不等于 `nsw/nuw`。

**当前实测限制**：`integer_nowrap` 编译成功，属性在 TTGIR 中保留；LLVM IR 的对应指令却是普通 `add i32`，没有 `nsw/nuw`。本例尚未证明后端利用了该承诺。若要把它发展成优化路线，需要先证明索引范围，再核查标志在哪个阶段保留或消失，最后验证数值与性能。

语义依据见 [MLIR 整数加法定义](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithaddi-arithaddiop)。不要把“IR 接受该属性”与“当前 lowering 完整保留它”混为一谈。

## 3. 直接指定区域与 SSA 的传值结构

SSA 是中间表示中“一个值只定义一次”的命名方式；区域参数和 terminator 决定值沿哪条边传递。Python 的变量和 `while` 交给前端推导这些结构，TTIR 则允许直接写出合法的区域接口。

一个已编译通过的例子是 **before 与 after 区域使用不同类型的 `scf.while`**：

```mlir
%r = scf.while (%i = %zero) : (i32) -> f32 {
  %cond = arith.cmpi slt, %i, %N : i32
  %f = arith.sitofp %i : i32 to f32
  scf.condition(%cond) %f : f32
} do {
^bb0(%f: f32):
  %ii = arith.fptosi %f : f32 to i32
  %next = arith.addi %ii, %one : i32
  scf.yield %next : i32
}
```

这里 before 接收 `i32`；`scf.condition` 向 after 或最终结果传 `f32`；after 再向下一次 before 传回 `i32`。MLIR 允许这种接口，实测 `asymmetric_while` 从 TTIR 编译到 cubin 成功。

Triton 3.7.0 的 `visit_While()` 使用同一组 `init_tys` 构造两侧 block，并把 `block_args` 传给 `scf.condition`；循环变量还需保持类型一致。它没有公开语法让用户直接指定上述不同的区域参数接口。

**这个差异的价值是控制数据流图，不是新增可计算的函数。** 同类计算通常可以通过拆循环、显式预取或增加状态变量写成 Triton；其导出图未必相同。这里的 cast 例子仅用于展示接口，未验证其所有 N 的数值语义，也没有优化价值证明。若据此探索“条件区计算的值直接传入循环体”，必须另做真实 load/依赖的候选、完整正确性与性能实验，不能从接口能力推导收益。

依据：[SCF while 定义](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scfwhile-scfwhileop) 与 [固定版本前端 `visit_While`](https://github.com/triton-lang/triton/blob/v3.7.0/python/triton/compiler/code_generator.py)。本结论针对直接生成的结构；不排除前端后续优化自动产生相似结构。

## 4. 从指定 TTIR 截点继续编译

这项是**编译入口控制**，应与“语法没有这个运算”分开记录。

源码路径在前端构造 IR 后执行 `make_ttir`，包括 inline、canonicalize、combine、broadcast 重排、CSE、DCE 和 loop unroll。文件入口 `triton.compile("candidate.ttir")` 从下一阶段 `make_ttgir` 继续。因此手写 TTIR 能作为一个精确的实验输入，提交源码路径在该截点未必保留的 op 排列、重复值、属性或区域接口。

例如，同一指针的两次普通 `tl.load` **语法能写**，但本轮 `source_duplicate` 在导出的 TTIR 中已经只剩一次 load。手写 `duplicate_loads.ttir` 可把两次读取交给下一阶段；不过本例的 TTGIR 也只剩一次、PTX 也只发出一次读取。绕过 `make_ttir` 不等于绕过所有 CSE，更不等于保证加载次数、调度、寄存器数或 barrier 数。

所以，直接改 TTIR 的实际优势常是**研究某个编译阶段的输入与变换**：排除 Python AST 推导和前端规范化的影响，再检查下游是否保留预期差异。它不是通用的“禁止优化”开关。该入口规则已由 [实验协议 §4](../TTIR_EXPERIMENT_REQUIREMENTS.md#4-从-ttir-开始的统一编译链) 和本库 pipeline 核查固定。

## 5. 这些优化也能用 Triton 表达

| TTIR 中的写法/变换 | 常规 Triton 的对应方式 | 应如何归类 |
|---|---|---|
| 改逻辑 tile 宽度、展开/分组、按 N 分派 | `tl.arange`、constexpr、`tl.static_range`、运行时 `if`/循环 | 算法与张量结构选择；导出结果仍需核查 |
| `tt.reshape` + `tt.reduce` | `tl.reshape`、`tl.sum` / `tl.reduce` | 并非 TTIR 独有 |
| `.ca/.cg/.cs`、eviction policy、volatile | `tl.load/store` 的 `cache_modifier`、`eviction_policy`；`tl.load(..., volatile=True)` | API 已公开；不是“只有 TTIR 才能强制重载” |
| 倒数复用、rsqrt、FMA、移动统计量 store | 普通表达式、`tl.rsqrt`、`tl.fma`、调整源代码次序 | 源码次序不保证最终调度；TTIR 也一样 |
| 内联汇编 | `tl.inline_asm_elementwise` | 两者都能表达；本项目是否允许另看实验契约 |
| 多值/自定义规约 | `tl.reduce` 与 combine 函数 | 不能把一般 reduction region 当作 TTIR 独有 |
| 对齐、连续性等事实提示 | `tl.multiple_of`、`tl.max_contiguous`、`tl.assume` | 可表达提示；不保证生成特定 signature 属性或指令 |
| 循环流水、展开等公开属性 | `tl.range` 的 `num_stages`、`loop_unroll_factor`、`disable_licm` 等参数 | 要先查本版本 API，不能把所有 loop attribute 都列作独有 |

API 依据：[Triton v3.7.0 core.py](https://github.com/triton-lang/triton/blob/v3.7.0/python/triton/language/core.py) 与 [semantic.py](https://github.com/triton-lang/triton/blob/v3.7.0/python/triton/language/semantic.py)。既有第二轮 1.440× 证明所选组合在固定矩阵上的收益，**没有证明这些收益依赖语法无法表达的机制**。

物理寄存器分配、精确 warp/shared 布局和逐条增删 lowering barrier，也不能反过来当作普通 TTIR 的直接能力；边界见 [TTIR 模型](ttir-model.md#直接控制间接影响与本轮限制)。允许内联 ISA 或换用其他语言层时，应另行声明范围。

## 证据与使用方式

本轮证据在 [`ttir-vs-syntax-20261118`](../../KernelForge-workspace/runs/ttir-vs-syntax-20261118/)：

- `probe.py`：可复现的 4 个源码对照与 5 个手写 TTIR 编译探针，输入和编译器回显分别保存，未触碰冻结算子源码。
- `compile-v1/results.json`：官方 Triton 3.7.0 的环境/工具链、完整 options、输入与各阶段 SHA256，全部 9 个编译到 cubin 成功。
- `observations.json`：flags 在 TTGIR/LLVM 的保留情况、load 合并现象；`sources/installed-manifest.json` 保存实际核查的前端代码身份。
- 编译显式指定 `GPUTarget("cuda",90,32)`，`enable_fp_fusion=False`；未初始化或启动 GPU kernel，未进行数值和性能验证。C 级结论不能升级为优化路线的 A/B 级性能证据。

选优化方向时，依次确认：公开 API 是否已有表达方式；差异是语义、IR 结构还是编译截点；完整 lowering 是否接受且保留它；是否满足冻结契约。只有最后经过准确性和同协议性能测量，才能声称它是一项有效优化。
