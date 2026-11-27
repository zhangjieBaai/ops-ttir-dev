# 外部 Softmax LLVM IR 优化：可迁移经验与边界

本记录用于选择归约类算子的优化假设。来源是外部LLVM IR重写，尚未在本项目实现或验证对应TTIR候选。

## 来源、范围与证据

- 阅读日期：2026-11-19。
- 固定commit：[`0x45f/ir-ops@fe26315a12ed9739f611a7f2c57926a0ecc1fc57`](https://github.com/0x45f/ir-ops/tree/fe26315a12ed9739f611a7f2c57926a0ecc1fc57)，分支 `opt-softmax-alll-shape`。
- baseline：[triton_softmax.py](https://github.com/0x45f/ir-ops/blob/fe26315a12ed9739f611a7f2c57926a0ecc1fc57/triton_softmax.py)；改写器：[edit_softmax_ir.py](https://github.com/0x45f/ir-ops/blob/fe26315a12ed9739f611a7f2c57926a0ecc1fc57/edit_softmax_ir.py)；运行时：[softmax_llir.py](https://github.com/0x45f/ir-ops/blob/fe26315a12ed9739f611a7f2c57926a0ecc1fc57/softmax_llir.py)。以下“该实现”均指此commit。
- 外部正式环境：H20/SM90，PyTorch 2.12.0+cu130、Triton 3.7.0；二维连续FP32，固定4096行，N=256…12672、步长128，共98点。
- **B级性能**：外部完整矩阵、每点5轮交替顺序；未做本地独立进程复测。**C级静态事实**：已阅读实现、核对运行时哈希。**D级迁移假设**：本项目TTIR实现方式与收益。
- 本次没有执行外部代码、GPU benchmark或sanitizer。只核对源码与保存的哈希，复算已有数据；不登记为本项目A级成果。

该实现以保存的 `.llir` 为运行时输入，经Triton NVIDIA后端编译为PTX/cubin。改写器直接写LLVM SSA文本，含内联PTX；没有通过修改Triton计算源码生成优化IR。其实验允许修改host、grid、warps、metadata和分派，范围比本项目部分固定配置的TTIR契约更宽。

## 已启用策略与机制

下表描述 **C级代码事实**；组合性能为B，不能把总收益分配给单项策略。

| 策略 | 该实现怎么做 | 可迁移判断与代价 |
|---|---|---|
| 按行宽选择线程分工 | 固定128线程/CTA。N≤1024时每warp一行、每CTA四行；更大N每CTA一行 | 小行减少跨warp同步，大行增加合作线程；阈值和线程数必须随目标硬件、dtype重新筛选 |
| 寄存器驻留 | 向量加载后保留输入，完成max、exp、sum、归一化再写回；取消原持久化行循环和cp.async多级输入流水线 | 将shared容量压力转为寄存器存活压力；不保证寄存器更少或所有尺寸更快 |
| 分层归约 | 线程内树形归约，再warp butterfly shuffle；大行每warp写一个FP32部分值，共16 B shared | 小行0 B shared；大行复用scratch，max和sum两次归约合计四次CTA barrier，不能称零同步 |
| 非二次幂展开 | 小行按128列、大行按512列向上取整，最小256，保存37份IR | 减少尾部无效元素和归约工作；文件数量、代码体积、首次编译与缓存成本增加 |
| 完整组免mask | 除最小256列档外，最后展开组之前的load/store无谓词；尾组保留边界检查 | 依赖“选择最小覆盖分档”的运行时不变量，不能任意搭配更大的IR文件 |
| 向量访存与对齐 | 每线程四个连续FP32，内联PTX `ld/st.global.v4.b32` | 指针需16字节对齐；运行时要求列数为16倍数，不满足时补-inf并拷贝，输出再裁剪为连续张量 |
| 归一化算术 | 保留减行最大值，使用exp2近似、分母近似倒数及乘法 | baseline已经使用exp2近似；明确变化之一是逐元素`div.full`改为复用`rcp.approx.ftz`，需验证舍入与特殊值 |
| 编译与加载缓存 | 按GPU和列宽分档缓存可执行kernel；调用时查询当前stream | API总收益包括host路径变化，不能全部称为LLVM计算优化；编辑IR后缓存必须刷新 |

### 非二次幂范围的实例

| N | baseline展开长度 | 改写后展开长度 | 无效元素数变化 |
|---:|---:|---:|---:|
| 5504 | 8192 | 5632 | 2688→128 |
| 8320 | 16384 | 8704 | 8064→384 |
| 12672 | 16384 | 12800 | 3712→128 |

这些是 **C级范围计算**，不能单独证明性能贡献。N=8320的外部组合内核约2.004×，同时还改变线程数、流水线、归约、数学指令和grid。

### 为什么可以只检查尾组

大行分档单位为128线程×4元素=512列。选择最小覆盖档B后，前B/512−1组全部落在有效范围；最后一组仍按线程的四元素向量起点判断。该实现先保证列数和地址对齐，才允许整向量访问。小行采用相同思路，组宽为32线程×4元素=128列。

**合法性先于性能**：证明必须覆盖所有实际执行的load和store，包含没有最终有效输出的lane；还要检查部分CTA的行范围和跨warp同步条件。只验证store mask或常见对齐shape不足以支持删mask。

## 外部结果的正确引用

来源：[正式结果JSON](https://github.com/0x45f/ir-ops/blob/fe26315a12ed9739f611a7f2c57926a0ecc1fc57/perf_llir_round2/results.json)、[摘要](https://github.com/0x45f/ir-ops/blob/fe26315a12ed9739f611a7f2c57926a0ecc1fc57/perf_llir_round2/summary.json)、[测试入口](https://github.com/0x45f/ir-ops/blob/fe26315a12ed9739f611a7f2c57926a0ecc1fc57/benchmark_softmax_llir.py)。数值与文件身份见 [证据索引](evidence-index.md#外部-softmax-llvm-ir-经验)。

| 指标 | 外部记录 | 等级与限制 |
|---|---:|---|
| 完整调用几何平均加速 | 1.359663× | B；CUDA events计时，包含分配和启动，排除首次编译；不等于主机墙钟请求延迟 |
| CUDA graph内核几何平均加速 | 1.464707× | B；预分配输出、graph replay；与完整调用缓存/调用条件不同 |
| 最低逐点加速，API / kernel | 1.050483× / 1.073465× | B；无显著退化，只约束这次外部矩阵 |
| 98点最大绝对误差 | 2.9802322387695312e-08 | 外部随机输入记录；不是任意输入误差上界 |
| 验收状态 | `accepted=false` | 目标为API几何平均>1.6×，没有达标 |

外部正式参数为5轮、warmup=30ms、API rep=150ms、kernel rep=20ms。逐点取每轮结果的中位数，再汇总等权几何加速比；脚本另做配对轮次log-ratio bootstrap。bootstrap区间只描述已保存轮次的波动，不证明无干扰或跨进程可重复性。正式列宽均为128倍数，没有测量非对齐padding路径的性能。

核对：全部运行时文件与正式JSON的 `source_sha256` 一致；98点各5轮，逐点中位数和汇总几何均值复算一致。本次没有重测GPU。外部测试及工具日志记录边界、NaN/Inf、非默认stream、跨GPU和memcheck/racecheck；这些保持为外部记录，本地状态为NOT_RUN。

## 未采用尝试与诊断经验

来源：[第二轮记录](https://github.com/0x45f/ir-ops/blob/fe26315a12ed9739f611a7f2c57926a0ecc1fc57/optimization_softmax_1_6.md)、[第三轮记录](https://github.com/0x45f/ir-ops/blob/fe26315a12ed9739f611a7f2c57926a0ecc1fc57/optimization_softmax_round3.md)及其链接的原始数据。候选计时属于 **B级外部筛查**，不推广为通用失败规律。

| 尝试 / 诊断 | 外部结论 | 可迁移经验 |
|---|---|---|
| 持久化、多行CTA、增加线程/非二次幂warp | 没有形成稳定优于正式实现的配置 | 取消persistent的收益取决于行数、CTA成本和资源，不能机械套用 |
| 缓存提示、组轮转、整数redux max、减少barrier | 未合入 | 检查是否改变瓶颈；减少某类指令不保证减少耗时 |
| 不减最大值的exp快路径 | 带稳定回退仍变慢，未合入 | 保留稳定性；额外分支与双路径可能抵消算术节约 |
| TMA整行加载/写回、L2预取、两遍读取 | 没有稳定收益，未合入 | TMA/shared中转和额外读取都有成本，需要匹配复用和吞吐瓶颈 |
| 强制maxnreg | 首次命令行限制未生效；后续LLVM属性限制确实生效，但候选回退 | 先看实际寄存器是否变化，再判断实验；不能把未生效参数当负结果 |
| 纯拷贝诊断 | 同一读写布局去掉数学后，整体耗时与Softmax接近 | 支持当前路径访存受限的解释；不是Softmax成绩，也不是严格带宽上界 |
| 压缩/额外舍入 | 未接入正式实现 | 新的数值或内存契约须单独声明，不能用更宽容差冒充原目标收益 |

测量流程还有一个明确反例：早期graph捕获曾闭包保存默认stream，导致空graph；该批数据被作废，改为launch时查询当前stream后重测。复用测试框架时，应确认捕获图中确有目标kernel。

## 迁移到本项目TTIR的实验顺序

以下均为 **D级假设**，没有Softmax TTIR性能证据。

1. **先固定契约和baseline。** 重新声明GPU、dtype、对齐、行宽范围和可改的host/grid/options；按本项目协议运行同链路pipeline核验。外部H20的FP32结果不继承为本项目H100或FP16结果。
2. **先检验有效范围和尾部。** 用合法二次幂子张量组成完整分组与尾组，分别改变展开范围和mask。不能直接把 `tt.make_range` 的end改成8704就假定普通TTIR后端支持。
3. **独立检验线程分工和数据存活。** 改逻辑行映射、分组或重载后，检查TTGIR布局、实际向量指令、shared/barrier、registers/spill。普通 `tt.reduce` 不直接指定warp shuffle或16 B shared；精确ISA控制另查契约。
4. **算术单独验证。** 倒数复用另建候选；检查误差、NaN/Inf、全-inf行和极端有限值。内联PTX与approx/FTZ是否允许，不能从外部实现推定。
5. **组合重新验收。** 做边界、部分CTA、padding/未对齐、stream测试，再按本项目完整矩阵、多轮配对、独立进程复测和sanitizer要求决定选择。不能给单项变换继承组合加速比。

本例补充的是 [T2分派](techniques.md#t2-按尺寸分派-tile-宽度最强杠杆)、[T5算术](techniques.md#t5-算术表达式改写收益小)之外的有效范围与边界证明，技术入口见 [T15](techniques.md#t15-根据有效范围减少归约的-padding-和边界判断)。寄存器驻留与 [LayerNorm重载经验](layernorm-round2.md#e2-多保留一次-load可能比长时间保留整行数据划算)适用条件不同；两者都需要先判断存活压力和访存成本。
