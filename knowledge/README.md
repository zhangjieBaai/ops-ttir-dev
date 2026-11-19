# TTIR 优化知识库

跨算子复用的、有证据支撑的 TTIR 优化知识。**不重复**已有文档的内容，只补它们不做的那部分。

## 这个库解决什么问题

仓库里已有两份文档，职责明确：

| 文档 | 职责 | 不覆盖 |
|---|---|---|
| [TTIR_EXPERIMENT_REQUIREMENTS.md](../TTIR_EXPERIMENT_REQUIREMENTS.md) | 实验**协议**：环境、冻结、编译链、验证、计时 | 具体技术怎么做、为什么某条路走不通 |
| `<op>/optimization_plan.md` | 单个算子的**契约与轮次** | 跨算子可迁移的判断 |

本库补的是第三块：**当你打算改 TTIR 时，哪些改动值得做、哪些已被证伪、某个数字到底能信到什么程度。**

## 索引

| 文件 | 内容 | 什么时候读 |
|---|---|---|
| [ttir-model.md](ttir-model.md) | TTIR 在哪、能控制什么、**不能控制什么** | 动手前。决定收益是否是 TTIR 可表达的 |
| [layernorm-round2.md](layernorm-round2.md) | 已完成第二轮的有效方法、负结果及迁移边界 | 想从实战过程选择候选时 |
| [techniques.md](techniques.md) | 技术目录，每项带证据等级与代价 | 选下一个候选时 |
| [failure-patterns.md](failure-patterns.md) | 已证实的陷阱与确切报错 | 候选编译/正确性失败时 |
| [diagnosis.md](diagnosis.md) | NCU 指标与反汇编怎么读 | 需要解释"为什么快/慢"时 |
| [evidence-index.md](evidence-index.md) | 每个数字 → 源文件 | 引用任何结论前 |
| [onboarding.md](onboarding.md) | 新算子开工清单 | 开始一个新算子时 |
| [verify_claims.py](verify_claims.py) | 独立能力探针、历史配对样本复算 | 复验本轮知识修订时 |

## 证据等级

任何结论必须标等级。等级不是主观评分，是**测了什么**：

| 等级 | 含义 | 可否写进最终报告 |
|---|---|---|
| **A** | 完整冻结性能矩阵、多轮配对、独立进程复测；本库 LayerNorm 为 30 shape | 可以，注明算子、矩阵和计时边界 |
| **B** | 中间性能证据：子集初筛，或尚缺独立复测的完整矩阵 | 可以，写明“初筛”或“完整矩阵、未独立复测”，不能称最终结果 |
| **C** | 单点、单次、静态检查或独立能力探针 | 可引用已验证事实与机制线索，不能证明性能收益 |
| **D** | 假设，未测量 | 只能写成假设 |

初筛数字**不能**代替完整矩阵结论。已有教训：首轮初筛 6 点约 1.362×，完整 30 点为 1.293×，因为 1024/4096 权重过高。

等级约束的是**当前这条结论**：同一候选的计时可能为 B，机器码哈希为 C，正确性与 sanitizer 则分别记录实际用例和 PASS/FAIL/NOT_RUN。不能把邻近的 A 级总收益传给单项变换或静态解释。首轮初筛实际 rep=30ms、第二轮 rep=10ms；以原始 JSON 为准。静态事实可被直接证实，C 不表示事实不可信，而是它不能证明加速。

上述 A/B 澄清保留既有等级，修正了“canonical 为 A”“所有 B 都是 6 点”的矛盾。性能有效性另行判断：弃用、干扰和哈希不匹配不因矩阵完整而变有效。报告区间、逐点最差值和跨进程差异；历史缺失的检查保持未测，详见 [诊断中的不确定性说明](diagnosis.md#性能不确定性与无效样本)。

## 当前算子状态

| 算子 | 冻结 baseline | 最优版本 | 收益（A 级） | 备注 |
|---|---|---|---|---|
| layer_norm | ✅ | `r2_mixed_stream` | 几何 1.440× / 算术 1.499× | 第二轮完整矩阵、独立复测与 sanitizer 已有记录；相对 parent 部分点回退 |
| mm | ✅，31点映射到6组 | 128×32×64组 `r1_unsigned`；其余baseline | 该组3点几何1.0510× / 复测1.0507× | 无整体31点可靠加速结论；见[T14](techniques.md#t14-将已证明非负的整数索引除余改为无符号运算) |

LayerNorm当前选择以 [`selection.json`](../layer_norm/results/selection.json) 和候选哈希为准；第一轮 `tile_dispatch_math_v2` 的 1.293× 是历史结果。第二轮 `full_stable.json` 的 `optimized` 标签指第一轮 parent，不能按今天的 `ir/optimized.ttir` 重解释旧样本。MM当前选择以 [`round1_selection.json`](../mm/results/round1_selection.json) 为准；上述A级收益只限128×32×64组的384³、512³、640³三点，不能推广到其他组或全部31点。

## 第二轮经验入口

[LayerNorm 第二轮复盘](layernorm-round2.md)把完整组合与单项证据分开：分组累加/输出、范围分派和加载/缓存表达的组合有A级1.440×；32寄存器窄循环、volatile重载、延迟store与affine组合等没有成为最终选择。另记录了基线漂移整轮弃用、候选别名验证失败和平均收益掩盖局部回退的教训。新算子先迁移判断方法，不直接迁移tile宽度或收益百分比。

## 使用方式

1. 动手前读 [ttir-model.md](ttir-model.md) 的"表达力边界"一节，先判断你要的收益是否可能由 TTIR 承载。
2. 选候选时查 [techniques.md](techniques.md)，优先复用 A/B 级技术，谨慎对待已被证伪的方向。
3. 失败时查 [failure-patterns.md](failure-patterns.md)，多数坑已有确切报错可对照。
4. 引用任何数字前，到 [evidence-index.md](evidence-index.md) 核对出处与等级。

## 用 KernelForge 检验知识

先用问题检索规则与反例，再回到本项目的契约、产物和原始 JSON。当前 KernelForge 没有 LayerNorm/MM 执行适配器；它的 task 返回 `discovered=false`，不能据此宣称 KernelForge 已执行算子 qualification。本轮使用其 workflow、benchmark-evidence、diagnose-kernel 和 review-agent 技能进行核查；算子执行仍走本项目官方 Triton venv。

```bash
cd /root/KernelForge
KERNELFORGE_PLATFORM=h100 python -m kernelforge task --op layer_norm --platform h100 \
  --query 'TTIR 能力边界、失败适用条件与配对性能证据'
# 先确认空闲 GPU，再用新目录保留探针、日志和全部产物。
cd /root/ops-ttir-dev
CUDA_VISIBLE_DEVICES=0 /root/venvs/ir-ops-dev/bin/python knowledge/verify_claims.py \
  --output /root/tmp/ttir-claims-new
```

复现主实验前仍需先运行算子自己的 `verify_pipeline.py`。探针是独立 FP32 能力检查，不是 LayerNorm 新候选，不改变其 FP16/禁用内联 PTX 契约。历史样本 bootstrap 是派生分析，不是新 benchmark；原始来源哈希、限制和本次路径见 [核查证据](evidence-index.md#kernelforge-知识核查)。KernelForge 的其他算子计时标准不替代本项目冻结的 CUDA graph kernel 指标。

## 维护规则

- 新增技术或失败模式，必须附证据等级与源路径；没有证据的写为假设（D）。
- 证伪的结论**保留**，标记为"已证伪"并写明反证条件。删掉会让下一个人重跑。
- 数字变化（如某候选被独立复测推翻）时，改结论不改原始证据，并注明修订原因。
- 本库只写跨算子可迁移的内容。单算子细节留在该算子的 `optimization_plan.md` 与 `results/`。
