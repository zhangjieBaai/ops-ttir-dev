# LayerNorm TTIR 首轮实验

## 契约和基线

- 官方 Triton v3.7.0 `05-layer-norm.py` 的 `_layer_norm_fwd_fused`，保留原始前向计算。上游快照和 SHA256 见 `upstream/`。
- FP16 X/Y/W/B，FP32 累加、Mean/Rstd；每行一个 program，X/Y 同 stride，连续列；不支持输入输出别名。
- 固定 `BLOCK_SIZE=8192, num_warps=8, num_ctas=1, num_stages=3, enable_fp_fusion=True`，H100/SM90。
- baseline 来自 `(1151,8192)` 教程调用。其 TTIR 的 stride/N 有 `tt.divisibility=16`，实验契约因此限定正 N、stride 都为 16 的倍数；不把不满足该前提的尺寸纳入正确性结论。
- `eps=1e-5`；X/Y 地址 16 字节对齐，W/B 对齐、连续。默认随机分布沿用旧实验 `-2.3+0.5*randn`。
- 冻结 `ir/baseline.ttir`，禁止隐式覆盖。所有候选从该文件确定性生成，禁止改 Python kernel、TTGIR、LLVM、PTX 或 cubin。运行时 N 保持动态，候选可在 TTIR 内加入范围判断。
- 官方教程源码 → 导出 TTIR → 官方 Triton 编译 TTIR → 真实 CUDA launch。重编译 baseline 与源码路径的 cubin 一致，Y/Mean/Rstd 逐位一致。

## 验证与测量

1. C0：TTIR 解析和全后端编译、参数签名一致；保存输入与编译产物 SHA256、寄存器与 spill。
2. C1：31 个原始 shape；压力分布 constant/offset/zero_mean；16 对齐的分支边界、padding、单行和小 M；同时校验 Y/Mean/Rstd 和 padding canary。
3. Y 对比 native ATen，绝对误差 `<1e-2`；Mean 对比 FP64 参考 `rtol=2e-5, atol=1e-5`；Rstd `rtol=2e-4, atol=1e-5`。参考 PyTorch 计算只在测试代码中使用。
4. 基于同一输入和预分配输出的 CUDA graph kernel 计时，交错轮转顺序，保存所有样本。报告是 graph replay kernel 时间，不是原 LLVM 实验包含分配的计时，不能直接比较两者的加速比。
5. 主性能矩阵 `M=4096, N=1024..15872 step512`，30 个 shape；初筛取 6 点、3 轮，候选复核取全矩阵、7 轮并独立重复。
6. 最终版运行 compute-sanitizer memcheck/racecheck。失败、编译错误和性能退步均保存；没有运行的检查不能称通过。

## 首轮可证伪假设和预算

| 候选 | 父版本 | 假设 |
|---|---|---|
| canonical | baseline | 仅删除调试位置，机器码应相同，用作路径对照 |
| arithmetic | baseline | N 倒数复用、sqrt+div 改 math.rsqrt 可能缩短标量路径 |
| single_tile | baseline | N≤8192 时展开三遍单 tile 循环，让公共加载有机会合并 |
| tile_dispatch | baseline | TTIR 内 1024/2048/4096/8192 单 tile 分支和≤16384双 tile 展开，减小无效工作并复用数据；其他 N 保留通用循环 |
| tile_dispatch_math | arithmetic + tile_dispatch | 组合两类修改，检验是否比结构修改更好 |

首轮仅以上 5 个候选；不做无限参数搜索。保留中心化两遍方差，不采用容易受大偏移影响的 `E[x²]-E[x]²`。发现收益后以逐项候选和后端资源辅助解释，不把组合收益归因于未经测量的单项。

## 执行记录与修订

- 首次多分支 TTIR 在 C0 暴露分支张量大小与外部零常量大小不匹配；原文件以失败候选保留，未执行 GPU kernel。修正采用分支局部、同大小的零常量，新命名 `tile_dispatch_v2` / `tile_dispatch_math_v2`，逻辑预算不扩展。
- 第一轮 6 个合法版本通过全部 142 个用例（总 852 次 kernel 检查）。压力/边界沿用旧项目的意图，但用符合 baseline 属性的 16 对齐值，padding=16。
- canonical 与 baseline 的完整 cubin 因调试信息而不同，ELF `.text._layer_norm_fwd_fused` SHA256 完全一致；6 点筛选计时约 1.000×。
- 初筛范围特化候选收益高于 arithmetic 和 single_tile，因此仅这两个合法范围特化候选进入完整 30 点、7 轮复核。筛选数据不作为最终独立验证结果。

## 完整复核与独立复测

- 完整 30 点、每点 7 轮的几何平均：tile_dispatch_v2 为 1.288×，tile_dispatch_math_v2 为 1.293×。按事先约定的全矩阵几何平均选中后者，冻结 selection.json；optimized.ttir 是该候选的字节副本。
- 选中版本独立进程、seed=123、每点 5 轮复测：算术平均 1.316×，几何平均 1.294×，与初轮 1.314× / 1.293× 接近。初筛 6 点中的 1024/4096 权重较高，不能用初筛约 1.362× 代替全矩阵结论。
- 下一轮重点是双 tile 复用产生的寄存器压力与 spill；本轮不再扩展候选预算。
