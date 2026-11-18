# Baseline 一致性核查

用户指出跨层级优化比较应使用相同 baseline，这个要求正确。此前仅确保每条实验内部 baseline/候选使用各自相同后端，没有保证跨实验后端一致，不能把各自加速比的差别全部归因于 TTIR/LLIR 修改。

## 已确认的事实

1. 当前 TTIR 编译出的 baseline LLVM IR 与旧 `ir_opt/baseline.llir` 的全部计算代码逐字节一致；区别仅为末尾调试元数据中的文件名、路径和行号。
2. 旧项目原始 `compiled/_layer_norm_fwd_fused.cubin` 与当前 `artifacts/baseline/kernel.cubin` 的 ELF `.text` SHA256 **完全一致**：`4fb552d537bd8bff5396cb8a2287af0f9a70a37c34eef528bd83b5f376b52996`。
3. 旧 LLIR 实际实验使用的 `ir_opt/baseline.cubin` 是经过独立 llc 重编译的版本，`.text` SHA256 为 `813cd28bdfa0c13b604cd4b19c4369e94a95dca96eb8526a5cc3ce79998b70c3`，与教程原始 cubin 不同。
4. 两份 LLVM IR 分别通过旧 `compile_llir.py` 重编译，均复现旧实验 baseline 的 `.text` 哈希；旧 LLVM IR 经 Triton 的原生 make_ptx/make_cubin（包含 init_targets 和 set_short_ptr 初始化）编译，则复现当前 baseline 的 `.text` 哈希。差异因此定位到 LLVM→PTX→cubin 后端流程，与教程计算源码和 TTIR 修改无关。

## 后端差异

旧命令是 `llc -march=nvptx64 -mcpu=sm_90`，然后 `ptxas -O3 -arch=sm_90`。Triton 则使用内置 LLVM，显式启用浮点融合、MAD wide 优化、short pointer 等设置，使用对应 PTX features，ptxas 还有 `--regAllocOptLevel=2` 等参数。不能把“同为官方 Triton 3.7 导出的 LLVM IR”视作后续编译参数也相同。

| 静态 SASS 指令 | 旧 llc baseline | 原生 Triton baseline |
|---|---:|---:|
| FADD | 239 | 174 |
| FMUL | 110 | 45 |
| FFMA | 0 | 65 |
| 指令总数 | 856 | 792 |

在旧 llc 命令中仅增加 `-fp-contract=fast`，也得到 65 条 FFMA、相同上述指令计数，但机器码哈希仍不同。因此可以确认浮点融合策略不一致；未经性能消融，不能把全部 4.5% 都精确归因于 FMA，地址生成、寄存器分配和调度等后端差异也存在。

## 修正比较口径

上次四版本同 GPU 计时已保留每个测量块的全部耗时，因此可以用同一个 baseline 重新计算，无需更改或猜测旧数据。

| 统一参考（30 shape 几何平均） | LLIR 优化版加速比 | TTIR 优化版加速比 |
|---|---:|---:|
| 当前原生 baseline | **1.412×** | **1.295×** |
| 旧 llc baseline | **1.476×** | **1.353×** |

这才是使用共同参考的比较；不应直接以 `1.476×` 对 `1.295×` 作为纯优化效果比较。上表只统一比较参考，两个优化版的生成后端仍不同，尚不能看作严格隔离“优化层级”这一因素的消融。严格研究 TTIR/LLIR 的效果，还需固定从共同 LLVM 阶段之后的编译链。

既有 baseline 文件、候选和原始测量均未修改；历史结果保留为各自流程内部的有效结果。后续跨路线比较应检查 baseline `.text` 哈希及编译配置，报告统一参考；原生教程 baseline 是当前共同参考的合理选择。

## 证据

- `results/baseline_audit/audit.json`：计算 IR 一致性、交叉编译和机器码哈希、指令统计。
- `results/baseline_audit/common_reference.json`：从既有配对块重算的共同 baseline 比值。
- `audit_baseline.py`：审计脚本。可再生 cubin/PTX 在忽略目录 `artifacts/baseline_audit/`。
- 审计工具接入过程的失败日志保留：初次误用不支持 LLVM 文本的 `triton.compile` 文件入口；第二次缺少 LLVM target 初始化而终止；第三次缺少 short-pointer 初始化导致哈希断言失败；最终补齐与 Triton make_llir 相同的初始化后，所有交叉哈希断言通过。它们均为审计编译流程问题，没有执行新的 GPU kernel。

```bash
/root/venvs/ir-ops-dev/bin/python layer_norm/audit_baseline.py \
  --output-dir layer_norm/artifacts/baseline_audit
```
