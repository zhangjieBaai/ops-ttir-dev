# MM 六组 TTIR 优化首轮：1组保留收益，5组继续baseline

六组均完成8个候选的尝试（共48份不同候选文件，不含最终别名）。最终只保留 **128×32×64组的无符号索引改写**：所属384³、512³、640³三点主测几何平均 **1.0510×**，独立进程复测 **1.0507×**。其余五组保留baseline。本轮没有证明31点整体可靠加速，尤其大tile组A/A对照有不稳定点。

| BM×BN×BK | 所属shape数 | 最终选择 |
|---|---:|---|
| 32×64×32 | 1 | 保留 baseline |
| 128×32×64 | 3 | 采用 r1_unsigned |
| 64×128×64 | 6 | 保留 baseline |
| 128×256×64 | 18 | 保留 baseline |
| 64×256×32 | 1 | 保留 baseline |
| 128×128×32 | 2 | 保留 baseline |

## 保留收益与实际运行路径

| M=N=K | 主测 baseline→candidate µs | 主测配对中位加速比 | 独立复测 |
|---:|---|---:|---:|
| 384 | 3.445 → 3.282 | 1.0497× | 1.0517× |
| 512 | 3.843 → 3.620 | 1.0616× | 1.0589× |
| 640 | 4.501 → 4.318 | 1.0418× | 1.0416× |

主测该组95%配对轮次bootstrap区间为[1.0494, 1.0517]，复测[1.0493, 1.0515]。区间只覆盖保存轮次的波动，不含选择偏差、时钟/温度变化或跨环境泛化。384和640曾参与初筛，512是finalist冻结后首次候选性能测量；换seed是独立进程复测，不是新shape holdout。

保留候选仅直接把TTIR的非负索引`arith.divsi/remsi`改为`divui/remui`，不改浮点dot/累加顺序。正尺寸、对齐和存储偏移范围约束确保测试域中这些除余操作数非负。PTX显示对应div/rem采用unsigned并减少部分符号修正；这是静态代码变化，不宣称已完整分离其性能因果。该组baseline/candidate均72 registers、0 spills、81920 bytes shared。

[round1_selection.json](results/round1_selection.json)冻结每组选择、源TTIR和optimized.ttir的SHA256。每组`ir/<family>/optimized.ttir`是所选版本的字节副本，五份baseline副本明确标记BASELINE_FALLBACK。`optimized.load_kernels()`使用原`common.compile_ir(family,"optimized")`，`optimized.matmul`按原31点shape_to_family启动该组kernel；不会重新autotune，也不改变组映射。实际31调用和所有6份最终cubin与所选源字节一致，见[dispatch_final.json](results/runs/mm_opt_r1_20261118/dispatch_final.json)。原baseline.json、其绑定文件和全部baseline.ttir哈希保持不变。

## 逐方向结果与停止依据

- canonical：仅去loc；六组ELF可执行section哈希与baseline完全相同，作为A/A控制。完整cubin/PTX因debug位置可能不同，不写成全文件一致。
- unsigned：仅128×32×64组形成可接受重复收益；其他组无足够收益或回退。
- wrap_select：边界行列选合法0索引，数值通过，但寄存器上升、大组出现spill，初筛明显回退。
- k_aligned：K整除时无尾mask循环、否则原循环，数值通过的组也回退；大组shared达到294912 bytes，超过232448限制，编译/加载失败记录保留，不计时。
- cache_cg、unroll2：六组最终ELF可执行section均与baseline相同，不能把时间噪声称为优化。
- wrap16、wrap16_unsigned：在现有正16倍数前提下将越界fallback改为index&15，增加已证明合法的contiguous16提示；仍无有效改善，停止该方向。48份候选预算用完，不继续重复同类变体。

首轮候选/编译检查共47份候选可运行，1份资源失败。正确性累计2643次kernel用例（含重复的baseline和控制，不能当作独立shape数）；覆盖各组全部所属性能点、21项原边界/压力用例和K分支两侧。每次重置输出NaN sentinel，核查padding canary、有限性和输入不变，参考native ATen FP16、禁TF32 FP32，小尺寸CPU FP64，固定atol=1e-2、rtol=1e-3。

最终配置每组执行memcheck/racecheck，共58项/工具（去重后个别组少1项）；[memcheck](results/runs/mm_opt_r1_20261118/memcheck.log)为0 errors，[racecheck](results/runs/mm_opt_r1_20261118/racecheck.log)为0 hazards、0 errors、0 warnings。失败会阻止本轮接受。

## 计时有效性与完整证据

指标是CUDA graph kernel replay：相同输入、预分配输出、同一公共launcher，编译/加载/预热和Python launch在计时外。初筛12个冻结点每点3轮；最终完整31点主测7轮、独立seed=123进程5轮，同轮baseline/candidate随机顺序，rep=20ms。fallback组的全量控制是canonical对baseline；最终winner组另有candidate对baseline。

大组在包含重回退候选的初筛以及单独unsigned复查中出现漂移，外来compute进程检查未发现其他进程，但不能由此断言具体原因。最终主测无效控制点：[1664, 1920, 2048, 2560, 2688, 3200, 3328, 3456, 3584, 3840, 3968, 4096]；复测无效控制点：[1792, 1920, 2560, 2688, 3072, 3200, 3328, 3456, 3584, 3712, 3968, 4096]。每个点均保存原样、标记valid=false，不用于候选接受。有效候选组的主/复测所有要求点及A/A均通过。五组fallback表示“未建立可接受收益”，不是“已证明绝无优化空间”。对于不稳定组，收益状态是INCONCLUSIVE，不能用看似很窄的区间覆盖无效控制。

没有拼接快卡/快轮次，没有删除负结果。clock未锁定/核实锁频，无主动cache flush，不声称冷缓存或连续独占。只在GPU6（UUID c93dcf8b-2363-d1c2-02b4-4572cef79c41）、H100 SM90、指定venv官方Triton3.7.0/PyTorch2.12.0+cu130上验证；来源快照未做本次远程独立核验，沿用provenance.json中的限制。未测冷启动编译延迟，产物数量和hash可查，不能声称零编译成本。

原始证据均在[mm_opt_r1_20261118](results/runs/mm_opt_r1_20261118/)：pipeline.json（31点完整options/源码一致性/工具hash）、check/与followup_check/（每组C0/C1与可执行section hash）、screen*.json/followup_screen.json/big_group_recheck.json（含无效负结果）、full.json/repeat.json（配对样本）、reflection1.json和kernelforge_reflection.json（诊断转向）、finalists.json/selection.log、memcheck/racecheck与dispatch_final.json。全部候选生成器直接改TTIR；未改教程Python或下游IR/binary。

## KernelForge流程与复现

已按 [framework-workflow](/root/KernelForge/skills/framework-workflow/SKILL.md)、optimize-kernel、diagnose-kernel、benchmark-evidence及[review-agent](/root/KernelForge/skills/review-agent/SKILL.md)执行。KernelForge task确认matmul未注册执行adapter；使用本项目既有原生TTIR adapter，不宣称KernelForge CLI qualification。用户的实验要求和CUDA graph指标优先。计划审查API连接失败后依规定宿主只读fallback，round002计划approve；相关状态在 `/root/KernelForge-workspace/runs/mm-ttir-six-r1/advisors/`，反思也已登记。

计划和验收阈值见[optimization_round1.md](optimization_round1.md)。最终close review按相同流程在代码、证据完成后执行，审查记录为其独立状态来源。本报告的数值来自原始证据，不以审查意见代替测量。

```bash
# 先确认空闲设备；完整重放当前选择，无重新autotune
CUDA_VISIBLE_DEVICES=6 bash mm/test_optimized.sh
# 仅核查最终公共入口与所选TTIR/cubin对应；输出必须为新路径
CUDA_VISIBLE_DEVICES=6 PYTHONNOUSERSITE=1 LD_LIBRARY_PATH= \
 /root/venvs/ir-ops-dev/bin/python mm/check_round1_dispatch.py --output /root/tmp/mm-dispatch-new.json
```
