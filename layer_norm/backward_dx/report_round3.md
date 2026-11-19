# backward dX 第三轮：中心化代数与联合规约

本轮6个新候选，晋升family：[1024, 2048]。`optimized.ttir`仅更新通过主测、独立复测和数值/sanitizer检查的组。前向实验与所有原始baseline未修改。

baseline是官方Triton3.7.0教程的dX段提取，双方不做dW/dB或锁；结果不能称为完整fused backward的加速。FP16输入输出、FP32统计量与累加，N/stride是16倍数且N≤16384，无别名。

## 最终分组结果（A级）

| family | 最终候选 | 主测 / parent | 独立复测 / parent | 主测 / baseline |
|---|---|---:|---:|---:|
| b1024 | r3_centered_stream | 1.0320× | 1.0531× | 1.0756× |
| b2048 | r3_centered_stream | 1.0690× | 1.0916× | 1.0937× |
| b4096 | baseline | 1.0000× | 1.0000× | 1.0002× |
| b8192 | baseline | 1.0000× | 1.0000× | 1.0000× |
| b16384 | joint_coefficients | 1.0000× | 1.0000× | 1.0368× |

表中未晋升组的parent比值按保留版本写1；baseline列为该组最终选择在同会话的实测值。主测7轮rep20ms，独立复测seed123/5轮rep10ms，M4096、N1024..15872步长512，共30点。组级接受门槛：对parent几何≥1.02、最差≥0.95，并且对原baseline几何≥1.02，主测和复测均须通过。

| 测量 | 最终组合 / baseline | 最终组合 / parent | parent / baseline | A/A |
|---|---:|---:|---:|---|
| 主测 | 1.0268× | 1.0055× | 1.0212× | 30/30通过 |
| 独立复测 | 1.0286× | 1.0076× | 1.0211× | 30/30通过 |

最终组合由同一会话内各shape已冻结选择的候选记录汇总；对parent为各kernel耗时中位数之比，每条原始计时仍与共同baseline配对，不伪称直接parent/candidate交错配对。别名的IR/cubin身份由最终pipeline检查验证。

## 有效方向与限制

将规约从sum(xhat*wdy)改为sum((x-mean)*wdy)，把rstd移到行级alpha系数；仍显式中心化和FP32运算，输出使用rstd³*c1与rstd*c2。另加DX的cs store。b1024/b2048的寄存器仍为32、零spill、2处静态CTA barrier；静态SASS的FMUL从25减至19，store变为STG.E.EF.128。这是编译产物线索，未把全部收益归因于某一条指令或cache属性。大N不能继承小N收益。

## 未采用的方向

- r3_rawmoment规约sum(x*wdy)，再减mean*sum(wdy)：常规135例通过，但新增72例中18例失败；其他候选和baseline在同一saved stats参考下均通过。存在抵消误差，不放宽容差，不做性能筛选。
- r3_joined把两个向量join为[B,2]再规约：b16384升到237寄存器；未表现出优于parent的初筛整体收益。
- r3_group_center1024/2048：b16384降到128/126寄存器，零spill；2048版大组主测对parent仅1.0019×、复测0.9996×，未通过晋升。整30点相对baseline主测0.9850×、复测0.9857×。单点N8704收益不能代表整个family。
- 第一份screen_batch1在N8704的A/A=1.0413，整份不用于晋升；保留原件，补跑screen_batch1_repeat后全部A/A通过。

## NCU瓶颈证据（C级）

固定base clock、cache-control=all，NCU仅用于诊断；其耗时不混入CUDA graph性能结果。下面是parent→group_center2048：

| N | registers | achieved occupancy | DRAM读取MB | long-scoreboard / issue | NCU µs |
|---:|---|---|---|---|---|
| 8192 | 80→64 | 34.89→45.17% | 134.29→136.01 | 8.29→11.55 | 63.17→70.40 |
| 8704 | 220→126 | 12.30→23.04% | 142.68→142.70 | 1.48→1.86 | 95.42→87.78 |
| 15872 | 220→126 | 12.23→22.36% | 260.15→260.75 | 2.80→5.43 | 132.90→136.99 |

分组中心化避免了第二轮纯重载的DRAM读取近翻倍，但N15872占用率提升后仍变慢，long-scoreboard指标增加；指令、加载依赖和调度仍可能抵消资源收益，尚未完成单因素因果拆分。

小N的NCU未复现graph收益：N1024 parent→centered_stream为9.728→9.984µs，N2048为15.936→16.384µs。NCU为base clock且每次清cache，性能协议是反复CUDA graph执行；两者条件不同。不能宣称冷cache或实际单次调用获得相同加速，静态FMUL减少仍只是机制线索。原始数据见results/round3/ncu_stream_summary.json。

## 正确性与编译路径

首批810次、第二批540次常规检查通过；额外48边界/版本，额外压力72例/版本；独立seed123再验证。压力测试原先用autograd重算stats，导致baseline和parent也有17例不一致；原件保留，新增显式aten.native_layer_norm_backward传入同一mean/rstd，同时比较FP64 saved-stats公式，未改容差。常规135例继续要求原native autograd参考。

memcheck/racecheck分别覆盖30个边界与padding用例/版本，均零错误；候选使用不可变名称和TTIR哈希。原始数值指标全部保留，不声称高rstd梯度的绝对误差统一小于0.01。

```text
baseline.ttir  ─┐
                ├─ triton.compile(.ttir, 同组相同options)
optimized.ttir ─┘     → make_ttgir → make_llir → PTX → Triton自带ptxas → cubin
                      → CompiledKernel._init_handles → 相同CUDA launcher/相同grid与参数
```

官方upstream Triton3.7.0固定venv，无FlagTree；TTIR文件入口从make_ttgir继续。五组各自固定warps/stages等，不改grid/ABI/options，不使用独立llc、手改低层IR或内联PTX。最终pipeline要求每组baseline LLVM/PTX/cubin与冻结教程提取源码一致，并验证最终候选IR与cubin身份。

证据：[主测](results/round3/full_stream.json)、[独立复测](results/round3/repeat_stream.json)、[最终选择](results/round3/selection_final.json)、[分组失败决策](results/round3/decision.json)、[NCU](results/round3/ncu_summary.json)、[计划](optimization_plan_v3.md)。

## 最终别名直接复核

更新后的optimized入口直接跑完整30点、3轮rep10ms：几何1.0282×原baseline，30/30 A/A通过。这是最终入口的补充复核；组级晋升依据仍是前述7轮主测与5轮独立复测。
[直接复核记录](results/round3/final_selected_benchmark.json)、[最终编译链](results/round3/pipeline_final.json)、[最终别名正确性](results/round3/accuracy_selected.json)。
