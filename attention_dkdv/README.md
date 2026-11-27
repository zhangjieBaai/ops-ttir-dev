# Attention dK/dV TTIR 实验

[最终版逐项消融](report_ablation.md)：全部32点主测/独立复测，D128逐项加入、最终移除与结构交互；D64单项dV后移。实验契约在 [ablation_plan.md](ablation_plan.md)，原始样本、资源、哈希及CSV在 `results/ablation_v2/`。

官方 Triton v3.7.0 fused-attention 教程冻结提取的 dK/dV 子算子，从 `.ttir` 文件入口经同一官方原生后端编译和同一 launcher 执行。候选只修改 TTIR。当前有效实现按四组分别保存在 `multiround/r6/ir/<group>/optimized.ttir`，选择与哈希见 [r6 selection](multiround/r6/selection.json)；根 `ir/<group>/optimized.ttir` 是历史首轮版本，不应当作当前最终版。

## 测试入口

命令均从仓库根目录执行。先用 `nvidia-smi` 选择空闲 H100；每卡独占，benchmark 与 sanitizer 不并发。固定官方 `/root/venvs/ir-ops-dev/bin/python`，不使用系统Python或FlagTree运行GPU实验。

```bash
# 当前最终版及完整消融：pipeline → 全部数值验证 → 32点配对主测
bash attention_dkdv/run_experiment.sh 4 20 7 attention_dkdv/results/runs/main_new

# 独立进程、新输入seed复测；设备5须空闲
bash attention_dkdv/run_experiment.sh 5 123 5 attention_dkdv/results/runs/repeat_new

# 从原始证据重建报告、CSV及带哈希的汇总（新目录）
/root/venvs/ir-ops-dev/bin/python attention_dkdv/report_ablation.py \
  --main attention_dkdv/results/runs/main_new/measurement \
  --repeat attention_dkdv/results/runs/repeat_new/measurement \
  --output attention_dkdv/results/runs/summary_new
```

SM锁为1200MHz，入口脚本在退出时恢复自动频率。已有输出目录会报错；不会覆盖历史数据、冻结源码或baseline。`make_ablation.py`确定性生成两组新增rows64+dV对照，相同名字只接受相同内容。

- `round2/verify_pipeline.py`：32点源码/文件入口LLVM、PTX、cubin、选项和输出一致性。
- `ablation_benchmark.py`：先44输入/176项数值检查，全部通过后运行预分配CUDA graph随机交错计时、A/A和实际CUDA event预热；输出环境、全部样本及哈希。
- `report_ablation.py`：核查完整矩阵、数值门槛、A/A、主复测IR与下游哈希相同，生成逐项加入/移除、静态资源与全部shape表。
- `multiround/r6_run_locked.py`：历史第六轮搜索、完整验证与sanitizer入口；本轮补测不重选候选。

## 当前结果

[最终优化报告](report.md)保留第六轮32点正式结果：几何均值主测1.7998×、独立复测1.7996×（A 级），并解释D64与D128采用不同结构的原因。当前补测的完整数字见 [report_ablation.md](report_ablation.md)，与原正式结果分别标明来源。

[首轮12点与负结果](report_round1.md) · [第二轮扩展32点](round2/report.md) · [第三至五轮](multiround/report.md) · [第六轮](multiround/r6/report.md) · [首轮公开Triton源码回写](source_port/report.md)。源码可表达、机器码一致和性能一致是不同结论；首轮源码性能不能代替r6源码实测。

## 实验契约与限制

H100 / SM90；FP16连续Q/K/V/dO/dK/dV，FP32 LSE(log2)/delta；无别名、有限输入，N为128倍数，D64或128，4warps/5stages/1CTA。32点为每组8个固定(B,H,N)，含N128/256/512/1024/2048及B/H扩展；另测zero_dO、constant、large压力。参考门槛固定为FP64数学 `rtol=0.03, atol=0.008`，候选另须与原基线逐位一致。

只计已经准备好预缩放K、LSE与delta之后的dK/dV kernel，不含forward、delta准备、dQ、分配、编译或Python launch；是重复warm-cache graph设备时间。原始官方固定配置未经逐shape autotune，不称最强baseline，不外推完整训练step、其他dtype、dropout、ragged序列或其他GPU。

第六轮冻结最终路线memcheck为0 errors、racecheck为0 errors/0 warnings；新增rows64_dv仅完成本轮正确性/性能消融，未另跑sanitizer。来源与SHA见 [upstream/provenance.json](upstream/provenance.json)，历史首轮合同见 [optimization_plan.md](optimization_plan.md)。冻结源码/上游不得格式化；可再生产物在忽略的`artifacts/`，原始数值/性能与哈希纳入实验记录。
