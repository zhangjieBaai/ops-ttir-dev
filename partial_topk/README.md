# Partial Top-K TTIR 实验

[当前完整30点结果](report.md) · [最终版逐项消融与两阶段交叉](report_ablation.md) · [消融契约](ablation_plan.md)

采用LayerNorm的报告顺序：契约、编译链/数值、完整主复测、优化过程、逐项加入与最终移除、失败与复现。当前最终版为r6 `merge_heads_packed_v2`，见[最终选择和哈希](multiround/r6/selection.json)及`multiround/r6/ir/<group>/{local,merge}/optimized.ttir`。根`ir/optimized.ttir`/`selection.json`属于首轮历史。

## 测试入口

从仓库根执行，先用`nvidia-smi`选择空闲卡并独占；固定官方Triton3.7.0环境。

```bash
export CUDA_VISIBLE_DEVICES=7

# 再生10条消融路径，已有文件必须字节一致
/root/venvs/ir-ops-dev/bin/python partial_topk/make_ablation.py

# 全部30点/两个stage的pipeline和压力正确性，无性能采样
/root/venvs/ir-ops-dev/bin/python partial_topk/ablation_benchmark.py --run my_accuracy --check-only

# 正式主测7轮和独立复测5轮，20ms，生成消融报告
RUN_ID=my_topk_recheck bash partial_topk/run_experiment.sh
```

`--check-only`使用min(M,2)行并对照CPU FP64稳定排序；正式计时前normal额外检查完整M。每次先核验源码/文件baseline LLVM/PTX/cubin和options，再完成全部版本数值检查，随后完整两kernel序列计时。每点A/A中位数必须在[0.97,1.03]；输出ID必须全新。

`make_ablation.py`冻结算法链和baseline/仅最终local/仅最终merge/最终两stage的2×2；`ablation_benchmark.py`执行；`report_ablation.py --main <ID> --repeat <ID>`重算比值并核对验证/哈希后生成报告、CSV、JSON；`report.py`从已验收的最终版消融样本重建主报告，包含每shape时间、方法增益和算术/几何平均。共用基础设施在`../complex_suite/`，实验协议和数据归本目录。

## 证据和解释范围

[新消融IR、manifest、原始数据](ablation_v1/) · [第六轮性能和sanitizer](multiround/r6/report.md)。计时只包括两个kernel的预分配CUDA graph，不含分配、编译或CPU launch；后端产物位于已忽略的`artifacts/`。

有限连续FP32，K8/16、chunk1024。index严格精确、value数值精确，±0等价。merge依赖局部列表有序；不能用于任意workspace。当前头部归并没有源码回写性能实测，首轮源码等价结论不外推。中止的首轮新消融因计时parent别名冲突被排除，见[弃用记录](ablation_v1/rejected_run.json)；正式结果使用修正后的`split_v2_20261124_main/repeat`。

## 历史

[第一轮6点](report_round1.md) · [来源与首轮合同](optimization_plan.md) · [第一轮源码回写](source_port/report.md) · [第二轮30点](round2/report.md) · [第三至五轮](multiround/report.md) · [第六轮](multiround/r6/report.md)
