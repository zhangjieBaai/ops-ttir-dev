# Norm＋动态INT8 TTIR 实验

[当前完整30点结果](report.md) · [最终版逐项消融](report_ablation.md) · [消融契约](ablation_plan.md)

沿用LayerNorm的组织方式：先读契约，再看主测/独立复测及逐shape结果，再看逐项加入、从最终版移除和交互。当前最终版仍为第二轮`reciprocal_early`；r3–r6未晋升。当前IR与哈希在[第六轮选择](multiround/r6/selection.json)和`multiround/r6/ir/<group>/optimized.ttir`。根`ir/optimized.ttir`/`selection.json`是首轮历史。

## 测试入口

仓库根执行，先用`nvidia-smi`选择空闲卡，独占设备；官方环境固定为`/root/venvs/ir-ops-dev/bin/python`。

```bash
export CUDA_VISIBLE_DEVICES=6

# 再生只编辑TTIR的消融；已有文件必须字节相同
/root/venvs/ir-ops-dev/bin/python norm_quant/make_ablation.py

# 全部30点pipeline和数值检查，不做性能采样
/root/venvs/ir-ops-dev/bin/python norm_quant/ablation_benchmark.py --run my_accuracy --check-only

# 主测、独立进程复测与报告；每个输出ID必须全新
RUN_ID=my_norm_recheck bash norm_quant/run_experiment.sh
```

`--check-only`把各分布行数缩到min(M,3)，并保留CPU FP64对照；正式计时前另跑normal完整M。主测7轮/20ms/seed42，复测5轮/20ms/seed123；每个进程先核验源码与文件baseline LLVM/PTX/cubin/options，再完成全部版本的数值检查，随后计时。每点A/A中位数必须在[0.97,1.03]。结果目录存在会退出，不覆盖旧数据。

核心文件：`make_ablation.py`生成9条路径并核对final与r6逐组一致；`ablation_benchmark.py`运行独立消融；`report_ablation.py --main <ID> --repeat <ID>`核对全部哈希/验证/原始比值后生成报告、CSV、汇总；`report.py`从已验收的最终版消融样本重建主报告，包含每shape时间、方法增益和算术/几何平均。共用编译与计时基础设施在`../complex_suite/`，实验协议和数据归本目录。

## 结果与范围

[消融原始数据、IR和manifest](ablation_v1/) · [第六轮原始样本与sanitizer](multiround/r6/report.md)。`artifacts/`为忽略的可再生后端产物。

连续有限FP16输入，FP32 Z/scale和INT8 Q，无别名。Z精确，scale对FP64 rtol3e-5/atol1e-9，Q最多差1 code并限制重构误差；不是INT8逐位等价。只计单fused kernel。当前最终版未再次做源码回写性能测试；首轮源码证据只适用首轮。

## 历史

[第一轮8点](report_round1.md) · [来源与首轮合同](optimization_plan.md) · [第一轮源码回写](source_port/report.md) · [第二轮30点](round2/report.md) · [第三至五轮](multiround/report.md) · [第六轮](multiround/r6/report.md)
