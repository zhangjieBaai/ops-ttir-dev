# INT4 GEMM TTIR 实验

[当前完整30点报告](report.md) · [最终组合消融](report_ablation.md) · [消融协议](ablation_plan.md)

冻结基线本身融合 INT4 解包、group128 反量化与 GEMM。当前最终版为 `multiround/r6/ir/<shape>/optimized.ttir`，保留第三轮 `group128`；根目录 `ir/optimized.ttir` 是历史首轮别名。最终选择与哈希见 [multiround/r6/selection.json](multiround/r6/selection.json)。源码与TTIR都通过官方Triton3.7.0原生后端，不使用独立llc。

## 测试入口

从仓库根目录运行，先用 `nvidia-smi` 选择空闲H100，每张卡仅运行一个GPU实验。环境固定为 `/root/venvs/ir-ops-dev/bin/python`。

```bash
export CUDA_VISIBLE_DEVICES=0

# 所有30点、10条消融路径、5种分布；显式pipeline后再完整正确性
bash int4_gemm/test_accuracy.sh my_accuracy

# 完整30点主测：先pipeline、全部正确性，再7轮配对计时
bash int4_gemm/run_experiment.sh my_main 42 7

# 独立进程复测，先确认所选卡空闲，使用新名字
bash int4_gemm/run_experiment.sh my_repeat 123 5
```

每次输出在 `ablation/results/<新会话名>/`，同名路径存在会退出。pipeline结果保存为同目录 `<会话名>_pipeline.json`。每个benchmark先完成全部1500项版本/分布检查才计时；原始样本、顺序、A/A、环境、输入及IR/下游哈希和资源随结果保留。生成器只直接修改冻结TTIR，`final`对全部30点必须与第六轮正式alias字节相同，运行不会重新选择或改变该alias。

核心入口：

- [ablation/generate.py](ablation/generate.py)：再生逐项加入和最终移除的不可覆盖候选。
- [ablation/verify_pipeline.py](ablation/verify_pipeline.py)：核对30点源码/文件入口LLVM/PTX/cubin、完整options及输出。
- [ablation/run.py](ablation/run.py)：复用已验证的30点数值及CUDA graph计时协议。
- [ablation/render.py](ablation/render.py)：审核本次`v2_main`/`v2_repeat`原始JSON，重算主报告、消融报告和[全部shape CSV](ablation/results/v2_all_shapes.csv)。
- [run_experiment.sh](run_experiment.sh)：当前最终版消融复现入口，包含pipeline和全量正确性。

## 实验契约

H100/SM90；A/scale/zero连续FP16，packed为`[N,K/2]` uint8，低nibble对应偶K；group=128，K为128倍数，N为16倍数，M允许尾部，无别名。FP32反量化后舍入至FP16再dot，FP32累积，FP16输出。固定[30点逐shape配置](round2/expanded/manifest.json)，不重新autotune。

主指标为预分配CUDA graph中的单个fused kernel；排除数据构造、参考、分配、编译、加载和Python launch，不代表冷缓存或生产端到端。主测7轮、复测5轮，rep20ms；不同GPU独立测同场配对比，不跨卡比较绝对时延。数值参考和归因边界见[当前报告](report.md)。

## 结果与历史

- [本次最终版消融](report_ablation.md)：逐项加入、窄加载/K mask的2×2对照、最终版移除、完整30点主测/复测。
- [首轮报告](report_round1.md)与[首轮协议](optimization_plan.md)：7点的`scalar_unsigned_v1`，不是当前最终版。
- [第二轮扩展](round2/report.md)、[第三至五轮](multiround/report.md)、[第六轮负结果及最终选择](multiround/r6/report.md)。
- [首轮Triton源码回写实测](source_port/report.md)：仅覆盖首轮组合；当前最终组合未进行源码性能回写，不证明TTIR独占收益。
- [第六轮memcheck](multiround/r6/memcheck.log)、[racecheck](multiround/r6/racecheck.log)：正式路径零错误；本次新增中间消融没有另跑sanitizer。

冻结 `tutorial.py`、`upstream/`、baseline和已有候选均保留。`artifacts/`是可再生编译产物，Git忽略；原始证据保存在`results/`。
