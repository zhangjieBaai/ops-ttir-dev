# Sum 直接 TTIR 优化实验

当前统一展示见[完整结果](report.md)：每个shape只列基线时间、优化时间、加速比，保留算术/几何平均和方法收益。

逐项收益、组合消融和公开Triton等价实现见[消融报告](report_ablation.md)。
本轮全部30点重新配对测量，最终收益主要来自移除第二个短kernel；其余单项未产生整体正收益。
可运行的公开Triton封装见[ablation_example.py](ablation_example.py)。

第二轮达到30点等权平均加速超过10%的目标：**几何平均主测1.347×、独立复测1.346×**，
算术平均分别1.411×、1.410×。n≤16384改用一个CTA、一个kernel，其余shape保留原两阶段。
这是单列的 `sum_launch_v2` 新启动契约，不能称为原固定启动契约下的纯TTIR收益。
若按30种shape各调用一次的累计耗时计算，比值约1.061×；不代表任意应用端到端收益。
见[第二轮报告](report_round2.md)、[新选择映射](results/selection_round2.json)。

```bash
CUDA_VISIBLE_DEVICES=4 bash sum/test_round2.sh
```

旧版入口和别名保留历史语义：首轮**未发现可靠加速，最终保留 baseline**。5 个直接 TTIR 候选全部通过精度检查；
2 个候选完成 30 shape × 7 轮主测及独立进程复测，几何平均均约 1.000×。
完整结果、逐点耗时和误差见 [report.md](report.md)。

后续 NCU / Nsight Systems 诊断与优化优先级见 [diagnosis.md](diagnosis.md)。
诊断沿用冻结 kernel，没有生成新候选或变更当前选择。

研究范围为指定 FlagGems `sum.py` 的 FP32 连续输入全量两阶段归约，
不是 `sum_dim`、其他 dtype 或生产代码集成。30 个性能 shape 已冻结在
[optimization_plan.md](optimization_plan.md)，实际 shape→两阶段 TTIR 映射见
[baseline.json](baseline.json)。两阶段共 39 组；同组 baseline/候选共享完整编译配置、签名和 grid。

使用官方 Triton 3.7.0，固定 venv，运行时从 TTIR 文件 JIT 编译、原生加载及启动。
性能是预分配输出的**两个 kernel 总 CUDA graph replay 设备时间**。
GPU 7 初始计时因另一实验并发而整轮弃用，最终证据均在 GPU 6 独占测量，
迁移记录见 [protocol.json](results/runs/gpu6_baseline/protocol.json)。

在仓库根目录复现（先确认所选 GPU 空闲）：

```bash
CUDA_VISIBLE_DEVICES=6 bash sum/test_accuracy.sh
CUDA_VISIBLE_DEVICES=6 bash sum/test_performance.sh
```

脚本各自新建结果目录，不覆盖旧证据。`optimized.ttir` 是每组 baseline 的字节副本，
选择理由及哈希在 [selection.json](results/selection.json)。不可变候选和所有负结果均保留。
基线已冻结；不要重复运行 `export_baseline.py` 覆盖它。

源码快照和来源在 `upstream/`；`make_candidates.py` 只改 TTIR。
`verify_pipeline.py` 逐 shape 验证源码/文件链一致性；`validate.py` 验证中间结果与输出；
`benchmark.py` 交错配对并保留所有 replay 样本和 A/A 对照。
`select.py` 按预注册门槛选择候选；当前没有候选满足两轮几何平均均 >1.01× 的门槛。
