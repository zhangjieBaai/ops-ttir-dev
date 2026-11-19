# Sum 第二轮：30点等权平均加速超过10%

**目标达成：主测几何平均1.346768×，独立复测1.345741×；算术平均分别1.410644×、1.409556×。**

收益属于新实验组 `sum_launch_v2`：n≤16384直接由一个CTA完成归约，移除原第二次短kernel；其余15点保留原两阶段实现。它改变了启动序列与mid约定，**不是原固定启动契约下仅修改TTIR的34.7%收益**。严格两阶段TTIR尾部候选仅初筛约1.000429×，未晋升。

统计口径：沿用原30个shape逐点等权的配对加速比。若每个shape各调用一次，把所有耗时累加后再相比，主测仅1.060850×、复测1.060897×；大输入占据多数总时间，因此不能把34.7%外推成任意应用端到端收益。

## 冻结范围与对照

日期2026-11-19（北京时间）。FP32连续一维全量sum，输出FP32，原30点不增删；不覆盖dim、FP16/BF16、整数、非连续和空输入。源码及原baseline仍来自FlagGems commit `ef8aea78fed8321c3e02a15bf31e847cd0579e4f`。生产FlagGems未修改。

本轮协议：[optimization_plan_round2.md](optimization_plan_round2.md)。threshold=16384在候选计时前冻结，未根据逐点结果挑选路径；没有继续调整阈值。全部30点已在前轮和诊断中使用，本轮是固定工作负载的优化复核，**没有独立holdout**。seed123是独立进程复测，不是新shape泛化证据。

统一官方venv `/root/venvs/ir-ops-dev/bin/python`，Python3.12.3、Triton3.7.0、PyTorch2.12.0+cu130，H100 SM90。GPU4 UUID `GPU-7f7377cd-12e4-c549-c539-32bdeb268511`；所有主测、复测、sanitizer使用同卡。UUID规范化锁、约1秒一次其他PID/频率/功耗/温度采样均保留；未采样到同卡其他compute进程或监控异常，但不能排除采样间隙活动。

## 两条路线的区别

| 组 | 唯一/主要变化 | baseline与candidate | 结论 |
|---|---|---|---|
| strict r2_tail | stage1完整CTA无mask load，尾部保留mask；其余不变 | 原两阶段相同签名、grid、workspace、options | 83例×2通过，11点3轮B级初筛几何1.000429×，不足目标；未做完整收益验收 |
| sum_launch_v2 | 小输入从一份stage1 TTIR扩大张量宽度为next_power_of_2(n)，grid=1，第二指针改绑out，只启动该kernel | 原两阶段为每shape匹配reference，单CTA为新启动契约candidate；相同输入/out、同compiler/runtime/timer | 两轮完整30点均超过1.10×，选中 |

没有修改Python Triton kernel后重导出新算法，没有手改TTGIR/LLVM/PTX/cubin或注入binary。`make_round2.py`直接重写冻结TTIR，15个小shape各保留其真实签名和runtime参数选择；n=1的M已在源基线特化，只有两个运行时指针，CPU检查和设备精度均通过。

两版都从TTIR文件经同一个`compile_path`函数调用原生`triton.compile`，完整解析options一致：num_warps=4,num_ctas=1,num_stages=3,enable_fp_fusion=True及其余官方默认值，SM90 target一致。没有单独调warp数或改后端。文件入口从make_ttgir继续。所有两版实际cubin、resources和阶段hash保留。

本轮先逐30点重新核查原源码TTIR与文件baseline的LLVM/PTX/完整cubin字节一致及输出逐位一致。round2 compiler再核对其编译baseline全部阶段hash与该pipeline相同。新候选不要求与原baseline机器码相同，签名/映射/真实options逐点保留在[round2_manifest.json](round2_manifest.json)、[pipeline_v2.json](results/runs/r2_launch/pipeline_v2.json)。

## 精度、内存与同步检查

普通数值测试213个输入实例×baseline/candidate，共426次检查，包括所有30点随机输入、小输入每点全部压力分布和NaN/±Inf分类，以及原大输入压力用例。输出同时对照CPU FP64和native ATen FP32；两阶段路径检查所有中间部分和。单kernel路径mid不再是有用输出，明确检查其保持sentinel且完全未写；两条路径都检查输入不变、mid/out尾部canary、输出覆盖。

沿用误差标准 `abs(error) <= 2e-5 + 2e-6 * sum(abs(x))`，它是L1缩放标准，不能称为紧输出相对误差。小输入区域baseline最大绝对误差0.2580566，候选0.2419434（n=4096,offset）；全范围最大误差3001.1846313（n=67108864,offset），该点使用原两阶段，baseline和candidate相同。所有检查中误差/允许界最大约0.03150。[原始精度](results/runs/r2_launch/accuracy_v2.json)保留逐例值。

最终memcheck和racecheck各179个输入×两版=358次检查，覆盖全部30点随机和所有小shape压力/特殊值：[memcheck](results/runs/r2_launch/memcheck.log) 0 errors；[racecheck](results/runs/r2_launch/racecheck.log) 0 hazards、0 errors、0 warnings。未将未测的大shape额外压力sanitizer声称为通过。

## 正式计时与结果

无profiler，预分配输入/mid/out，完整算子设备序列全部进入同一个`triton.testing.do_bench_cudagraph`。旧版两个kernel，新版小输入一个kernel；编译、加载、分配和Python主机issue时间不计入。20次普通预热+100ms graph预热，7轮rep=20ms，随机配对顺序，保存每次replay值和A/A。两版都不清缓存、不主动锁频，实际时钟采样可查。

主测seed42和独立进程seed123均完整30点。先取每点7个配对比的中位数，再做30点等权算术/几何平均；95%区间按逐点配对轮次bootstrap2000次，它不包含跨日期/设备变异。

| 指标 | 主测 | 独立复测 |
|---|---:|---:|
| 几何平均加速比 | 1.346768× | 1.345741× |
| 算术平均加速比 | 1.410644× | 1.409556× |
| 最差点加速比 | 0.990986× | 0.986939× |
| 最好点加速比 | 2.026031× | 2.024113× |
| 几何平均95%区间 | [1.344249,1.348443] | [1.343651,1.348048] |
| A/A所有点中位比范围 | [0.988346,1.009525] | [0.990346,1.004086] |

主测最差n=2097152为0.990986×；复测最差n=65536为0.986939×。这两点走同一份原baseline的同一路径，因此局部差异在此次A/A变异附近，仍全部保留，没有删除慢样本或把相同机器码的噪声写成确定回退原因。

## 逐shape结果（µs）

| n | 原两阶段主测 | 候选主测 | 主测比 | 复测比 | 候选路径 |
|---:|---:|---:|---:|---:|---|
| 1 | 2.3029 | 1.1375 | 2.026031 | 2.024113 | single CTA |
| 15 | 2.3351 | 1.2058 | 1.932384 | 1.935898 | single CTA |
| 16 | 2.3665 | 1.2024 | 1.975433 | 1.983592 | single CTA |
| 31 | 2.3690 | 1.2233 | 1.937379 | 1.920840 | single CTA |
| 32 | 2.3996 | 1.2264 | 1.959648 | 1.943254 | single CTA |
| 63 | 2.4005 | 1.3591 | 1.761817 | 1.764355 | single CTA |
| 64 | 2.4364 | 1.3543 | 1.811101 | 1.821946 | single CTA |
| 127 | 2.3841 | 1.3578 | 1.749849 | 1.745768 | single CTA |
| 128 | 2.4368 | 1.3678 | 1.800126 | 1.812509 | single CTA |
| 255 | 2.5376 | 1.3802 | 1.858317 | 1.860676 | single CTA |
| 256 | 2.5872 | 1.3889 | 1.869923 | 1.842388 | single CTA |
| 511 | 2.4211 | 1.3949 | 1.725341 | 1.731973 | single CTA |
| 1024 | 2.6339 | 1.4093 | 1.875449 | 1.877939 | single CTA |
| 4096 | 2.7236 | 1.6159 | 1.687901 | 1.686999 | single CTA |
| 16384 | 3.0292 | 2.2555 | 1.342513 | 1.341500 | single CTA |
| 65535 | 3.0859 | 3.0855 | 1.000143 | 1.004802 | 原两阶段 |
| 65536 | 3.0993 | 3.0964 | 1.000746 | 0.986939 | 原两阶段 |
| 65537 | 3.1570 | 3.1251 | 1.010627 | 1.000152 | 原两阶段 |
| 262144 | 3.0583 | 3.0830 | 0.997450 | 1.001406 | 原两阶段 |
| 1048575 | 3.7273 | 3.7245 | 0.999657 | 1.001719 | 原两阶段 |
| 1048576 | 3.5089 | 3.5147 | 0.999207 | 1.001841 | 原两阶段 |
| 1048577 | 3.8173 | 3.7794 | 1.009726 | 0.995688 | 原两阶段 |
| 2097152 | 4.1214 | 4.1589 | 0.990986 | 0.998990 | 原两阶段 |
| 4194304 | 5.2954 | 5.2855 | 0.999360 | 0.999693 | 原两阶段 |
| 8388608 | 7.5123 | 7.5278 | 0.998883 | 1.001695 | 原两阶段 |
| 16777215 | 25.1607 | 25.1549 | 0.999841 | 1.000065 | 原两阶段 |
| 16777216 | 24.8469 | 24.8492 | 1.000048 | 1.000101 | 原两阶段 |
| 16777217 | 25.1934 | 25.1953 | 0.999536 | 0.999871 | 原两阶段 |
| 33554432 | 46.0756 | 46.0783 | 0.999978 | 0.999915 | 原两阶段 |
| 67108864 | 88.2401 | 88.2484 | 0.999926 | 1.000062 | 原两阶段 |

## 选择与复现

[selection_round2.json](results/selection_round2.json)按冻结目标选择r2_single16384。选择脚本重新核对当前TTIR/cubin与数值、pipeline、两轮timing、memcheck、racecheck的身份、协议/映射/driver哈希、同卡UUID和控制状态，避免旧选择脚本未来晋升的未绑定哈希风险。所有selected别名重新原生编译后与对应候选完整cubin字节一致；没有复制load metadata或替换binary。

新的别名位于`ir/launch_v2/selected/n<shape>/stage*.ttir`，新选择映射包含45个实际stage文件。旧`ir/<group>/optimized.ttir`和旧selection.json均保留前轮baseline，不静默改写历史。

```bash
CUDA_VISIBLE_DEVICES=4 bash sum/test_round2.sh
```

脚本先源码pipeline，再round2 pipeline、数值和完整30点计时，每次新目录保存结果。主入口`round2.py`只使用冻结映射中的30种输入。`make_round2.py --single`是不可变生成入口，manifest存在时拒绝覆盖；从现有工作目录复现不需要重导出baseline。

初版pipeline/accuracy日志保留，driver增加采样和artifact JSON规范化比较后，以`pipeline_v2.json`和`accuracy_v2.json`为实际验收文件。strict tail初筛未增加周期监控，只有A/A和前后设备状态证据，因此保留为B级探索结果，不用于正式晋升或声称全矩阵通过。

达到目标后停止搜索，没有额外做无必要的NCU或配置穷举。已使用KernelForge的[framework-workflow](/root/KernelForge/skills/framework-workflow/SKILL.md)、[optimize-kernel](/root/KernelForge/skills/optimize-kernel/SKILL.md)、[benchmark-evidence](/root/KernelForge/skills/benchmark-evidence/SKILL.md)、[review-agent](/root/KernelForge/skills/review-agent/SKILL.md)；plan审查已批准，新启动边界及工作负载复用均明确披露。最终审查记录归属 `/root/KernelForge-workspace/runs/sum-round2-10pct/`。
