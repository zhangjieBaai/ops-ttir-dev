# logsumexp TTIR 第一轮实验报告

选中 `r1_trim`。完整 30 点几何平均加速 1.0297×，独立进程复测 1.0283×（A级，仅冻结矩阵）。

主测配对轮次 bootstrap 95% 区间 [1.02847203514028, 1.0305132222358757]；复测 [1.0275130190036463, 1.0298645680822933]。区间仅估计已冻结 shape 上的轮次波动，不代表应用分布或 shape 总体置信区间。

10 个 validation 点几何平均 1.0491× / 1.0478×；最差点 0.9880× / 0.9819×。

## 逐点结果

| M,N,K | 组 | baseline µs | candidate µs | 主测加速 | 复测加速 |
|---|---|---:|---:|---:|---:|
| 1x128x1 | g_c6892bb035fa | 1.522 | 1.521 | 1.0007× | 1.0006× |
| 1x1008x1 | g_4d6c54a25a3a | 1.673 | 1.673 | 1.0003× | 1.0005× |
| 1x4096x1 | g_2c3d9cf9cc2b | 1.923 | 1.906 | 1.0088× | 0.9996× |
| 1x8192x1 | g_6a72b2de463b | 2.161 | 2.161 | 0.9997× | 0.9996× |
| 1x32768x1 | g_df236e18c0a5 | 3.490 | 3.492 | 0.9996× | 1.0038× |
| 1x65536x1 | g_fc25ed1f09af | 10.064 | 10.067 | 0.9998× | 0.9980× |
| 64x128x1 | g_78effb105bfb | 1.737 | 1.721 | 1.0103× | 0.9915× |
| 64x1008x1 | g_38408436523c | 1.829 | 1.822 | 0.9996× | 0.9995× |
| 64x4096x1 | g_2672f7722871 | 2.174 | 2.174 | 1.0001× | 0.9991× |
| 64x8192x1 | g_505c3890a49d | 2.477 | 2.461 | 1.0000× | 0.9999× |
| 64x32768x1 | g_1c037e05ab35 | 4.018 | 4.018 | 0.9998× | 1.0000× |
| 64x65536x1 | g_493b7cfcff50 | 10.468 | 10.460 | 0.9995× | 1.0001× |
| 1024x128x1 | g_78effb105bfb | 2.294 | 2.293 | 1.0003× | 0.9997× |
| 1024x1008x1 | g_38408436523c | 2.619 | 2.614 | 1.0009× | 1.0003× |
| 1024x4096x1 | g_2672f7722871 | 5.277 | 5.257 | 1.0002× | 0.9976× |
| 1024x8192x1 | g_505c3890a49d | 6.867 | 6.866 | 0.9999× | 1.0014× |
| 1024x32768x1 | g_1c037e05ab35 | 33.231 | 33.217 | 1.0001× | 1.0001× |
| 1024x65536x1 | g_493b7cfcff50 | 56.385 | 56.382 | 1.0001× | 1.0000× |
| 1x128x64 | g_25df7a76052b | 2.675 | 2.270 | 1.1779× | 1.1785× |
| 1x512x64 | g_25df7a76052b | 2.776 | 2.392 | 1.1623× | 1.1536× |
| 1x2048x64 | g_25df7a76052b | 3.710 | 3.417 | 1.0895× | 1.0834× |
| 1x8192x64 | g_25df7a76052b | 7.019 | 7.101 | 0.9880× | 0.9819× |
| 16x128x64 | g_0b7ba13d19cb | 3.652 | 2.393 | 1.5256× | 1.5294× |
| 16x512x64 | g_0b7ba13d19cb | 4.053 | 3.865 | 1.0500× | 1.0547× |
| 16x2048x64 | g_97b63d5ad18c | 5.976 | 5.978 | 0.9997× | 0.9973× |
| 16x8192x64 | g_97b63d5ad18c | 12.556 | 12.568 | 0.9991× | 1.0010× |
| 256x128x64 | g_89a9edc29b16 | 7.716 | 7.726 | 1.0001× | 1.0005× |
| 256x512x64 | g_c3f1a9a2be29 | 12.971 | 12.956 | 1.0009× | 0.9998× |
| 256x2048x64 | g_c3f1a9a2be29 | 36.905 | 36.900 | 1.0001× | 1.0001× |
| 256x8192x64 | g_c3f1a9a2be29 | 126.389 | 126.389 | 1.0000× | 0.9999× |

## 边界与证据

唯一计算改动是 TTIR；原始 Python kernel、51 点映射、配置、grid、launcher 均冻结。基线来自用户指定 FlagGems logsumexp.py，原始提交及完整源码快照见 upstream/provenance.json。逐点源码/TTIR 的 LLVM、PTX、完整 cubin 字节一致，TTGIR 经相同 parser round trip 一致。

使用 upstream Triton 3.7.0 / PyTorch 2.12.0+cu130 / H100 SM90。文件入口从 make_ttgir 继续，经原生编译、加载和启动；无独立 llc、无二进制替换。RECORD 哈希核对证明本地安装一致，不是独立远程 wheel 来源认证。

计时为预分配输入输出的 CUDA graph replay 单 kernel 设备时间；7 轮主测、5 轮独立进程复测，rep=20ms，保存全部样本和顺序。无清缓存，无主动锁频；锁频配置未知。GPU 前后和运行中遥测均保存。GPU7 首次初筛发生其他进程干扰，整轮已弃用，见 round1_screen/invalid_timing.json；主测和复测均在同一物理 GPU0，不能与 GPU7 的绝对时间混比。

FP16 contiguous，axis=1；有限值/-inf，51 个 shape ×8 种分布；oracle 在 CUDA FP64 上执行并舍入到 FP16，并对照 CUDA native FP32 ATen。rtol=1e-3、atol=2e-3，输出 guard 与输入保留检查。不支持 +inf/NaN、空输入、任意 stride、其他 dtype 或完整 FlagGems API；未映射 shape 拒绝。

## 复现

```bash
CUDA_VISIBLE_DEVICES=0 bash logsumexp/test_accuracy.sh
CUDA_VISIBLE_DEVICES=0 bash logsumexp/test_performance.sh
```

`results/selection.json` 绑定候选和完整结果。每组 `optimized.ttir` 与所选不可变候选逐字节一致。资源变化、候选负结果、sanitizer 状态由随附 acceptance.json/补充说明记录；资源指标本身不构成性能因果证明。

## 候选与资源审计

20点/3轮有效初筛（B级）：i32 0.9995×、trim 1.0203×、two_pass 0.9541×、combined 0.9786×；第五个 mixed 候选0.9855×。这些数字不能代替最终30点结果。Two-pass在(1024,65536,1)回退至约0.641×；mixed最差约0.681×，均未采用。不同进程候选不作微小差距排名；mixed因同进程相对baseline的严重回退排除。详见 results/round1_disposition.json。

最终计时的17组中只有两个non-inner组的ELF `.text`改变；其余15组代码相同。inner性能应视为无实质收益。完整cubin因去loc调试信息变化并不相同，不能把 `.text`一致写成完整cubin一致。TILE_K=1组寄存器32→31，TILE_K=8组40→40；共享内存分别保持64/512字节，无spill。运行时N分支缩小逻辑张量工作量；没有profiler或严格消融证据证明寄存器变化是加速原因。

第二种子816项精度检查通过；memcheck/racecheck各204项（51 shape×2分布×baseline/candidate）通过。过滤器只检查logsumexp kernel；不宣称PyTorch oracle kernel已被sanitizer检查。正向计时探针将完整kernel执行两次并观察到约2倍时间，原始结果见 final_repeat_gpu0/timer_probe.json。拒绝未映射shape和不连续输入的检查见 results/refusals.json。

各轮绝对时间未跨GPU合并。主测/复测每点的ratio差异与A/A原始样本都保留，不把独立复测视为新shape holdout。validation shape在候选固定后首次暴露其性能，此后未再改候选。所有冻结baseline、原始候选、失败调用日志、整轮弃用数据均保留。完整接收证据索引见 results/acceptance.json。

全仓 `pre-commit run --all-files` 已执行：black/isort通过，flake8受并发其他实验 `mean/common.py` 的E402及 `mean/benchmark.py` 未使用导入阻塞；logsumexp范围全部pre-commit检查通过。未修改其他实验文件，日志见results/final_precommit_all.log与final_precommit_scoped.log。
