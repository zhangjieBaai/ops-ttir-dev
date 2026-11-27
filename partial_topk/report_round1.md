# 大 N partial Top-K：直接 TTIR 的循环状态重写

最终采用 `ordered_carry_v3`。冻结6点几何平均：主测 **1.0827×**，独立进程复测 **1.0816×**；最差点分别 **1.0242× / 1.0155×**。性能证据为A，仅限本矩阵和本研究baseline。

固定两阶段研究baseline：每1024输入保留K候选，再合并成全局Top-K；沿用FlagGems分块重复选择思路，未全排序。输出value/原始index，值降序，同分按较小index优先。候选把FP32 score映射为单调uint32 key，以unsigned max归约，并将编码移出K循环、key作为loop-carried SSA值保留；min-index tie规则保留。kernel数、workspace、ABI/grid和4warps/3stages不变，计时覆盖两kernel。此baseline配置固定，未逐点autotune。

| shape | baseline μs | candidate μs | 主测 | 复测 |
|---|---:|---:|---:|---:|
| (1, 65536, 8) | 7.802 | 6.832 | 1.1420× | 1.1415× |
| (16, 65536, 8) | 11.458 | 10.561 | 1.0855× | 1.0925× |
| (128, 65536, 8) | 45.946 | 44.863 | 1.0242× | 1.0155× |
| (16, 262144, 8) | 27.122 | 25.983 | 1.0439× | 1.0579× |
| (16, 262144, 16) | 52.787 | 49.081 | 1.0757× | 1.0548× |
| (4, 65537, 8) | 8.431 | 7.459 | 1.1298× | 1.1326× |

时间为样本中位数，加速比为配对比值中位数，两者不必恰好相除。

## 协议与证据

固定官方 Triton3.7.0 / PyTorch2.12.0+cu130 / H100。基线先从冻结源码导出，再与文件入口核对LLVM/PTX/cubin字节一致；每次主/复测在计时前重复核验，实际完整backend options一致。所有候选从TTIR文件编译，未改TTGIR/LLVM/PTX/cubin。

主测7轮、独立进程复测5轮、每轮20ms；配对内随机次序、每轮A/A。预分配完整设备序列CUDA graph时间，不含编译、分配和Python launch；cache未清理、时钟未锁定。每shape保存计时输入seed与SHA256。初筛后冻结单一候选，不按独立复测择优。

固定shape、重采样配对轮次的95% bootstrap区间：主测 `[1.0823705798047045, 1.0831040625325354]`，复测 `[1.0563513698092808, 1.0934704411250469]`。它只反映这些样本的波动，不表示跨模型/设备的不确定性。

主证据：[main_v2](results/main_v2/results.json)、[repeat_v2](results/repeat_v2/results.json)、[manifest](manifest.json)、[选择和哈希](selection.json)。memcheck **0 errors**，racecheck **0 errors/0 warnings**，只过滤检查本实验kernel；[memcheck日志](results/logs/memcheck_final.log)、[racecheck日志](results/logs/racecheck_final.log)。

初版main/repeat也保留；审阅发现它们的timing tensor在seed初始化前分配，配对公平但无法按所记seed完整重建输入。本报告改用已修复seed并保存输入哈希的main_v2/repeat_v2，没有把旧样本混入平均。

## 数值、资源与负结果

正式主/复测共192项版本/分布检查通过；另保留各次初筛与sanitizer内检查。所有输出有canary、sentinel及输入不修改检查。

| shape | regs baseline→candidate | spill baseline→candidate | shared B baseline→candidate |
|---|---|---|---|
| (1, 65536, 8) | 30→40 | 0→0 | 16→16 |
| (16, 65536, 8) | 30→40 | 0→0 | 16→16 |
| (128, 65536, 8) | 30→40 | 0→0 | 16→16 |
| (16, 262144, 8) | 30→40 | 0→0 | 16→16 |
| (16, 262144, 16) | 30→40 | 0→0 | 16→16 |
| (4, 65537, 8) | 30→38 | 0→0 | 16→16 |

资源是C级静态证据；Top-K表列local阶段，merge资源在原始JSON。不把寄存器或同步减少直接解释为全部性能收益。

有限FP32的value数值精确、indices严格精确；**±0等价，不要求保留零符号位**，baseline的max结果本来就不保证选中index处的零位型。v3先归一化±0以保持同分index规则。已选值/padding使用-inf的有序key，小于所有finite key；尾块不足K也不能胜过真实输入。主测扩展到正/负subnormal及最低有限值，另含normal/全tie/整数tie/±0/递增；小rows对CPU FP64稳定排序。

负结果：联合value/index argmax导致全6点约0.810×；仅改有序max但每轮重算key约0.980×。两者在旧main完整7轮有B级证据，见 `results/main/results.json`；joint/local_joint/cg消融在screen。v3的改变是循环状态表示，不能把全部收益简单归因于少一次reduce。未实施packed64候选；未测试NaN/Inf、不同K、不同chunk配置或动态workspace协议。

## 复现

先查看 `nvidia-smi` 并设置独占 `CUDA_VISIBLE_DEVICES`，Python固定 `/root/venvs/ir-ops-dev/bin/python`。在仓库根执行下列命令，输出目录必须不存在：

```bash
CUDA_VISIBLE_DEVICES=7 /root/venvs/ir-ops-dev/bin/python partial_topk/experiment.py \
  --run my_recheck --names baseline,ordered_carry_v3 --rounds 5 --rep 20 --seed 234
```

脚本先自动执行pipeline/数值门槛，再计时；`--check-only`只验证。候选生成器对已有IR做字节一致性断言，拒绝覆盖不同内容。已冻结manifest时不要再次`--setup`。

新增的实际源码回写比较见 [source_port/report.md](source_port/report.md)，其中同轮比较原基线、TTIR优化版与正常Triton源码编译版。
