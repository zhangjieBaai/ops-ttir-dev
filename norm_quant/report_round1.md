# Residual RMSNorm + 动态 INT8：直接 TTIR 实验

最终采用 `stream_v1`。冻结8点几何平均：主测 **1.0548×**，独立进程复测 **1.0521×**；最差点分别 **0.9883× / 0.9859×**。性能证据为A，仅限本矩阵和本研究baseline。

融合研究baseline一次完成残差加法、RMSNorm、amax、INT8量化，并写出FP32 residual、scale和INT8。它参考FlagGems结构，但不是生产kernel原样提取；每个点只在4/8 warps中预调优，未穷尽配置。候选把max(abs(rho*u))改为rho*max(abs(u))，将sum(z²)/max(abs(u))合为tuple reduction，再对输出使用cs cache hint；单kernel、ABI/grid/options不变。

| shape | baseline μs | candidate μs | 主测 | 复测 |
|---|---:|---:|---:|---:|
| (1, 4096) | 2.968 | 2.858 | 1.0364× | 1.0361× |
| (128, 1024) | 2.219 | 2.097 | 1.0583× | 1.0603× |
| (128, 4096) | 3.695 | 3.606 | 1.0222× | 1.0246× |
| (2048, 4096) | 29.695 | 27.791 | 1.0686× | 1.0699× |
| (512, 8192) | 15.985 | 13.388 | 1.1941× | 1.1718× |
| (513, 4097) | 14.645 | 14.463 | 1.0132× | 1.0120× |
| (32, 16384) | 8.076 | 8.162 | 0.9883× | 0.9859× |
| (64, 257) | 2.238 | 2.078 | 1.0697× | 1.0663× |

时间为样本中位数，加速比为配对比值中位数，两者不必恰好相除。

## 协议与证据

固定官方 Triton3.7.0 / PyTorch2.12.0+cu130 / H100。基线先从冻结源码导出，再与文件入口核对LLVM/PTX/cubin字节一致；每次主/复测在计时前重复核验，实际完整backend options一致。所有候选从TTIR文件编译，未改TTGIR/LLVM/PTX/cubin。

主测7轮、独立进程复测5轮、每轮20ms；配对内随机次序、每轮A/A。预分配完整设备序列CUDA graph时间，不含编译、分配和Python launch；cache未清理、时钟未锁定。每shape保存计时输入seed与SHA256。初筛后冻结单一候选，不按独立复测择优。

固定shape、重采样配对轮次的95% bootstrap区间：主测 `[1.0540127834869335, 1.0561889960576514]`，复测 `[1.0506641106664159, 1.0528457296864264]`。它只反映这些样本的波动，不表示跨模型/设备的不确定性。

主证据：[main_v2](results/main_v2/results.json)、[repeat_v2](results/repeat_v2/results.json)、[manifest](manifest.json)、[选择和哈希](selection.json)。memcheck **0 errors**，racecheck **0 errors/0 warnings**，只过滤检查本实验kernel；[memcheck日志](results/logs/memcheck_final.log)、[racecheck日志](results/logs/racecheck_final.log)。

初版main/repeat也保留；审阅发现它们的timing tensor在seed初始化前分配，配对公平但无法按所记seed完整重建输入。本报告改用已修复seed并保存输入哈希的main_v2/repeat_v2，没有把旧样本混入平均。

## 数值、资源与负结果

正式主/复测共224项版本/分布检查通过；另保留各次初筛与sanitizer内检查。所有输出有canary、sentinel及输入不修改检查。

| shape | regs baseline→candidate | spill baseline→candidate | shared B baseline→candidate |
|---|---|---|---|
| (1, 4096) | 56→56 | 0→0 | 4096→4096 |
| (128, 1024) | 30→29 | 0→0 | 32→64 |
| (128, 4096) | 56→56 | 0→0 | 4096→4096 |
| (2048, 4096) | 56→56 | 0→0 | 4096→4096 |
| (512, 8192) | 93→93 | 0→0 | 4096→4096 |
| (513, 4097) | 120→117 | 0→0 | 4096→4096 |
| (32, 16384) | 156→170 | 0→0 | 4096→4096 |
| (64, 257) | 28→28 | 0→0 | 2048→2048 |

资源是C级静态证据；Top-K表列local阶段，merge资源在原始JSON。不把寄存器或同步减少直接解释为全部性能收益。

量化合同预先允许Q相对FP64至多差1 code、scale rtol=3e-5/atol=1e-9，另验证重构误差；**不声称INT8逐位等价**。Z精确、Q显式限制在[-127,127]，因此-128 sentinel不能蒙混过关。覆盖normal/zero/constant/offset/cancellation/tiny/signed-weight，小rows另有CPU FP64。

4种候选的完整初筛保留在 `results/screen/results.json`（B）：因式分解、联合统计、消去rho、输出cache hint；stream收益是组合结果。N=16384、rows32约1.4%回退仍保留，未用逐点挑选掩盖。未涵盖FP8、BF16、任意非连续layout、NaN/Inf。初次setup曾错误比较metadata的cache key，修正为完整backend options比较；失败日志保留，没有放过LLVM/PTX/cubin一致性门槛。

## 复现

先查看 `nvidia-smi` 并设置独占 `CUDA_VISIBLE_DEVICES`，Python固定 `/root/venvs/ir-ops-dev/bin/python`。在仓库根执行下列命令，输出目录必须不存在：

```bash
CUDA_VISIBLE_DEVICES=6 /root/venvs/ir-ops-dev/bin/python norm_quant/experiment.py \
  --run my_recheck --names baseline,stream_v1 --rounds 5 --rep 20 --seed 234
```

脚本先自动执行pipeline/数值门槛，再计时；`--check-only`只验证。候选生成器对已有IR做字节一致性断言，拒绝覆盖不同内容。已冻结manifest时不要再次`--setup`。

新增的实际源码回写比较见 [source_port/report.md](source_port/report.md)，其中同轮比较原基线、TTIR优化版与正常Triton源码编译版。
