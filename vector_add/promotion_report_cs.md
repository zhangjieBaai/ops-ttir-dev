# 当前正式结果：optimized.ttir 与 baseline.ttir

当前 `optimized.ttir` 使用 `r1_store_cs`，唯一 kernel 改动是 TTIR store 的
`cacheModifier=cs`。用户明确接受此前约 1.206x 的局部收益，授权采用该版本。
原 v1 的整体 1.02x 门槛仍然记为未通过；本次是明示的选择政策修订。

`optimized.ttir` SHA256：
`ddd8ec671f57abe0e0538e70b6e38dcdfc3f494ec33b73b87073cb3fe6e25c0e`。
baseline、教程源码、BLOCK_SIZE=1024、四个 warps、grid=ceil(N/1024)、
运行时 ABI、FP32/对齐/无别名契约、完整编译选项和 launcher 均保持不变。
旧 optimized 文件和 selection 记录原样归档在
`results/runs/promote_cs_20261119/history/`。

GPU 7 正被其他实验占用。本轮正式测量使用 GPU 0，并在同一物理 GPU 上做
baseline/optimized 随机交错对比和独立进程复测。GPU 0 UUID：
`GPU-cc248ebf-79d0-42d7-da65-99d1ef6f4ed1`。历史 GPU 7 的约 1.206x 是
另一物理设备上的结果，不能与本轮样本合并或承诺精确重现。

测量通过实际 `ir/optimized.ttir` 文件入口编译与启动。主测 9 轮、复测 7 轮，
rep=20ms，两个不同 seed，全部官方 16 个长度，逐点 A/A 控制，预分配输出、
CUDA graph replay，无 profiler；所有样本与实际 IR/cubin 哈希均保存。
数值检查使用两个进程、全部 36 个用例、原 FP32 位级比较与 canary/refusal。
缓存未主动清理，时钟未锁定，结果不等于教程默认 do_bench 或应用端到端时间。

正式测量已完成，两个 36-case 数值运行均通过，主测和复测的全部 A/A 控制通过。
旧 sanitizer 证据与当前 optimized 的 TTIR 和完整 cubin SHA256 均完全一致，
据此复用 GPU 7 上 memcheck/racecheck 各 23 个用例的零错误记录；这不记作
新的 GPU 0 sanitizer 执行。核对记录见 `sanitizer_reuse.json`。

| 统计 | 主测 9 轮 | 独立复测 7 轮 |
|---|---:|---:|
| 几何加速比 | 1.00935x | 1.01239x |
| 95% 配对轮次 bootstrap 区间 | [1.00761, 1.01094] | [1.00977, 1.01457] |
| 算术加速比 | 1.00998x | 1.01338x |
| N=2^22 | 1.15390x | 1.19596x |
| 最差点 | N=65536，0.99511x | N=2097152，0.99889x |

局部收益的幅度在两个进程间存在差异，报告保留两次实测，原因尚未隔离。
区间只反映保存的配对轮次，不能覆盖这种跨进程/地址/缓存状态的全部变化。
这里不将历史约 1.206x 作为当前 GPU 0 的精确结果。

以下全是实际 `optimized.ttir` 与 baseline 的正式对比。时间是配对轮次均值的
中位数，单位微秒；加速比是轮次内 baseline/optimized 比值的中位数。

| N | baseline µs（主测） | optimized µs（主测） | 主测加速比 | 复测加速比 |
|---:|---:|---:|---:|---:|
| 4096 | 1.2521 | 1.2525 | 0.99970x | 0.99997x |
| 8192 | 1.2952 | 1.2948 | 1.00057x | 0.99979x |
| 16384 | 1.3045 | 1.2898 | 0.99976x | 1.00025x |
| 32768 | 1.3142 | 1.3164 | 0.99922x | 0.99930x |
| 65536 | 1.3779 | 1.3847 | 0.99511x | 1.00190x |
| 131072 | 1.5371 | 1.5243 | 1.00850x | 1.00014x |
| 262144 | 1.7037 | 1.7041 | 1.00033x | 1.00052x |
| 524288 | 1.9996 | 2.0031 | 1.00106x | 1.00006x |
| 1048576 | 2.8130 | 2.8142 | 0.99834x | 1.00323x |
| 2097152 | 4.6556 | 4.6487 | 1.00095x | 0.99889x |
| 4194304 | 16.9969 | 14.7306 | 1.15390x | 1.19596x |
| 8388608 | 34.7316 | 34.7102 | 1.00055x | 1.01225x |
| 16777216 | 67.3399 | 67.3113 | 1.00039x | 1.00056x |
| 33554432 | 132.0866 | 132.0246 | 1.00039x | 1.00041x |
| 67108864 | 261.5779 | 261.4725 | 1.00039x | 1.00037x |
| 134217728 | 521.1164 | 520.8021 | 1.00057x | 1.00050x |

本轮原始证据：

- [主测](results/runs/promote_cs_20261119/full.json)
- [独立复测](results/runs/promote_cs_20261119/repeat.json)
- [编译链核查](results/runs/promote_cs_20261119/pipeline.json)
- [正确性主测](results/runs/promote_cs_20261119/accuracy_main.json) 与 [复测](results/runs/promote_cs_20261119/accuracy_repeat.json)
- [sanitizer 精确身份复用](results/runs/promote_cs_20261119/sanitizer_reuse.json)
- [旧选择归档](results/runs/promote_cs_20261119/history/selection.json)
- [当前选择](results/selection.json) 与 [用户授权后的协议](promotion_plan_cs.md)


本目录 black/isort/flake8 和 scoped pre-commit 均通过。
已按要求运行全仓 `pre-commit run --all-files`，该次因另一项正在进行的
`layer_norm/backward_dx/report.py` 工作存在四条 flake8 错误而未通过；
未改动该任务文件。完整日志和本目录通过日志在本轮结果目录保存。
