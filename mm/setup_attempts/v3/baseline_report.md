# MM 官方 31 点 baseline；未开始优化

性能 shape 已按用户要求严格对齐官方教程：**M=N=K=256…4096，步长128，共31个方阵点**。原文位于 [教程快照第411行](upstream/03-matrix-multiplication.py.txt#L411)，`x_vals=[128 * i for i in range(2, 33)]`。export_baseline.verify_extraction 从该快照读取实际 x_vals，与 PERF_SHAPES 逐项核对。本轮没有矩形或128³性能点。

正式证据：[完整运行日志](results/runs/baseline_20261118T130140Z_RWeNZh/run.log)。实验标识 `mm-fp16-sm90-fixed-v3-official31`，GPU6 H100 80GB HBM3 / SM90，UUID `c93dcf8b-2363-d1c2-02b4-4572cef79c41`；指定 venv、Triton 3.7.0、PyTorch 2.12.0+cu130。

按要求更新测试矩阵及对应正确性覆盖，源码、TTIR、编译配置、launcher、计时方法保持不变，不重新 autotune、不生成优化候选。配置仍为代表点1024³选择的 `BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, GROUP_M=8, warps=4, stages=4, CTAs=1`；本轮只对齐官方 shape，不代表逐 shape autotune 最优。源码快照来自本机已有 v3.7.0 官方快照，本次未完成独立远程来源核验，来源限制见 upstream/provenance.json。

- 源码 SHA256：`50ccd205e79d5275b8da09b3bc72cf86b0dc95bc2ef4531548509226b0fcc7b3`
- TTIR SHA256：`af7b105bfb513f99fb5662f5a3ee0f385ebdb1a9ebed671934e7113c1bed7ee1`
- cubin SHA256：`a9d7a79bfb26e22870e81782fa1825e96ba2cf7cdc29df8628e00bfb0bd7ddd2`
- 资源：124 registers，0 spills，98304 bytes shared memory。

源码/TTIR文件 pipeline 核查通过：完整 target/options 一致，LLVM IR、PTX、cubin 字节一致；TTGIR 相同 parser round trip 后一致；代表输入输出逐位一致。环境路径、wheel RECORD、动态库映射和工具 hash 保存在 [pipeline.json](results/runs/baseline_20261118T130140Z_RWeNZh/pipeline.json)。TTIR文件入口从 make_ttgir 继续，不重复 Python 前端。

首轮 seed=42 与独立进程 seed=123 各 **52 个正确性用例通过**，覆盖全部31性能点及21个边界、padding、压力用例。与 native ATen FP16 的观测最大绝对误差为0；与 FP32/CPU FP64 的最大绝对误差可达0.25，按冻结的 `atol=1e-2, rtol=1e-3` 判定通过，不代表所有绝对误差小于1e-2。CPU FP64仅检查小尺寸。memcheck/racecheck各5个边界用例通过，分别为0 errors、0 hazards / 0 errors / 0 warnings。

每点5轮、rep=20ms的 CUDA graph kernel 计时，独立进程 seed=123 重复，共 **310个原始时间样本**。预分配输出，编译、加载、Python launch和分配不计入时间。全部样本见 [benchmark.json](results/runs/baseline_20261118T130140Z_RWeNZh/benchmark.json) 与 [benchmark_repeat.json](results/runs/baseline_20261118T130140Z_RWeNZh/benchmark_repeat.json)；两轮逐点中位数最大相对偏差 2.78%。时钟未锁定/核实锁频，未主动清缓存，进程前后检查不等于连续独占证明。这里只报告 baseline，不计算优化加速比。

| M=N=K | 首轮中位数 µs | 独立复测 µs | 首轮 TFLOPS |
|---:|---:|---:|---:|
| 256 | 3.220 | 3.220 | 10.42 |
| 384 | 3.606 | 3.607 | 31.41 |
| 512 | 4.072 | 4.075 | 65.92 |
| 640 | 4.492 | 4.514 | 116.72 |
| 768 | 4.923 | 4.924 | 184.03 |
| 896 | 5.536 | 5.536 | 259.89 |
| 1024 | 5.981 | 5.982 | 359.07 |
| 1152 | 9.542 | 9.277 | 320.43 |
| 1280 | 10.061 | 10.069 | 416.90 |
| 1408 | 11.646 | 11.598 | 479.37 |
| 1536 | 17.836 | 17.833 | 406.34 |
| 1664 | 21.718 | 21.787 | 424.29 |
| 1792 | 26.051 | 26.195 | 441.79 |
| 1920 | 31.519 | 31.741 | 449.12 |
| 2048 | 36.583 | 36.734 | 469.61 |
| 2176 | 49.630 | 49.928 | 415.20 |
| 2304 | 58.499 | 58.352 | 418.15 |
| 2432 | 67.929 | 67.474 | 423.51 |
| 2560 | 78.428 | 78.266 | 427.84 |
| 2688 | 93.524 | 93.698 | 415.33 |
| 2816 | 104.552 | 104.578 | 427.17 |
| 2944 | 116.429 | 116.025 | 438.31 |
| 3072 | 136.990 | 136.542 | 423.26 |
| 3200 | 153.919 | 152.055 | 425.78 |
| 3328 | 172.254 | 170.530 | 427.97 |
| 3456 | 200.153 | 197.444 | 412.47 |
| 3584 | 215.789 | 213.765 | 426.68 |
| 3712 | 241.436 | 241.334 | 423.69 |
| 3840 | 284.566 | 284.765 | 397.96 |
| 3968 | 307.332 | 306.567 | 406.57 |
| 4096 | 327.281 | 326.919 | 419.94 |

旧15点结果保留在原 results/runs 目录，不并入本轮31点结果。旧报告、manifest、common.py及契约归档在 [setup_attempts/v2](setup_attempts/v2/)，新 manifest 记录父 hash 和改动原因。更早的签名检查失败、GPU7受干扰复测及5个 autotune shared-memory OutOfResources 记录也保持原样。

复现：先确认GPU空闲，再运行 `CUDA_VISIBLE_DEVICES=6 bash mm/test_performance.sh`。
