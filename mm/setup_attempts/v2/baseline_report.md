# MM baseline 已跑通；未开始优化

正式证据：[GPU6 完整运行](results/runs/baseline_20261118T124859Z_Vh0qLn/run.log)。本轮仅官方源码和相同 TTIR 的原生重编译，不存在优化候选或加速比。

环境为指定 venv、Python 3.12.3、Triton 3.7.0、PyTorch 2.12.0+cu130，NVIDIA H100 80GB HBM3 / SM90，物理 GPU 6，UUID `c93dcf8b-2363-d1c2-02b4-4572cef79c41`。来源固定官方教程 v3.7.0；本机快照 SHA256 与历史来源记录一致，本次网络获取失败，未作独立远程来源认证。

配置由代表点 1024³ 的官方 16 配置 autotune 选出并冻结：`BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, GROUP_M=8, warps=4, stages=4, CTAs=1`。其他测试点共享配置，不代表逐 shape 最优。初始选择和全部时间保留于 [baseline.json](baseline.json)；5 个配置返回 Infinity，后续只编译/加载诊断，全部确认 shared-memory OutOfResources，见 [autotune_failures.json](results/runs/baseline_20261118T124859Z_Vh0qLn/autotune_failures.json)，未重调参。

源码 → 冻结 TTIR → triton.compile(TTIR) → 原生 CompiledKernel 已完整跑通。源码与文件入口完整 target/options 相同；LLVM IR、PTX、cubin 字节一致；TTGIR 经相同 parser round trip 后一致；代表输入输出逐位一致。文件入口从 make_ttgir 开始，不重新运行 Python 前端或 make_ttir。wheel RECORD 和实际 Python/Triton/libtriton 路径检查通过，加载的 torch/c10/cudart 均在 venv 内；动态库全路径、后端/libtriton/ptxas/libdevice hash 均见 [pipeline.json](results/runs/baseline_20261118T124859Z_Vh0qLn/pipeline.json)。

- 源码 SHA256：`50ccd205e79d5275b8da09b3bc72cf86b0dc95bc2ef4531548509226b0fcc7b3`
- TTIR SHA256：`af7b105bfb513f99fb5662f5a3ee0f385ebdb1a9ebed671934e7113c1bed7ee1`
- cubin SHA256：`a9d7a79bfb26e22870e81782fa1825e96ba2cf7cdc29df8628e00bfb0bd7ddd2`
- 资源：124 registers，0 spills，98304 bytes shared memory。

正确性首轮 seed=42 和独立进程 seed=123 各 36 个用例通过（共 72 次）；含性能矩阵、最小尺寸、tile/K/group 边界、padding、零/常数/交替符号/大/小输入。输出 sentinel、canary 和输入不修改检查均通过。测试矩阵内与 native ATen FP16 的最大绝对误差为 0；与 FP32/CPU FP64 参考的最大绝对误差可达 0.25，符合预先冻结的 `atol=1e-2, rtol=1e-3`，不能写成“绝对误差都小于 1e-2”。CPU FP64 仅覆盖小型用例，完整误差逐项保存于 accuracy*.json。

memcheck 与 racecheck 各执行 5 个 padding 边界用例，退出状态均为 0；[memcheck](results/runs/baseline_20261118T124859Z_Vh0qLn/memcheck.log) 为 0 errors，[racecheck](results/runs/baseline_20261118T124859Z_Vh0qLn/racecheck.log) 为 0 hazards / 0 errors / 0 warnings。

性能口径为同一 GPU6、预分配输出的 CUDA graph replay kernel 时间：每 shape 5 轮、rep=20ms，另用独立进程 seed=123 重复。全部 150 个时间样本在 benchmark*.json，不包含分配、Python launch、JIT 和模块加载。两轮逐点中位数最大相对偏差 1.28%。未锁定/核实锁频，未主动清缓存；设备进程检查和状态快照不能证明整个时间窗不存在短暂外部干扰，不宣称完全受控或冷缓存。

| M × N × K | 首轮中位数 µs | 独立复测 µs | 首轮 TFLOPS |
|---|---:|---:|---:|
| 128 × 128 × 128 | 2.812 | 2.786 | 1.49 |
| 256 × 256 × 256 | 3.224 | 3.224 | 10.41 |
| 384 × 384 × 384 | 3.618 | 3.621 | 31.30 |
| 512 × 512 × 512 | 4.076 | 4.045 | 65.86 |
| 768 × 768 × 768 | 4.935 | 4.936 | 183.58 |
| 1024 × 1024 × 1024 | 6.024 | 6.023 | 356.51 |
| 1536 × 1536 × 1536 | 17.731 | 17.766 | 408.76 |
| 2048 × 2048 × 2048 | 36.697 | 36.996 | 468.15 |
| 3072 × 3072 × 3072 | 139.460 | 137.939 | 415.76 |
| 4096 × 4096 × 4096 | 330.017 | 331.347 | 416.46 |
| 256 × 1024 × 512 | 4.111 | 4.058 | 65.30 |
| 1024 × 256 × 768 | 4.911 | 4.909 | 81.99 |
| 128 × 2048 × 512 | 4.071 | 4.069 | 65.94 |
| 2048 × 128 × 1024 | 5.626 | 5.629 | 95.43 |
| 272 × 400 × 528 | 4.594 | 4.609 | 25.01 |

失败与异常均保留：

- `baseline_20261118T124552Z_xb6jlW` 在 pipeline 签名检查失败：检查脚本错误地假设 SSA 参数叫 `%arg`，实际是源变量名。v1 manifest/检查脚本存于 setup_attempts/v1；修复仅签名检查，manifest 显式升级 v2，源码、TTIR、配置、数值测试与计时均未修改。
- `baseline_20261118T124700Z_gO2bUH` 完成正确性与 sanitizer，但 GPU7 的独立性能复测中途出现另一个 LayerNorm benchmark。后半段显著变慢，整组性能标记不接受，见 performance_status.json，不与 GPU6 结果混合。其正确性仍按原始记录保留。
- GPU6 完整重跑从 pipeline/正确性重新开始，未只挑选某张卡较快的 shape 组合。

复现入口：`CUDA_VISIBLE_DEVICES=6 bash mm/test_performance.sh`，先确认卡空闲。新增入口在每个 GPU 进程前后拒绝已有 compute process；该检查不等于系统级资源预约。后续优化等待用户指示。
