# INT4 GEMM 最终组合消融协议

固定第二轮冻结的 30 个 `(M,N,K)`、逐点 tile/warps/stages、FP16 输入输出、uint8 nibble packing、group=128 及原始容差。最终目标是第六轮仍保留的 `r3/group128`，不重新搜索或修改 optimized 别名。`ablation/generate.py` 只变换冻结 TTIR；每点生成的 `final` 必须与第六轮 `optimized.ttir` 字节相同。

逐项加入为 baseline → parent（首轮共享/标量化/unsigned 组合）→ narrow → narrow_full_k → group_pair → final。另测 full_k 构成窄加载/K mask 的 2×2 对照。最终移除包括：group_pair（去 group128 右移）、no_full_k（恢复 K mask）、no_pair（去组内复用）、no_narrow_pair（联合移除窄加载及组内复用）。组内复用的构造依赖窄加载，所以联合移除不能称为窄加载的纯独立贡献。BK128 的组内复用步骤是恒等对照。

本轮只研究当前已采用的组合；首轮共享/标量化/unsigned 合并为一个起点，不把旧搜索数字当单项独立贡献。所有条件效应存在交互，不能把百分比相加。

官方 `/root/venvs/ir-ops-dev/bin/python`、Triton 3.7.0、H100，先运行 `ablation/verify_pipeline.py` 核对 30 点源码/文件入口 LLVM/PTX/cubin 和输出。每次 runner 先编译所有路径，再对全部 30×10×5 分布完成正确性才开始性能。对照 FP32 ATen，小 N 另用 CPU FP64；要求候选与 baseline 逐元素完全一致（torch.equal）、输入不变及 canary 完好。

主测 GPU0，seed42，7 轮、rep20ms；独立进程复测 GPU1，seed123，5 轮、rep20ms。每张卡串行运行自己的 pipeline/正确性/benchmark，两卡互不共享计时结果。每版本每 shape 稳定100ms；随机测量顺序，记录所有同轮时间、配对比值、A/A、输入/IR/下游哈希和资源。每点 A/A 中位数须在 [0.97,1.03]。只测预分配 CUDA graph 单 fused kernel，排除主机/分配/编译，不代表冷缓存或端到端。

使用 `CUDA_VISIBLE_DEVICES=<空闲卡> bash int4_gemm/run_experiment.sh <新的会话名> [seed] [rounds]` 复现。输出到 `ablation/results/<会话名>/`，已存在则退出。报告重算不改变原始结果。已有第六轮对正式路径的 memcheck/racecheck 保留引用；本轮新增中间消融未运行 sanitizer 时明确写出。
