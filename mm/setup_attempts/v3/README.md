# MM TTIR baseline

目前仅建立并运行官方教程 baseline，尚未开始优化。契约见 [optimization_plan.md](optimization_plan.md)。

使用官方 Triton v3.7.0 tutorial 03 的 FP16 MM，原始计算体不变。完整快照、来源及 SHA256 在 [upstream/provenance.json](upstream/provenance.json)。网页的 main 版本可能继续变化，本实验固定 v3.7.0，与指定编译器版本一致。当前网络无法重新获取源码，沿用本机已有官方快照并核对其历史来源 hash，未宣称本次进行了独立远程校验。

只在 `1024³` 上执行一次官方 16 配置 autotune，选择后冻结；整个 shape 矩阵共享该配置。这是固定配置实验组，不代表教程逐 shape autotune 的最佳性能。冻结记录在 `baseline.json`，包含全部初始选择时间；已有 baseline 的运行只校验，不重新调参或覆盖。

编译流程为：教程源码 → make_ttir → 冻结 `ir/baseline.ttir` → `triton.compile(文件)` → make_ttgir/LLVM/PTX/cubin → 原生 CompiledKernel。产物写 `artifacts/`，hash/资源与完整环境记录保留在每个结果目录中。没有候选，也没有 `optimized.ttir`。

从仓库根目录运行（先用 `nvidia-smi` 确认设备空闲）：

```bash
CUDA_VISIBLE_DEVICES=6 bash mm/test_accuracy.sh
CUDA_VISIBLE_DEVICES=6 bash mm/test_performance.sh
```

性能入口依次校验冻结 baseline、pipeline 一致性、完整正确性、31 个官方方阵 shape（M=N=K=256…4096，步长128）× 5 轮 CUDA graph kernel 计时、独立进程 seed=123 正确性与计时、memcheck/racecheck。每次写独立 `results/runs/` 目录，失败日志也保留。显式清空外部 Python 和动态库路径；不会把 JIT、加载、分配计入 kernel 时间。时钟未锁定，未主动清缓存。

本轮限定 FP16、正 16 倍数 M/N/K 和行 stride、列 stride=1、指针对齐、无别名；支持行尾 padding。转置、融合激活、其他 dtype、空矩阵或不满足 IR 前提的输入不在本组范围内。31 个性能 shape 及数值协议在契约和 common.py/validate.py 中。

已完成实验和数值、性能结果见 [baseline_report.md](baseline_report.md)。正式结果来自 GPU 6；GPU 7 的受干扰运行另行保留并标记，不参与本轮报告。入口在每个 GPU 进程前后检查已有 compute process，发现占用则终止；这不是系统级设备预约或连续的独占证明。

另行完成了官方31个shape逐点autotune的TTIR数量统计，见 [autotune_ttir_report.md](autotune_ttir_report.md)：本次获选6份不同TTIR，全部候选共16份（11份可执行、5份shared-memory超限）。该统计不替换上述固定配置基线。
