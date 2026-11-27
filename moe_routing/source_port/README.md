# MoE membership：TTIR 优化回写公开 Triton

将已验证的 group membership broadcast/reshape 忠实回写到公开 `@triton.jit`，保持原8点、配置、两kernel、workspace与device ABI。本组允许源码优化，旧纯TTIR实验保持原样。

- [冻结协议](optimization_plan.md)
- [冻结优化源码](membership_source_v1.py) 与 [哈希清单](source_manifest.json)
- [三路实验脚本](experiment.py)
- [报告](report.md) 与 [可读取汇总](summary.json)

三路分别使用原baseline TTIR、membership TTIR和**真实源码JIT返回的CompiledKernel**。源码导出再编译仅验证pipeline；计时没有把源码路径替换为文件入口。源码launcher保留原Python签名中的constexpr参数位置，底层device ABI与文件入口一致。

从仓库根运行，先确认GPU2空闲，每次指定新的结果目录：

```bash
CUDA_VISIBLE_DEVICES=2 /root/venvs/ir-ops-dev/bin/python moe_routing/source_port/experiment.py --run main_new --rounds 7 --rep 20 --seed 202611
CUDA_VISIBLE_DEVICES=2 /root/venvs/ir-ops-dev/bin/python moe_routing/source_port/experiment.py --run repeat_new --rounds 5 --rep 20 --seed 123
CUDA_VISIBLE_DEVICES=2 compute-sanitizer --tool memcheck --error-exitcode 99 /root/venvs/ir-ops-dev/bin/python moe_routing/source_port/experiment.py --run memcheck_new --check-only --seed 999
CUDA_VISIBLE_DEVICES=2 compute-sanitizer --tool racecheck --error-exitcode 99 /root/venvs/ir-ops-dev/bin/python moe_routing/source_port/experiment.py --run racecheck_new --check-only --seed 1000
```

每个实验先重验原baseline source/file与新优化source/file的LLVM/PTX/cubin一致性、三路完整options，再检查8点×6分布×3路径共144项精度/索引/tie/canary。主测/复测保存全量随机三路顺序、同轮耗时及三种配对比，含A/A和逐点输入seed/hash。

`membership_source_v1.py`已在导出前格式化并冻结，禁止再次修改。修正版必须用新文件名。`results/initial.log`保留初始harness的constexpr参数错位失败：该轮没有性能数据，修复仅作用于launcher，冻结源码不变；成功验证在`results/initial_v2/`。
