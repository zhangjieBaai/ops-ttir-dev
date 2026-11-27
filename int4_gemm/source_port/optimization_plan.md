本轮按用户明确要求，将既有 `scalar_unsigned_v1` 忠实回写公开 Triton 源码。这是独立源码对照组，覆盖旧轮“候选只能改TTIR”的范围限制；不修改旧 baseline/候选/结果，不调优配置。

三路：baseline=旧冻结TTIR，ir=旧scalar_unsigned_v1 TTIR，source=本目录source_v1.py经过普通JIT实际产生的CompiledKernel。源码导出TTIR及再编译只用于pipeline核验，不替代源码路计时。全部7点、grid、实际CUDA参数ABI、输入/输出、配置和数值合同沿用原轮；主机只分配和启动。

源码表达共享scale/zero、完整K tile的group标量化、已证明非负除余转uint32。保留FP16舍入、join/reshape/dot及累积次序。源码格式化后冻结，首轮最多一个忠实实现；若修复编译/语义错误必须新版本名保存旧版。

每次主/复测先执行原baseline源码与文件LLVM/PTX/cubin核验、新source与其导出文件同链核验、三路完整options一致。数值先于性能：原7点×5分布，CPU小例、canary、输入不改；候选与baseline逐位一致，FP32参考rtol=.003/atol=.035。

三路每轮随机顺序交错，共7轮主测与独立5轮复测、每次20ms CUDA graph；每轮两个baseline构成A/A。保存每次时间和顺序，逐轮直接计算baseline/IR、baseline/source、IR/source，逐点中位后几何平均。A/A每点中位比要求0.97–1.03。预分配输出、同输入，计时输入seed与hash明确，不混入历史数字。官方Triton3.7.0固定venv、H100 GPU0独占，未锁频/清缓存。source最终memcheck/racecheck。即使源码性能不同也完整报告，无择优隐藏。
