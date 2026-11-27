**INT4 的已知 TTIR 优化已成功回写成普通 Triton 源码，且7个测试点的源码版与TTIR优化版产生完全相同的可执行 `.text`。**

同轮实测源码相对原基线的几何平均加速为 **1.0949×**，独立进程复测 **1.0955×**。直接比较TTIR版时间/源码版时间为 **1.0009× / 1.0008×**，即性能实际相当。

这次测量回答‘已知这一优化以后，直接写Triton能否复现’；没有比较只看源码与看TTIR的搜索能力。没有重新调优tile/warps/stages，也未修改原实验的任何冻结源码或IR。

| 路径与指标 | 主测7轮 | 独立复测5轮 |
|---|---:|---:|
| baseline / optimized TTIR | 1.093976× | 1.094748× |
| baseline / optimized source | 1.094896× | 1.095525× |
| optimized TTIR / optimized source | 1.000931× | 1.000816× |

下表为每个shape同轮中位时间（微秒），最后两列由每轮直接配对时间比取中位，不是用两个历史加速比相除。比值>1表示源码更快；微小偏离1不代表稳定的编译优势。

| M,N,K | baseline μs | IR μs | source μs | 主测 IR/source | 复测 IR/source |
|---|---:|---:|---:|---:|---:|
| (16, 4096, 4096) | 74.168 | 64.675 | 64.673 | 0.999947× | 1.000195× |
| (128, 11008, 4096) | 326.777 | 298.483 | 298.515 | 0.999975× | 1.000077× |
| (1, 256, 128) | 3.796 | 3.542 | 3.529 | 1.003349× | 1.003666× |
| (32, 1024, 1024) | 19.955 | 18.345 | 18.332 | 1.000627× | 1.000433× |
| (64, 2048, 4096) | 74.360 | 67.219 | 67.225 | 1.000055× | 0.998928× |
| (128, 4096, 4096) | 117.227 | 110.115 | 110.147 | 1.000038× | 1.000008× |
| (17, 80, 256) | 6.056 | 5.552 | 5.538 | 1.002531× | 1.002410× |

**源码具体改动及编译结果。** [source_v1.py](source_v1.py) 从原冻结设备函数回写：low/high nibble共享scale/zero，完整且对齐的K tile内group由k0/group标量计算再广播，已证明非负的除余经 `tl.uint32` 表达，必要时cast回int32保持后续索引语义。FP16反量化舍入、join/permute/reshape、dot及累积顺序保持原样。

源码路径确实经过普通 `@triton.jit` 前端，三路计时中的source对象是 `ASTSource` 产生的原始CompiledKernel；它没有被导出TTIR的重编译对象替换。源码对象使用完整Python调用参数，文件对象使用已特化的IR参数；两者最终CUDA参数ABI、grid和设备计算相同，计时不含Python启动成本。

前端保留共享与标量group、unsigned除余，正常执行了broadcast/CSE/canonicalize等前端处理。TTIR文本和完整PTX/cubin受源位置、调试信息等影响不逐字节相同；所有7点的ELF可执行 `.text` 字节哈希相同，寄存器/shared/spill也相同。这个结论比‘API能表达’更强：本轮固定环境下已经生成相同设备机器码。哈希与资源在 [static_analysis.json](static_analysis.json) 及两轮原始JSON中。

**环境、验证与性能边界。** upstream Triton3.7.0、PyTorch2.12.0+cu130、H100物理GPU0独占。原7点和冻结配置不变；FP16 A/scale/zero/C、uint8 packed、group128。每次先证明原baseline source/file与新source/export-file的LLVM/PTX/cubin各自字节一致、三路全部解析后的backend options一致，之后才验证和计时。[toolchain.json](toolchain.json)保存包RECORD检查、backend/libtriton/ptxas指纹；RECORD并非远程来源认证。

主/复测各105项版本/shape/分布检查通过：7点×random/extreme_nibble/zero_scale/zero/constant×3路。每个候选输出与baseline逐位一致，对FP32 ATen参考沿用rtol=.003、atol=.035，两个小shape另对CPU FP64，检查NaN sentinel覆盖、64元素尾canary及输入不修改。memcheck/racecheck各再跑105项数值检查，过滤本实验kernel，分别 **0 errors** / **0 errors、0 warnings**。

每轮三路随机顺序测量，另有A/A；每次20ms预分配CUDA graph重放，保留全部样本、执行顺序、输入seed/SHA256。主/复测所有点A/A中位比都在[0.97,1.03]内。未锁频、未主动清cache；不包含分配、JIT、模块加载或Python启动，不外推调用端到端成本。

直接IR/source的固定shape配对轮次bootstrap95%区间：主测 `[1.0006215622535417, 1.0015726176429995]`，复测 `[0.9997078421066025, 1.001987569553077]`。相同机器码下仍有约千分之一的计时差，可能包括测量次序、模块地址等本轮未隔离因素；不能据此宣称源码特有优化。

首个 `main` 尝试在pipeline门槛处退出：错误地把源码TTIR与文件解析后的打印文本逐字节比较。修正为源码TTIR与冻结输入文件比较，继续严格验证LLVM/PTX/cubin后，完整主测记录为 `main_v2`。失败日志保留；冻结source_v1和原IR没有修改。

复现命令见 [README.md](README.md)。证据：[主测](results/main_v2/results.json)、[复测](results/repeat/results.json)、[memcheck](memcheck.log)、[racecheck](racecheck.log)、[机器可读汇总](summary.json)。
