# MM baseline 实验契约 v3（官方 31 点性能矩阵）

当前仅跑通基线，**不开始 TTIR 优化，不生成候选或 optimized.ttir**。

- 来源：官方 Triton v3.7.0 tutorial 03。保存完整快照，按原始行提取 CUDA 配置工厂、leaky_relu、matmul_kernel；只分离 autotune 包装，不改 kernel。AST 核查计算体与原文一致；来源限制见 upstream/provenance.json。
- 环境：/root/venvs/ir-ops-dev/bin/python，官方 Triton 3.7.0、H100 SM90；检查实际模块路径、wheel RECORD、动态库映射、完整编译 options、工具哈希和 GPU UUID。入口清空 LD_LIBRARY_PATH、PYTHONPATH、PYTHONHOME，记录实际环境。
- 单一配置分组：只在代表点 M=N=K=1024 用官方全部 16 个配置执行一次 autotune。获选配置和测量记录冻结到 baseline.json，其余全部 shape 沿用。不是逐 shape autotune 最佳性能，不与旧 LLIR 实验跨口径比较。
- 算子：C=A@B，无 activation；A/B/C FP16、FP32 累加，输出 FP16；独立无别名分配、行主序、列 stride=1。M/N/K 和行 stride 为正 16 倍数，指针对齐至少 16 字节；支持行尾 padding=16。所有尺寸、stride 和最大元素偏移小于 2^31。不支持转置、负 stride、空矩阵、任意非对齐维度、FP32/BF16/FP8、原地计算或融合激活。
- 检查实际导出 TTIR 的 signature：三列 stride=1 在源码路径被常量特化；文件入口仅传三指针、M/N/K、三行 stride。指针和 M/N/K/行 stride 的 tt.divisibility=16，动态形状不固定为 1024。
- 正确性：官方教程 31 个方阵性能 shape（M=N=K=256…4096、步长128），另有小尺寸、M/N tile 边界、K 尾块、group 尾部、padding 和压力输入；具体矩阵冻结在 common.py/validate.py。比较 native ATen FP16 与禁止 TF32 的 FP32 matmul 后转 FP16；小压力用例额外用 CPU FP64 参考。所有比较 atol=1e-2、rtol=1e-3。该容差适用于本轮输入范围，不能外推任意溢出输入。检查有限性、每次重置 NaN 输出 sentinel、行尾 canary、输入逐位未修改。
- 压力分布：零、常数 0.25、交替符号、较大但不溢出的输入（N(0,1)*4）、较小输入（N(0,1)*1e-3）；随机主矩阵为 N(0,1)。记录最大绝对误差，不把容差当观测误差。
- pipeline：源码 TTIR 与冻结文件字节一致；源码/文件 baseline target、完整解析 options、LLVM/PTX/cubin 字节一致；TTGIR 同 parser round trip 后一致；输出逐位一致。文件入口从 make_ttgir 继续，不执行 Python 前端或重新 make_ttir；禁止 override/inspection hooks/独立 llc/binary 替换。
- 计时：同一输入、预分配输出、原生 CompiledKernel、公共 launch、同一 CUDA stream。编译/加载/10 次预热/同步在计时外。triton.testing.do_bench_cudagraph，31 个方阵 shape（256…4096、步长128）、每点 5 轮、rep=20ms，保存全部样本、中位数、min/max、TFLOPS；baseline-only 不计算优化加速比。独立进程 seed=123 重复测量。未锁时钟、未主动清缓存，不能称冷缓存；GPU 状态前后记录，不声称完全无外部干扰。
- 证据：每次独立 runs 目录，保存命令、Git commit/dirty 状态、环境、输入 TTIR 和各阶段 hash、资源、全部误差和时间。正确性与 pipeline hash gate 先于性能。baseline 和证据不隐式覆盖。
- memcheck/racecheck 使用小型边界 suite，保存日志和退出状态。当前没有最终候选，不能报告“优化已验收”。
- 后续开始优化前，由用户另行指示；候选仅直接修改 TTIR，配置、签名、grid、测试和计时契约保持一致。

本次按用户要求将旧 15 点性能矩阵替换为官方完整 31 个方阵点；旧 manifest、common.py、契约及报告保存在 setup_attempts/v2。源码、baseline TTIR、配置、launcher 和计时方法不变，不重新 autotune。矩形与边界检查仍属于正确性范围，不计入性能矩阵。
