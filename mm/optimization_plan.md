# MM 六组 autotune baseline 契约 v4

当前仅建立并验证基线，不开始 TTIR 优化，不生成 optimized.ttir。

- 来源：官方 Triton v3.7.0 tutorial 03，保持 tutorial.py、上游快照和旧 ir/baseline.ttir 不变；来源 hash 和本次未远程独立核验的限制见 upstream/provenance.json。
- 设备/环境：指定 /root/venvs/ir-ops-dev/bin/python、官方 Triton 3.7.0、H100 SM90、FP16；先选择空闲物理设备，显式 CUDA_VISIBLE_DEVICES。检查真实模块路径、RECORD、动态库映射、完整编译 options、toolchain hash、UUID。清空外部 Python/LD_LIBRARY_PATH。
- 选择来源：冻结 results/runs/autotune_count_20261118T135029Z/summary.json 中31个官方shape的实测获选映射，不重新autotune。31个方阵点为 M=N=K=256…4096、步长128，共6种配置。每组 ir/<family>/baseline.ttir 从该次实际TTIR字节复制并校验hash。
- baseline.json 保存 shape_to_family、families、每组BM/BN/BK/GROUP/warps/stages/CTAs、选中shape列表、来源TTIR/cubin hash和完整协议hash；每组从TTIR编译，使用该组自己匹配的完整编译选项与真实resource metadata。自动dispatch只支持31个冻结shape，其他shape不猜测配置。
- 算子契约：C=A@B，无activation，FP16 A/B/C、FP32累加，行主序、列stride=1、指针16字节对齐、无别名。M/N/K和行stride均为正16倍数，最大元素偏移<2^31；允许行尾padding16。每组TTIR有9个运行时参数（三指针、M/N/K、三行stride），全部tt.divisibility=16；实际signature必须核验。
- 源码/TTIR一致性：对31个shape逐点使用所属配置验证：源码TTIR与所属baseline字节一致；target和全部解析options相同；LLVM/PTX/cubin字节一致；TTGIR相同parser round trip后一致；源码与文件启动输出逐位一致。文件入口从make_ttgir继续，不重新make_ttir；禁止override/inspection修改钩子/独立llc/二进制替换。
- 正确性：31个性能点只验证冻结dispatch；另对每组显式运行小尺寸、tile/K/group边界、padding、normal/zero/constant/alternating/large/small压力分布。组级扩展用例用于检验IR合法范围，不代表这些额外shape有自动性能dispatch。每次NaN sentinel、padding canary、输入不修改、有限性检查；参考native ATen FP16、禁TF32的FP32后转FP16，小尺寸另用CPU FP64。容差atol=1e-2、rtol=1e-3；逐项记录实际误差，不外推任意溢出输入。
- 性能：固定31方阵点、每点5轮、rep=20ms、triton.testing.do_bench_cudagraph；所属组kernel/输入/输出均在计时前编译加载预热10次并同步。公共launcher、同一stream、预分配输出，保存每点family/TTIR hash和全部样本/中位数/TFLOPS。seed=123独立进程复测。没有候选，不报告优化加速比。未锁/核实锁频、无显式cache flush，不能宣称冷缓存。
- pipeline/正确性与manifest/每组TTIR hash gate先于性能；31点对应输出及metadata逐点记录，6组资源各自记录。memcheck/racecheck对每组5个padding边界用例执行，失败阻止通过。
- 新证据写独立results/runs目录，保存命令、Git状态、环境/工具hash、所有IR产物hash/误差/样本。旧单配置manifest、脚本和报告保存在setup_attempts/v3，旧结果原样保留。旧ir/baseline.ttir仅为历史单配置baseline，不是活跃自动dispatch入口。
- 后续仅在用户要求后开始直接TTIR优化；每组候选必须共享该组冻结配置、签名、grid和同一测试/计时流程。
