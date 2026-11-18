# MM 六组 autotune baseline 已跑通

当前活跃实验 `mm-fp16-sm90-autotuned-six-v4`：31个官方方阵点各自使用已冻结的获选配置，共6份不同baseline.ttir。固定映射来自 [autotune统计](autotune_ttir_report.md) 的实测记录，未重新调参。旧单配置manifest/源码入口/契约/报告保存在setup_attempts/v3，旧root ir/baseline.ttir仅为历史原件。

六组路径、完整SHA256、配置和31点映射见 [baseline.json](baseline.json)。所有组均为官方教程原始计算体，不存在优化候选或手工IR改写；源码、编译器、加载、启动和计时仍走Triton原生流程。代码新增按shape选择配置组的dispatch，编译/验证/benchmark都读取同一映射。

| 配置组 | shape数 | registers | spills | shared bytes |
|---|---:|---:|---:|---:|
| [bm32_bn64_bk32_g8_w2_s5](ir/bm32_bn64_bk32_g8_w2_s5/baseline.ttir) | 1 | 95 | 0 | 24576 |
| [bm128_bn32_bk64_g8_w4_s4](ir/bm128_bn32_bk64_g8_w4_s4/baseline.ttir) | 3 | 72 | 0 | 81920 |
| [bm64_bn128_bk64_g8_w4_s4](ir/bm64_bn128_bk64_g8_w4_s4/baseline.ttir) | 6 | 124 | 0 | 98304 |
| [bm128_bn256_bk64_g8_w8_s3](ir/bm128_bn256_bk64_g8_w8_s3/baseline.ttir) | 18 | 210 | 0 | 147456 |
| [bm64_bn256_bk32_g8_w4_s4](ir/bm64_bn256_bk32_g8_w4_s4/baseline.ttir) | 1 | 212 | 0 | 81920 |
| [bm128_bn128_bk32_g8_w4_s4](ir/bm128_bn128_bk32_g8_w4_s4/baseline.ttir) | 2 | 174 | 0 | 65536 |

正式证据：[本轮GPU6运行](results/runs/baseline_20261118T135515Z_fg6zev/run.log)。环境是指定venv、Python3.12.3、官方Triton3.7.0、PyTorch2.12.0+cu130、H100 SM90，物理GPU6，UUID `c93dcf8b-2363-d1c2-02b4-4572cef79c41`。源码来自本机已有官方v3.7.0快照，未作本次独立远程来源核验，限制见upstream/provenance.json。

**31点pipeline逐项通过**：每点源码产生的TTIR与对应组baseline字节一致；解析后的完整编译options和target一致；LLVM IR、PTX、cubin字节一致；TTGIR相同parser round trip后相同；源码/文件路径输出逐位一致。每组均核查9参数signature和全部tt.divisibility=16。环境RECORD、实际模块/动态库路径、toolchain hash及逐点产物hash/完整options保存在 [pipeline.json](results/runs/baseline_20261118T135515Z_fg6zev/pipeline.json)。文件入口从make_ttgir继续，不重新运行Python前端或make_ttir。

**正确性**：seed=42、seed=123独立进程各157用例，共314次全部通过。每轮包括31个官方性能点按冻结映射dispatch，以及每组21个边界/padding/压力用例。额外组级用例验证IR可执行范围，不为额外shape新增自动dispatch。参考native ATen FP16和禁TF32的FP32结果，小尺寸另用CPU FP64；固定atol=1e-2、rtol=1e-3。输入不修改、输出sentinel、有限性、padding canary均通过，逐项实际误差见accuracy*.json，不将相对容差判定误写为绝对误差上限。

**内存与同步检查**：[memcheck](results/runs/baseline_20261118T135515Z_fg6zev/memcheck.log)与[racecheck](results/runs/baseline_20261118T135515Z_fg6zev/racecheck.log)各对6组×5边界用例执行，均退出0；memcheck 0 errors，racecheck 0 hazards / 0 errors / 0 warnings。

**性能**：31点各5轮、rep=20ms CUDA graph kernel时间，seed=123独立进程重复，共310个原始样本。每点记录所属family和实际TTIR hash；每组自己编译加载、使用真实metadata，不复制其他组资源。计时不包含分配、Python launch、编译和模块加载。两轮逐点中位数最大相对偏差5.51%。未锁定/核实锁频、无主动cache flush，进程前后检查不是连续独占证明。该表是新的分组baseline，不与历史单配置计时混为TTIR优化收益。

| M=N=K | BM×BN×BK | 首轮中位数 µs | 独立复测 µs | 首轮 TFLOPS |
|---:|---|---:|---:|---:|
| 256 | 32×64×32 | 2.994 | 2.998 | 11.21 |
| 384 | 128×32×64 | 3.444 | 3.446 | 32.88 |
| 512 | 128×32×64 | 3.862 | 3.865 | 69.51 |
| 640 | 128×32×64 | 4.484 | 4.502 | 116.91 |
| 768 | 64×128×64 | 4.959 | 4.923 | 182.68 |
| 896 | 64×128×64 | 5.571 | 5.526 | 258.23 |
| 1024 | 64×128×64 | 5.980 | 6.044 | 359.13 |
| 1152 | 64×128×64 | 9.789 | 9.541 | 312.34 |
| 1280 | 64×128×64 | 10.649 | 10.063 | 393.87 |
| 1408 | 64×128×64 | 11.947 | 11.660 | 467.26 |
| 1536 | 128×256×64 | 18.001 | 18.039 | 402.62 |
| 1664 | 128×256×64 | 19.747 | 19.849 | 466.64 |
| 1792 | 128×256×64 | 22.114 | 22.026 | 520.44 |
| 1920 | 128×256×64 | 24.940 | 25.319 | 567.60 |
| 2048 | 128×256×64 | 27.357 | 27.504 | 628.00 |
| 2176 | 64×256×32 | 46.087 | 45.966 | 447.13 |
| 2304 | 128×128×32 | 51.428 | 51.180 | 475.64 |
| 2432 | 128×128×32 | 56.802 | 56.771 | 506.47 |
| 2560 | 128×256×64 | 63.292 | 64.065 | 530.15 |
| 2688 | 128×256×64 | 69.984 | 70.531 | 555.04 |
| 2816 | 128×256×64 | 75.357 | 74.327 | 592.65 |
| 2944 | 128×256×64 | 105.517 | 106.039 | 483.64 |
| 3072 | 128×256×64 | 112.442 | 112.115 | 515.66 |
| 3200 | 128×256×64 | 123.219 | 122.726 | 531.86 |
| 3328 | 128×256×64 | 129.505 | 129.692 | 569.24 |
| 3456 | 128×256×64 | 139.581 | 139.795 | 591.46 |
| 3584 | 128×256×64 | 147.489 | 146.558 | 624.27 |
| 3712 | 128×256×64 | 189.256 | 189.379 | 540.51 |
| 3840 | 128×256×64 | 198.161 | 195.986 | 571.49 |
| 3968 | 128×256×64 | 210.483 | 209.848 | 593.65 |
| 4096 | 128×256×64 | 220.542 | 216.761 | 623.19 |

复现：先确认设备空闲，再从仓库根目录执行 `CUDA_VISIBLE_DEVICES=6 bash mm/test_performance.sh`。六份baseline对应同一份官方Python计算源码，但具有各自固定tile/warps/stages；后续每组候选必须与所属baseline保持相同配置和启动约定。
