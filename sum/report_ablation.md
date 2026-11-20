# Sum 消融：收益来自哪里，Triton 能否实现

**主要收益来自移除第二个短 kernel。** 主测最终单kernel为1.356914×，独立进程复测为1.356423×。
保留第二个拷贝kernel、只扩大第一阶段归约范围时，结果约1.000×。
因此不能把最终约35.7%的平均加速归因于索引缩窄、加法分组或缓存属性。

这些计算改动都能用公开Triton表达，并已实际编译、执行和验证。
唯一的API例外是`tl.load(cache_modifier=".cs")`：官方Triton3.7.0拒绝该参数。
历史TTIR虽然接受`.cs`属性，但本轮全部对应实例都生成与原实现相同的可执行指令，未产生有效优化。

本报告中的baseline（用于对照的原两阶段sum）来自前轮冻结文件。
TTIR（Triton中间表示）、cubin（GPU编译产物）及文件校验值均保留，用于确认计时和精度检查运行的是同一份代码。

## 每项改动的独立收益

下表所有版本都与同一baseline在同进程内交错配对；每列为逐shape配对比中位数的等权几何平均。
加速比大于1表示更快；例如1.356914×表示吞吐对应的加速约35.69%，不是延迟降低35.69%。
相应延迟降低为`1-1/1.356914`，约26.30%。

| 改动 | 30点主测 | 30点复测 | 小输入15点主测 | 结论 |
|---|---:|---:|---:|---|
| 索引 i64→i32 | 0.999548× | 0.999441× | 0.998662× | 无可靠整体收益 |
| reshape 分组归约 | 0.999679× | 1.000142× | 0.999787× | 无可靠整体收益 |
| i32＋分组归约 | 0.997200× | 0.998911× | 0.996782× | 未加速；主测小幅退步 |
| 加载 .cg | 0.942880× | 0.942795× | 0.999614× | 整体退步约5.7% |
| 加载 .cs | 1.000115× | 0.999705× | 1.000712× | 指令相同，差异为测量波动 |
| 完整CTA无掩码、末块保留掩码 | 0.994492× | 0.993965× | 0.989305× | 小幅退步，无整体收益 |
| 小输入扩大单CTA归约，仍保留拷贝kernel | 0.999080× | 1.000215× | 0.997911× | 约不变 |
| 小输入单CTA，移除第二kernel | 1.356914× | 1.356423× | 1.840157× | 主要收益来源 |
| single＋大输入tail | 1.357592× | 1.355520× | 1.841454× | 对single未增加可靠收益 |
| 公开Triton写法的single | 1.356217× | 1.356435× | 1.839246× | 与直接TTIR路径性能一致 |

最终single只在`n<=16384`的15个shape改变启动方式，其余15点保留原两阶段实现。
主测这15个小输入的几何平均为1.840157×；不能将其称为所有大尺寸输入均加速84%。
完整30点single的95%重采样区间为[1.355219,1.358328]，独立复测为[1.354274,1.372691]。
这些区间只描述本轮样本，后一个区间较宽，不能把千分级差异当成确定收益。

全部15种对照关系、30个shape的耗时和逐点区间在[per_shape.csv](ablation/per_shape.csv)。
完整汇总与每项区间在[analysis.json](ablation/results/analysis.json)。
这里没有复用或相减前轮的性能平均值；本轮单独使用一张卡重测。

## 拆开最终改动：扩大归约范围与移除第二kernel

只扩大张量宽度而不改变原grid会重复计算和写越界，不能当作合法sum参与消融。
因此构造了一个仍然正确的中间版本wide_two：

1. 原实现：多个CTA分别归约，再由第二kernel汇总。
2. wide_two：小输入由一个CTA完成全量归约并写mid[0]，仍启动第二kernel将这个值复制到out。
3. single：第一kernel直接写out，不启动第二kernel。

wide_two与single的第一kernel使用完全相同的TTIR文件和机器码，只改变输出指针绑定。
两者的直接配对测量隔离了“移除第二kernel及其单元素读写”的增量。
它不是对Python主机launch开销的测量，因为这里计时的是CUDA graph中的设备序列。

| 直接增量对照 | 30点主测 | 30点复测 | 小输入主测 | 含义 |
|---|---:|---:|---:|---|
| wide_two → single | 1.357464× | 1.358655× | 1.843822× | 移除第二kernel带来主要收益 |
| single → single_tail | 0.999787× | 0.999258× | 1.000283× | tail没有额外可靠贡献 |
| i32 → i32＋fold | 0.997777× | 0.998897× | 0.999624× | 组合没有独立正收益 |
| fold → i32＋fold | 0.998982× | 0.999537× | 0.999451× | 组合没有独立正收益 |
| 直接TTIR single → 公开Triton single | 1.000277× | 0.999629× | 1.000074× | 源码写法没有额外性能损失 |

各项收益有交互，不能将这些百分数相加。
尤其tail单独作用于小输入时有开销，single_tail在小输入上直接复用single，不会保留这部分分支。
这也是组合版与单项版的作用范围必须分别标明的原因。

## 每项是否可以用公开Triton实现

源码在[ablation_triton.py](ablation_triton.py)，可直接调用的分配和启动封装在[ablation_example.py](ablation_example.py)。
下表的“.text相同”指ELF可执行指令段字节相同，不是完整cubin相同；文件位置和调试信息会影响完整文件。
计数是逐shape逐阶段实例数，多个shape可能共用同一份编译文件。

| 改动 | 公开Triton写法 | 实测可执行指令一致性 | 判断 |
|---|---|---|---|
| 原两阶段对照 | `tl.program_id(0).to(tl.int64)`＋`tl.sum` | 60/60阶段实例相同 | 可实现 |
| i32索引 | 直接使用`tl.program_id(0)`，省去int64转换 | 60/60相同 | 可实现，本组偏移均在i32范围内 |
| 分组归约 | `tl.reshape(values,(4,B//4),can_reorder=True)`后两次`tl.sum` | 60/60相同 | 可实现 |
| i32＋分组 | 组合上述两项 | 60/60相同 | 可实现 |
| 加载.cg | `tl.load(...,cache_modifier=".cg")` | 60/60相同 | 可实现，但本轮退步 |
| 加载.cs | `tl.load(...,cache_modifier=".cs")`实际编译报错 | TTIR版与原baseline的60/60实例相同 | 该load提示不受公开API支持；本轮也未生效 |
| tail分支 | `if (pid+1)*B<=M`内无mask load，else保留mask | 49/60相同 | 语义可实现；部分指令生成不同 |
| wide_two | `B=next_power_of_2(n)`、grid=1，仍启动单元素拷贝 | 60/60相同 | 可实现，含主机启动方式调整 |
| single | 同上但直写out并省去第二次launch | 45/45相同 | 可实现，含主机启动方式调整 |
| single_tail | 小输入single，大输入tail | 35/45相同 | 语义可实现；tail部分代码生成不同 |

全部公开实现的源码编译产物与其导出TTIR文件重编译后的LLVM/PTX/完整cubin均字节一致。
这证明公开源码确实进入了同一原生编译后端，没有替换二进制或修改更低层IR。
公开实现与直接TTIR候选的1,917组输出和中间状态均精确相同，NaN按分类比较。
tail没有做到所有指令一致，所以未把直接TTIR tail的性能比冒充为公开tail的独立性能实测。

本轮把公开Triton single单独列入完整计时，并直接与TTIR single配对。
主测比为1.000277×、复测为0.999629×，区间包含1。
两者对应阶段的可执行指令也完全相同，因此这项约35.7%的平均收益不依赖直接编辑TTIR。

`.cs`的例外必须限定为**Triton3.7.0的load参数**。
同版本的`tl.store`支持`.cs`，不能笼统说Triton不支持`.cs`。
实际失败记录和本地官方semantic.py相关源码校验值见[public_api_evidence.json](ablation/results/public_api_evidence.json)、
[manifest.json](ablation/manifest.json)中的`public_failures`。

## 如何运行公开Triton例子

在`/root/ops-ttir-dev`执行，先选择一张空闲GPU：

```bash
CUDA_VISIBLE_DEVICES=3 /root/venvs/ir-ops-dev/bin/python - <<'PY'
import sys
sys.path.insert(0, "sum")
import torch
from ablation_example import sum_fp32
x = torch.randn(1024, device="cuda", dtype=torch.float32)
y = sum_fp32(x, variant="single")
print(y)
PY
```

同一封装还接受原两阶段、索引缩窄和分组归约等版本。
可选名称见[ablation_example.py](ablation_example.py)的`VARIANTS`。
为了避免把未验证范围当成支持承诺，这个实验封装只接受原30个shape的连续一维CUDA FP32输入。
它只做参数检查、分配和设备kernel启动，不用PyTorch compute计算sum。
这是可运行实验例子，没有注册到FlagGems的生产dispatcher。

## 测量、正确性与限制

日期2026-11-20（北京时间）。官方venv、Triton3.7.0、PyTorch2.12.0+cu130、H100 SM90。
正式数值和全部性能使用GPU3，UUID为`GPU-891abe27-275f-e39b-79ec-dbc9bc3ba6b2`。
导出最初在GPU4完成，随后GPU4被另一任务占用，设备锁拒绝了两次验证启动。
拒绝日志保留；实际验证迁移到GPU3并重新跑完30点源码/文件编译链核查。
没有混合两张卡的耗时。

本轮main每点7轮，repeat独立进程seed123每点5轮，共15种直接对照关系。
每轮随机对照顺序和对内顺序，同输入、预分配输出、原`do_bench_cudagraph`、rep=20ms。
编译和加载在计时前完成；每个版本先20次普通预热和100ms图预热。
原始replay值、随机顺序和A/A全部保留；这不是冷缓存或应用端到端测试。
UUID规范化设备锁及约1秒一次PID/时钟/功耗/温度采样未记录同卡其他进程或监控错误。
仍不能排除采样间隙内的短暂活动，没有锁定时钟。

两次测量全部30点的A/A均在[0.97,1.03]范围。
全部结果都保留，包括退步点；没有按最快样本或有利shape筛选。
95%区间按逐shape的配对轮次bootstrap2000次，不包含跨机器或跨日期变异。
这30点已经用于前轮优化，因此不是独立holdout；独立进程复测不等于新shape泛化证据。
完整矩阵＋复测支持本次平均结果，记为A级范围证据；可执行指令和API支持判断为C级静态/编译证据。

精度验证：19个版本×213个输入实例，共4,047次检查。
参考为CPU FP64和native ATen FP32，沿用`abs(error)<=2e-5+2e-6*sum(abs(x))`。
该L1缩放误差界不能称为紧输出相对精度。
输出预填标记值（sentinel）后每次重置，检查输出覆盖、尾部保护值、输入不变和各路径的中间约定。
wide_two只承诺mid[0]为全量和，其余mid必须不写；single路径所有mid必须不写。
普通两阶段继续验证各块部分和。

全部19个版本的memcheck和racecheck各覆盖179个实例，共3,401次算子检查。
memcheck为0 errors，racecheck为0 hazards、0 errors、0 warnings。
范围包括全部30点随机及小输入的压力/NaN/Inf输入，没有声称额外大shape压力分布也全部经过sanitizer。

原始数据：[主测](ablation/results/main.json)、[复测](ablation/results/repeat.json)、
[精度](ablation/results/accuracy.json)、[编译链](ablation/results/pipeline.json)、
[memcheck](ablation/results/memcheck.log)、[racecheck](ablation/results/racecheck.log)。
原baseline、旧轮次的optimized别名和选择记录均未修改。

若将每种shape各调用一次后累加全部耗时，最终single的比值见analysis.json的`cumulative_latency_ratio`。
它不同于这里报告的逐shape等权平均，大输入耗时占比高，因此累计时间收益更小。
本轮该比值为主测1.061911×、复测1.061597×，并非累计耗时降低35.7%。

重现完整消融：

```bash
CUDA_VISIBLE_DEVICES=3 bash sum/test_ablation.sh
```

脚本使用冻结导出物，新建结果目录，先精度和pipeline检查，再运行全部直接对照。
独立复算原始样本：

```bash
/root/venvs/ir-ops-dev/bin/python sum/analyze_ablation.py --output /root/tmp/sum-ablation-analysis-new.json
```

本轮使用[KernelForge benchmark-evidence](/root/KernelForge/skills/benchmark-evidence/SKILL.md)与
[review-agent](/root/KernelForge/skills/review-agent/SKILL.md)流程；没有将缺少sum执行适配器的KernelForge task声明为算子验收通过。
