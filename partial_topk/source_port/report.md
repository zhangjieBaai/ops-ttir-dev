# 大N partial Top-K：直接改 Triton 源码的实测结果

忠实回写后的源码版复现了现有TTIR优化的机器指令，性能与TTIR版基本一致。所有映射的ELF `.text` 完全相同；
源码路径和TTIR文件路径的完整cubin可能因调试位置等信息不同，不能把`.text`一致写成完整cubin一致。
本次不需要额外调优或第二个源码候选。

| 本轮几何平均 | 主测 | 独立复测 |
|---|---:|---:|
| baseline / TTIR优化 | 1.083043× | 1.082989× |
| baseline / 源码优化 | 1.082648× | 1.082253× |
| TTIR优化 / 源码优化 | 0.999995× | 0.999474× |

最后一行>1表示源码更快。三种比值均来自本轮逐轮直接对照，不是用历史加速比相除。
微小差异属于此次测量波动，不能在指令相同时宣称某条编译入口更快。
逐点中位数的乘除不一定等于直接比值的中位数，因此三行也不必精确满足乘法恒等式。

| shape | baseline μs | TTIR μs | source μs | 主测IR/source | 复测IR/source | 复测baseline/source |
|---|---:|---:|---:|---:|---:|---:|
| (1, 65536, 8) | 7.800 | 6.830 | 6.830 | 0.999983× | 0.999766× | 1.141923× |
| (16, 65536, 8) | 11.472 | 10.556 | 10.566 | 1.000167× | 1.000085× | 1.085806× |
| (128, 65536, 8) | 45.955 | 44.870 | 44.875 | 0.999887× | 1.000119× | 1.024141× |
| (16, 262144, 8) | 27.116 | 25.966 | 25.973 | 0.999890× | 1.000132× | 1.043959× |
| (16, 262144, 16) | 52.769 | 49.072 | 49.069 | 1.000059× | 0.999975× | 1.074998× |
| (4, 65537, 8) | 8.440 | 7.453 | 7.473 | 0.999984× | 0.996774× | 1.127540× |

## 实际源码改了什么

`source_v1.py`在K循环前归一化±0，用公开bitcast、signed shift、OR/XOR生成uint32有序key，
循环变量保持uint32，`tl.max(keys)`后仍以原始index求min解决ties；选中key改为-inf对应哨兵。
前端保留整数循环状态，PTX保留 `redux.sync.max.u32`，local/merge两阶段的指令段都与ordered_carry_v3一致。
保持原两kernel、chunk1024、工作区和稳定index合同，含尾块、subnormal、最低有限值和±0压力测试。
值要求数值相等，±0等价，不要求保留零符号位。

## 验证与计时

使用官方Triton3.7.0 / H100，同原shape映射和完整backend options，无重新autotune。
源码由正常 `@triton.jit` 产生CompiledKernel，**计时直接启动该对象**；
其导出TTIR的重编译只做LLVM/PTX/cubin和输出一致性检查，没有替代源码路径。
源CompiledKernel的Python launcher需附带constexpr参数槽，但CUDA实际参数、grid和设备序列与文件路径相同。
三方使用同一输出存储和计时包装，CUDA graph指标不包含Python启动、编译、分配。

主测7轮、独立seed123复测5轮、rep20ms；逐轮随机三路顺序，同shape每轮A/A。
计时输入在设置seed之后分配，记录实际输入哈希。未锁频、未主动清cache。
IR/source固定矩阵配对bootstrap95：主测 [0.998810037312905, 1.0007441232480634]；
复测 [0.9990408287063152, 1.0010497235618687]。这只描述保存样本的波动。

三路均先运行原参考/压力校验后计时，新源码的memcheck和racecheck已完成：0 errors、0 warnings。
本轮保留原baseline源码/TTIR pipeline核验，源码回写前先格式化后冻结，未改动原实验文件。

[主测](results/main/results.json) · [复测](results/repeat/results.json) · [汇总/代码对照](summary.json) · 
[memcheck](results/memcheck.log) · [racecheck](results/racecheck.log) · [源码](source_v1.py)

## 复现

```bash
CUDA_VISIBLE_DEVICES=7 /root/venvs/ir-ops-dev/bin/python complex_suite/source_port_runner.py \
  --op partial_topk --run my_repeat --seed 234 --rounds 5 --rep 20
```

从仓库根执行，结果目录必须不存在；`--check-only`只做pipeline与数值检查。
这是同一个已知优化的源码回写对照，不是两组Agent各自独立搜索的A/B/C试验。
