# Norm＋动态INT8：直接改 Triton 源码的实测结果

忠实回写后的源码版复现了现有TTIR优化的机器指令，性能与TTIR版基本一致。所有映射的ELF `.text` 完全相同；
源码路径和TTIR文件路径的完整cubin可能因调试位置等信息不同，不能把`.text`一致写成完整cubin一致。
本次不需要额外调优或第二个源码候选。

| 本轮几何平均 | 主测 | 独立复测 |
|---|---:|---:|
| baseline / TTIR优化 | 1.058272× | 1.055043× |
| baseline / 源码优化 | 1.058312× | 1.055791× |
| TTIR优化 / 源码优化 | 0.999497× | 1.000569× |

最后一行>1表示源码更快。三种比值均来自本轮逐轮直接对照，不是用历史加速比相除。
微小差异属于此次测量波动，不能在指令相同时宣称某条编译入口更快。
逐点中位数的乘除不一定等于直接比值的中位数，因此三行也不必精确满足乘法恒等式。

| shape | baseline μs | TTIR μs | source μs | 主测IR/source | 复测IR/source | 复测baseline/source |
|---|---:|---:|---:|---:|---:|---:|
| (1, 4096) | 2.960 | 2.860 | 2.861 | 0.999659× | 0.998693× | 1.037661× |
| (128, 1024) | 2.215 | 2.080 | 2.079 | 1.000182× | 1.002094× | 1.062723× |
| (128, 4096) | 3.702 | 3.593 | 3.596 | 0.998724× | 0.997993× | 1.028182× |
| (2048, 4096) | 29.924 | 27.823 | 27.825 | 0.999949× | 1.000066× | 1.074737× |
| (512, 8192) | 15.834 | 13.340 | 13.354 | 1.000136× | 1.000421× | 1.166415× |
| (513, 4097) | 14.742 | 14.459 | 14.458 | 0.999979× | 1.001305× | 1.014800× |
| (32, 16384) | 8.077 | 8.134 | 8.144 | 0.998411× | 0.999521× | 0.989249× |
| (64, 257) | 2.222 | 2.071 | 2.061 | 0.998938× | 1.004470× | 1.082022× |

## 实际源码改了什么

`source_v1.py`把 `sum(z*z)` 和 `max(abs(z*w))` 写入公开 `tl.reduce` 的tuple组合函数，
再以 `max_u*rho` 得到amax，输出沿用原除法、clamp、nearbyint、INT8 cast，三个store都用公开 `.cs`。
没有采用另一个未获选的消去rho方案。前端后的TTIR保留多值reduce和cs，最终指令段与stream_v1一致。
量化仍采用原先允许相对FP64最多差1 code的合同，另检查scale、重构误差、Z、INT8范围、canary和输入不修改。
源码与IR在计时normal输入上的输出相等另行记录，不能推广成所有可能输入逐位等价。

## 验证与计时

使用官方Triton3.7.0 / H100，同原shape映射和完整backend options，无重新autotune。
源码由正常 `@triton.jit` 产生CompiledKernel，**计时直接启动该对象**；
其导出TTIR的重编译只做LLVM/PTX/cubin和输出一致性检查，没有替代源码路径。
源CompiledKernel的Python launcher需附带constexpr参数槽，但CUDA实际参数、grid和设备序列与文件路径相同。
三方使用同一输出存储和计时包装，CUDA graph指标不包含Python启动、编译、分配。

主测7轮、独立seed123复测5轮、rep20ms；逐轮随机三路顺序，同shape每轮A/A。
计时输入在设置seed之后分配，记录实际输入哈希。未锁频、未主动清cache。
IR/source固定矩阵配对bootstrap95：主测 [0.9983172208109422, 1.0006939048522652]；
复测 [0.997681381879477, 1.0021236834127039]。这只描述保存样本的波动。

三路均先运行原参考/压力校验后计时，新源码的memcheck和racecheck已完成：0 errors、0 warnings。
本轮保留原baseline源码/TTIR pipeline核验，源码回写前先格式化后冻结，未改动原实验文件。

[主测](results/main/results.json) · [复测](results/repeat/results.json) · [汇总/代码对照](summary.json) · 
[memcheck](results/memcheck.log) · [racecheck](results/racecheck.log) · [源码](source_v1.py)

## 复现

```bash
CUDA_VISIBLE_DEVICES=6 /root/venvs/ir-ops-dev/bin/python complex_suite/source_port_runner.py \
  --op norm_quant --run my_repeat --seed 234 --rounds 5 --rep 20
```

从仓库根执行，结果目录必须不存在；`--check-only`只做pipeline与数值检查。
这是同一个已知优化的源码回写对照，不是两组Agent各自独立搜索的A/B/C试验。
