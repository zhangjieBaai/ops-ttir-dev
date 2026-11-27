# MoE membership 源码回写：TTIR 与公开 Triton 的三路实测

忠实源码回写保留了原TTIR优化收益。
完整8点主测的 baseline/IR、baseline/source、IR/source 几何均值分别为
**1.150350×、1.150168×、0.999872×**；
独立复测分别为
**1.150904×、1.150672×、1.000107×**。
这组结果为A级性能证据，仅限父组冻结8点。
源码与membership TTIR的ELF `.text`逐字节相同，资源相同；
微小计时差异不足以支持两种表达存在性能优劣。

## 回写内容和三路身份

[membership_source_v1.py](membership_source_v1.py) 仅替换stage2的group membership：
原 `[E,8]` 比较/AND/sum变成公开 `tl.broadcast_to` 的 `[8,E/8]` 布尔张量，再 `tl.reshape` 到 `[E]`。
stage1使用原冻结源码的同一个JIT函数。
原8点、4 warps/3 stages、grid、两kernel、workspace、device ABI及数值合同均不变，没有重新调优。
这是用户明确授权的独立源码回写组；父目录纯TTIR基线、候选和历史结果未改。

三路被测对象分别为baseline TTIR文件编译、membership TTIR文件编译、
原stage1源码JIT与新stage2源码JIT返回的真实CompiledKernel。
新源码导出TTIR再文件编译只用于pipeline核验，没有替代计时中的源码对象。
源码launcher保留Python签名中的constexpr槽位；底层device ABI与文件入口一致。
初次harness漏掉中间constexpr槽位而抛出TypeError，见[失败日志](results/initial.log)；
修正仅涉及launcher，冻结源码未改，失败轮没有性能数据。

## 前端和机器码

每次运行逐点检查原baseline source/file及优化source/file的LLVM/PTX/cubin字节一致，
并比较三路完整backend options和target。全部通过。
优化源码前端的TTIR保留 `[8,E/8]` broadcast/reshape，消除了旧 `[E,8]` membership中间量。

跨“membership TTIR vs 优化源码”比较时，完整LLIR/PTX/cubin不同，
但两个E配置的ELF `.text`完全相同；
ELF section逐项比较只有 `.debug_line`、`.nv_debug_ptx_txt`、`.rela.debug_line` 不同，
见[section证据](results/elf_section_comparison.json)。
不能把`.text`一致写成完整cubin一致。
stage2资源：E256为25 registers、16B shared、0 spill；E512为28 registers、16B shared、0 spill，
两种优化路径均相同；baseline分别为32 registers、1024/2048B shared、0 spill。

## 完整两kernel设备时间

GPU2 H100、官方Triton3.7.0，预分配CUDA graph replay；
主测7轮、复测5轮，每项rep20ms，每轮随机三路执行顺序。
每个比值使用同一轮时间对，逐点取配对比中位数，再做几何汇总。
不计分配、编译、加载、Python调用；未清cache、未锁时钟。
每点输入显式seed和X/bias SHA256、全部执行顺序与样本保存在
[主测](results/main/results.json) 和 [复测](results/repeat/results.json)。
A/A主测各点中位比范围0.999422–1.001204，
复测0.999400–1.002814，均通过[0.97,1.03]门槛。

| tokens×experts | baseline μs | IR μs | source μs | baseline/IR | baseline/source | IR/source主测 | IR/source复测 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16×256 | 7.191 | 6.845 | 6.846 | 1.0504× | 1.0503× | 0.9999× | 1.0002× |
| 128×256 | 7.637 | 7.279 | 7.280 | 1.0493× | 1.0494× | 1.0000× | 1.0035× |
| 512×256 | 10.641 | 9.553 | 9.555 | 1.1145× | 1.1142× | 0.9995× | 1.0003× |
| 4096×256 | 48.679 | 41.306 | 41.307 | 1.1785× | 1.1785× | 1.0000× | 1.0000× |
| 16×512 | 8.110 | 7.229 | 7.230 | 1.1219× | 1.1221× | 1.0000× | 0.9981× |
| 128×512 | 8.655 | 7.734 | 7.766 | 1.1156× | 1.1147× | 1.0002× | 0.9994× |
| 512×512 | 12.961 | 10.556 | 10.556 | 1.2273× | 1.2265× | 0.9994× | 0.9997× |
| 4096×512 | 63.796 | 46.256 | 46.256 | 1.3791× | 1.3793× | 1.0001× | 0.9997× |

时间列各自取中位数；配对比中位数不必等于两列时间之比。
`IR/source`大于1表示源码耗时更低，不表示TTIR收益更大。

## 精度与内存/同步检查

成功初验、主测、复测、memcheck、racecheck各144项（三路径×8点×6分布），
总计720项，全部通过。
分布沿用normal、ties、group_ties、bias_dominant、saturated、negative；
三个输出相对baseline逐位一致，CPU稳定indices完全相等，
CPU权重满足rtol=2e-5/atol=2e-6，且输入哈希不变、sentinel及canary通过。
新源码路径实际包含在两种sanitizer执行中：
[memcheck](results/memcheck.log) 0 errors；
[racecheck](results/racecheck.log) 0 errors、0 warnings。

## 可以得出的结论

本例的TTIR优化可以忠实回写公开Triton，前端保留目标结构，最终设备指令与TTIR方案相同。
因此这里没有测出必须直接改TTIR才能获得的独占收益。
这个结果只回答既有变换能否回写以及两条编译路径的性能，
不衡量Agent在未见答案时发现该优化的概率，也不是A/B/C盲化优化能力试验。
复现命令见[README](README.md)，机器可读三路汇总见[summary.json](summary.json)。
