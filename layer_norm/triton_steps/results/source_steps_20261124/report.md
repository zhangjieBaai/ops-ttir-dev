# Triton 逐项源码示例验证

九种源码表达和原始冻结TTIR基线共十条路径，每条完成142个原验证用例及120个范围边界用例，合计2620项检查通过。包含native ATen输出、FP64均值/方差倒数参考、padding canary和输入不修改。

全部九个源码版本正常JIT后导出TTIR，文件入口重编译的LLVM/PTX/cubin与各自源码入口逐字节一致；这一检查不代表这些源码与历史手写TTIR逐字节相同。原教程编译链一致性另行通过。

H100，官方Triton3.7.0，固定8warps/3stages/1CTA，原FP16合同。所有数值检查先于性能。30点M4096、N1024…15872步长512，5轮rep10ms、每版100ms预热、随机顺序配对和每点A/A；仅预分配CUDA graph kernel时间。

| 版本 | 相对冻结原始baseline的几何加速 | registers | spills |
|---|---:|---:|---:|
| baseline | 1.000069× | 60 | 0 |
| reciprocal_only | 1.000961× | 60 | 0 |
| rsqrt_only | 1.004869× | 60 | 0 |
| arithmetic | 1.005758× | 60 | 0 |
| grouped | 1.207406× | 32 | 0 |
| unroll | 1.294038× | 80 | 0 |
| mean_ca | 1.373897× | 48 | 0 |
| output_cs | 1.393043× | 48 | 0 |
| mixed | 1.409594× | 48 | 0 |

A/A中位数范围[0.999253, 1.000510]，全部通过。

本表为新源码示例的单会话完整矩阵观测（B级），没有独立进程复测，不替换此前TTIR正式实验加速比。寄存器/spill是C级静态资源，不能单独解释性能原因；未对全部新源码版本重跑sanitizer。

[完整结果和哈希](results.json) · [全部shape CSV](all_shapes.csv)
