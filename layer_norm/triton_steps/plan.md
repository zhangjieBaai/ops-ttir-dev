# LayerNorm 六步优化的 Triton 源码示例

用户要求查看相对原版Triton的改动。本目录为已有TTIR变换的源码表达与教学对照，不是新的TTIR候选搜索，不修改冻结教程、上游快照、原baseline/optimized或既有实验结果。

保留官方Triton3.7.0、FP16输入输出/FP32统计、8warps/3stages/1CTA、原ABI、行grid、N/stride为16倍数、无别名和原数值阈值。N继续作为运行时参数，范围分派留在设备kernel，不能通过host逐shape特化替代。源码只从冻结函数作文本/AST转写；注释和格式规范化不计优化。

九个源码：baseline；仅倒数复用；仅rsqrt；两项算术组合；2048动态块；范围静态展开；仅均值加载.ca；仅Y输出.cs；N≤3072混合1024块。每份可独立导入运行。步骤2的源码正文与算术版相同，变化仅为BLOCK_SIZE8192→2048；所有版本仍是同一行CTA内三遍处理。步骤3显式区分固定K的static_range与大N动态fallback；步骤6只改变前三个范围的宽度和组数。

验证先运行原verify_pipeline，固定官方venv并隔离空闲GPU0。逐版本正常源码JIT导出TTIR，再从新TTIR文件用原生后端编译，验证LLVM/PTX/cubin逐字节相同、完整options一致，并以实际文件kernel数值检查。原142例加全部120范围边界，复用原FP64统计和native ATen输出阈值。所有数值先于性能；完整30点相对同路径重编译的原baseline配对计时，5轮rep10ms、每版100ms预热、随机顺序、A/A门槛[.97,1.03]，保存每轮输入/IR/下游哈希和资源。运行结果ID必须新建。

已有TTIR历史加速比不自动适用于这些新源码。新计时只说明本次源码表达的结果；可表达、数值通过、下游一致、与历史手写TTIR机器码一致为不同命题，不混用。未新增sanitizer结论，不更改公开总报告既有性能数字。
