# 前向 LayerNorm：逐项消融与源码表达

本次固定官方Triton3.7.0、H100、FP16、8warps/3stages/1CTA、原始baseline、M4096/N1024..15872步长512，比较现有前向最终版r2_mixed_stream的组成。所有候选先数值验证，再完整30点主测5轮与独立seed123复测3轮，每轮10ms，所有版本同样100ms稳定阶段及A/A控制。

最终数字与每一项条件收益必须引用[消融报告](../layer_norm/report_ablation.md)，不要使用历史不同会话的6点初筛拼接贡献。

## 解释规则

1. “baseline单独加rsqrt”的效果与“最终组合移除rsqrt”的效果不是同一个量。两种测试必须写明其余条件。
2. 逐项加入的收益有交互，不能把百分比相加成总收益。静态展开、窄分组和小N分派存在结构依赖。
3. 动态分组从60降到32寄存器不代表最优；静态展开可增至80寄存器但更快；.ca和重载的组合又降到48。寄存器、静态load/barrier条数都只是资源线索。
4. 关闭静态范围分派并回到统一2048动态循环会同时关闭小N1024混合，必须称联合移除。较干净的展开对照为同为2048的动态分组→静态范围展开。
5. 低于约1%、方向不稳定的差异不应宣称可靠收益。本次没有计算置信区间。

## Triton可以表达最终全部算法变化

可执行公开API实现见[ablation_triton.py](../layer_norm/ablation_triton.py)：普通乘除表达倒数复用，`tl.rsqrt`，`tl.arange`/循环和累加器表达分组，运行时`if/elif`与`tl.static_range`表达范围展开，`tl.load(cache_modifier=".ca")`、`tl.store(cache_modifier=".cs")`表达缓存提示，constexpr helper参数表达1024/2048混合。

本例不是仅依据API文档判断：源码实现通过完整正确性/边界验证并进入同协议全矩阵性能测试。源码导出TTIR经文件入口重新编译后，LLVM/PTX/cubin与其源码编译路径一致；但与原手写TTIR最终版的机器码.text并不相同。算法可表达、逐阶段字节相同、资源相同、性能接近是不同强度的结论。

因此不能把前向TTIR约1.44×的收益称作普通Triton做不到的优化。TTIR提供精确编译截点与实验输入的控制，公开源码也可以表达本例的优化组合。来源：[源码路径验证](../layer_norm/results/ablation/source_formatted_parity.json)、[公开API源码快照](../layer_norm/results/ablation/public_api_evidence.json)、[资源及.text哈希](../layer_norm/results/ablation/resources.json)。
