# LayerNorm TTIR 第二轮

固定第一轮的 baseline、官方 Triton 3.7.0、SM90、8 warps、1 CTA、3 stages、相同 launcher/签名/grid、FP16 与容差。只改 TTIR；不使用内联 PTX 或手改下游 IR。第一轮 optimized 保留为 parent，不覆盖直到最终复核、独立复测和 sanitizer 完成。

目标用共同原生 baseline 比较，逼近 LLIR 成熟 cubin 的实际耗时；旧 LLIR 的 1.4x/1.476x 不同口径不可直接作为本轮目标。已有同协议共同原生 baseline 下 LLIR 几何平均约 1.412x，本轮 parent 约 1.293x。

## 假设与预算

第一批最多 8 个候选：

1. 延迟 Mean/Rstd store 到输出之后（无别名契约下合法），使跨三个阶段 X load 可被后端公共子表达式消除；风险为值存活时间变长。
2. 在 parent 上将输出归一化写成行级 scale/offset 及 FMA，减少输出中间量。
3. 将统计量累加和输出处理改为 1024/2048/4096 宽分组，三遍动态循环，减少寄存器；风险为循环指令开销。
4. 细分 N 范围、静态展开小分组，消除 mask 可判定的无效组，同时仅维持分组宽度的累加器。
5. 相同细分范围中用 volatile mean load 强制重载，作为输入复用/长存活期的对照，不预设减少加载一定更快。
6. 对 stats 的本地分组规约表达进行后续试验（预算未满时开展）。

后续最多追加 8 个基于第一批证据的候选，累计最多 16 个；追加前记录具体假设。禁止无限网格搜索。

## 验证

开始先 pipeline 核查。C0 编译保存完整输入哈希/资源和失败；每候选完整 142 用例验证后才可计时。初筛用原 6 点、3 轮（rep=10ms）；前两名及 parent 用完整 30 点、7 轮（rep=20ms），追加分支边界用例。候选顺序/对内顺序随机，所有样本保留。最终独立进程 seed=123 复测 5 轮，memcheck/racecheck 检查通过后再更新 optimized/selection。

与 LLIR 的补充测量只作性能参考，TTIR 主实验保持同原生链；四版对比脚本不允许用于替换 TTIR 候选 binary。最终报告给共同 baseline 下两个优化版的绝对耗时与加速比，并标注后端差异。保留所有失败和退步记录。

## 第二批追加（累计 13/16）

第一批正确性：6 个可编译新候选与 baseline/parent 共 1136 次通过。两份 affine 候选在 C0 因 arith.negf 未被当前文件入口的 lowering 接受而失败，保留原文件，用 subf -0 修复为新名。

初筛证据：r2_fine1024_reload 为 40 registers、无 spill，小 N 显著改善但大 N 回退；r2_delay 大 N 略有改善但 80 registers/4 spill。追加：affine_v2、delay_affine_v2（修复）、fine2048_reload（128-bit 向量化/更少分组）、fine2048_delay（输入全程复用对照）、fold2048（8192 宽加载经 TTIR reshape/reduce 折叠为 2048 宽累加器，输出2048）。保留最后 3 个预算用于证据驱动组合。用户授权 NCU，加入相同时钟和缓存条件下的 occupancy/内存/停顿分析。

## NCU 指导的追加（14/16）

NCU base clock/cache-control=all：重载版实际 occupancy 约65%~69%，parent约33%~35%，但 N=15872 重载版仍慢；其 long-scoreboard/issue 比值从5.67升到18.90。仅提高occupancy不足以解决大尺寸。第二批2048重载有48 registers，无spill，N=8704约52.3us，接近旧LLIR的约51.8us，但N=15872仍约100us。

检查PTX发现volatile mean load。因此追加 `r2_fine2048_ca`：改为显式 `cacheModifier=ca` 的普通load，与默认load属性区分，尝试避免TTIR CSE长存活且维持缓存/调度能力。不是手工改PTX，需核对实际SASS与计时。保留最后2个候选用于小/大范围组合和缓存提示。

## 最后两个候选（16/16，停止扩展预算）

r2_fine2048_ca 48 registers、无spill，初筛6点几何平均1.490x（parent 1.361x），其中4096约26.37us、8704约51.50us，但15872约96.76us仍回退。追加 r2_fine2048_stream，仅 TTIR 输出 store 设置 cacheModifier=cs；r2_mixed_stream 在此基础上把 N≤3072 分支改为1024分组，其他保持2048，检验小尺寸细粒度收益且保持整个kernel低寄存器。之后只做完整验证/复测，不再新增候选。


## 验收结果（A 级完整矩阵与独立复测）

- 最终16个候选（2个C0失败），选中 r2_mixed_stream。完整30点/7轮几何平均1.4397x、算术平均1.4995x；同轮parent为1.2952x/1.3171x。独立进程seed123/5轮为1.4395x/1.4991x。
- 相同原生baseline的补充LLIR对照（B级单进程3轮）：LLIR几何平均1.4124x左右，新TTIR1.4399x；新TTIR/LLIR耗时几何比0.9809。跨层候选后端不同，仅为性能参考。
- 48 registers，无spill；NCU实际occupancy约55%~57%（parent约33%~35%）；同步仍为默认lowering，未复现LLIR的两次barrier，不把所有收益归因于occupancy。
- 新分支边界120例/版本（normal/offset、±16、padding0/16）通过；同套120例经过memcheck/racecheck均0 errors，racecheck无warnings。
- 首次full_benchmark.json测量N=1024发生31~68us的baseline漂移，中止并整轮排除，保留partial；采用所有版本相同的untimed graph稳定阶段后重跑为full_stable.json，独立复测同样处理。未锁时钟，不推断漂移原因。
- r2_mixed_stream替换optimized之前，旧selection保存到results/round2/previous_selection.json。新selection绑定full_stable及候选SHA256。原始baseline/tutorial不变；原生pipeline复核通过。
- 结论边界：22/30点相对parent更快，最大回退约3.63%；大N不保证优于parent/LLIR。本轮达到全矩阵平均逼近LLIR的目标，停止本轮搜索。
