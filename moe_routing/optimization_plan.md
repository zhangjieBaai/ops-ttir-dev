# Grouped / biased MoE routing：冻结实验协议

本轮为直接 TTIR 实验；不存在独立、盲化的 A/B/C 源码对照组。原始实现为本地 FlagGems NVIDIA/Hopper `grouped_topk.py` 的两个未修改 kernel，来源和文件哈希见 `upstream/provenance.json`。完整设备序列包含 sigmoid 分数变换、每组 biased top2 求和、top4 group、top8 expert、未加 bias 的 sigmoid 权重提取、归一化和双输出。两个 kernel 的输出和 workspace 分配均在计时外。host 仅检查、分配、launch。

独立新契约：连续 FP32 logits 和 FP32 bias，E=256/512，连续等大 G=8 分组，选4组、8个 expert，有限 logits/bias，routed scale=1，renormalize=true。本轮 sigmoid **沿用原实现的 `0.5*tanh(0.5*x)+0.5` 浮点形式**；不替换为 exp sigmoid。专家分数为 sigmoid+bias，group 为最大两项之和；group/expert ties 均按较小索引优先。权重从未加 bias 的 sigmoid 得到。饱和导致全零时沿用原实现的 +1e-20 分母保护。

矩阵：tokens=16/128/512/4096，各配 E=256/512，共8点。每点先在 stage2 warps=4/8 两个配置上测 baseline 全序列，stage1 固定4 warps，stages=3。保留调参记录，按每点冻结映射；同组 baseline/候选始终相同签名、grid和完整后端options。候选冻结后只编辑 TTIR。

预算：membership（[E,G] 比较/归约改成 [G,E/G] broadcast/reshape）；argmax（expert value max 和 tie-index min 两次归约改成稳定 value/index 联合归约）；combined。不更改 nonlinear、加 bias、top2 group 数学和 normalization。候选失败保留。

数值：与原源码/TTIR基线 indices 严格相等、weights 逐位相等；独立 CPU reference 检查稳定索引和 weights（rtol=2e-5、atol=2e-6）。测试 random、全相等、分组/专家tie、bias主导、饱和、zero logits，以及tokens=16/32/128/4096。不宣称支持 NaN/Inf、任意分组、不同K或dtype。重置 sentinel，检查输出覆盖、canary及输入不变。

性能：CUDA graph replay 包含两个 kernel，不含主机/分配/编译；相同输入与预分配输出，随机 baseline/candidate 对内顺序，每点主测7轮、独立进程复测5轮，rep=20ms；另做A/A控制。记录全部样本和IR/下游哈希、寄存器、spill、shared、环境/GPU状态。cache未清、时钟未锁；限定热缓存设备序列。按主测全8点几何均值选择单一策略，复测不重新择优。验收需要主测与复测≥1.02×、最差点≥0.95×、memcheck/racecheck无error，否则保留baseline。

导出前 ABI 核查决定 tokens 限为正16倍数：源码 JIT 对 runtime M 带 divisibility=16；不将1/17套入该冻结IR。
