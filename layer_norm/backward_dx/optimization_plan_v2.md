# backward dX 第二轮

继承第一轮dX-only提取baseline、五组block/warps、Triton3.7.0原生链、FP16/FP32统计量、无别名、同ABI/grid/launcher和全部数值容差。前向及第一轮不可变候选不修改；保存当前selection为parent。只改TTIR，无内联PTX/下游手工IR/编译选项调参。

## 瓶颈证据与假设

第一轮NCU（C级）：8192组parent约2.91TB/s，actual occupancy约34%，常规约67.7us；16384组current joint_coefficients仅约12% occupancy（220regs）且N8704仅约2.12TB/s。大N下barrier/issue明显小于long scoreboard/issue，优先研究活跃状态与加载依赖，不直接删除同步。上一轮静态分组重载到118regs，但仍有回退，不能重复无差别展开。

本轮最多8个候选：

1. r2_reload_x：规约后只重新加载X并重算xhat，打断xhat的长存活；额外访存2N bytes。
2. r2_reload_wdy：规约后重载DY/W，只打断wdy存活；额外访存4N bytes。
3. r2_reload_all：结合两者，普通ca属性区分load，非volatile；检查真实产物是否保留重载。
4. r2_loop2048：两遍2048宽动态循环，二路窄累加器联合规约、输出重载，避免静态完全展开造成宽工作集。
5. r2_loop4096：扩大循环体以减少控制指令代价。
6. r2_fold_reload：对r2_reload_all的两个规约输入reshape/fold到2048后联合规约，测试局部规约表达。
7–8仅根据以上的C0/C1/初筛证据追加，提前记录假设，不无限搜索。

## 测量与接受

开始先验证源码/基线原生pipeline和parent选择。全部135个正确性用例先于性能。筛选沿用6点3轮rep10ms和A/A，比较候选+不可变parent（各组复制r1_parent.ttir）。前两名全30点7轮rep20ms；每组只根据主测冻结选择，独立seed123/5轮rep10ms只验收，不用复测重选。

每组相对原始baseline及r1_parent均报告。新组级候选需主测/独立复测相对原始baseline>=1.02、对parent几何>=1.02、组内最差对parent>=0.95，A/A中位数0.97..1.03；不通过则保留parent（而非撤回旧成果）。最终alias再验pipeline、C1及30点性能。新增循环尾部和N边界套用第一轮sanitizer30例/版；最终memcheck/racecheck无错误后晋升。NCU按同base clock/cache-control=all与parent采集，仅作机制证据。时钟未锁定的常规graph性能与NCU分开报告，保留失败及全部负结果。

## 初筛后最后2个候选（8/8）

前6候选全部C1通过但无整体提升。r2_loop2048在N8704为1.198x、N15872仅0.627x；r2_reload_x在6点均回退，不继续缓存排列。追加r2_range2048/r2_range4096：只对b16384组N<=10240用动态窄循环，N>10240用第一轮静态grouped2048（118寄存器，最大N约比parent慢3%）；其他四组直接为不可变r1_parent副本。共同kernel的资源可能被fallback抬高，因此必须检查实际寄存器/occupancy，不能假设组合取各单路径最好性能。编译后如果失去低资源档位或计时仍回退则不晋升，停止本轮8候选搜索。

## 范围候选完整复核（主测后冻结，不根据复测重选）

30点7轮：parent几何1.0212x，range2048为0.9956x，range4096为1.0049x。所有组都未满足对parent>=1.02的门槛，provisional_selection全部保留parent。range两版实际123/126regs，单独loop2048为40；N8704的低寄存器路径组合后反而较慢，未继续追加第9个候选。

独立进程仍做完整5轮以确认负结果；补充DRAM read/write总字节NCU采集，区分实际HBM重读与仅load条数。新增分支10240的±16、padding0/16、normal/offset/lowvar共18例，连同原30边界例形成round2-boundaries套件（48例/版），用于最终候选工具验证。数值容差和原135用例不变。


## 收尾结论

- 独立5轮：parent几何1.0212x，range2048/4096分别0.9958x/1.0046x，所有A/A通过。没有新组晋升，decision.json记录promoted_blocks=[]，根selection.json和各optimized.ttir内容不变。
- 额外48例/版（含10240±16）C1、memcheck、racecheck均通过，每个工具192次kernel检查；0 errors，racecheck0 warnings。
- b16384 parent静态barrier2（baseline5）；reload_all为82regs、loop2048为40、range为123/126。单独降低寄存器没能改善全矩阵。
- C级专项NCU实际流量：N15872 parent读260.14MB/写118.31MB/131.264us，loop2048读501.83MB/写117.85MB/208.960us，DRAM读取约1.93倍；不把源load数量当HBM流量，也不将单次NCU作为正式加速比。
- 解析traffic CSV时先假定单位byte的断言失败，未产生派生数据；改为读取实际Mbyte并按SI转换。原始CSV不修改，分析脚本保留raw_units并可复现，不把解析错误算kernel失败。
- 同一原生pipeline后验通过。停止本轮8候选预算，下一轮若研究多CTA/行映射/编译选项需另设契约；本轮未放宽容差、变更ABI或启动配置。
