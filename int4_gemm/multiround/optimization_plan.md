# INT4 第三至第五轮：顺序增量实验

固定第二轮 expanded/manifest.json 的30点、配置、grid/ABI和数值合同。anchor为第二轮r2_group_pair，r3的parent=anchor；每轮仅在独立主测/复测通过后晋升，否则下一轮继续使用同一parent。原源码/IR/历史结果不改。候选仅直接编辑TTIR，不使用优化Python kernel、下游IR修改、inline PTX或重新autotune。

- r3：固定GROUP128的除法改为shift7；显式利用连续stride关系；两项组合。GROUP128、连续布局本来就是合同，不缩窄shape范围。函数签名保留，host仍传原参数。
- r4：按2/4/8行tile分组的PID映射，保持grid总量和一一映射。单个M tile的shape保持原映射，因为新映射只会增加无效地址工作；此条件预先声明，不按测量逐点择优。最后不足整组使用真实group size。
- r5：第二个64-K片段的A/packed加载提前到第一个dot之前；group scale加载提前；输出cs cache hint。每项独立，从本轮parent生成；BK128时无第二片段，因此preload_pair保持parent。

每轮原baseline、固定round2 anchor、当前parent和新候选同场比较，保存direct parent/candidate、anchor/candidate、baseline/candidate。全部30点原5种压力分布与精度/CPU FP64/canary/输入不变先于计时。baseline源码与文件LLIR/PTX/cubin逐点一致，完整backend options一致。

初筛3轮、rep10ms；固定全30点最优单一候选后主测7轮rep20ms、独立seed123复测5轮rep20ms。rep是do_bench_cudagraph参数，内部10次重放默认均值；保留全部外层样本及随机顺序。parent A/A中位在[0.97,1.03]，超限轮作废。两轮增量geo>=1.02且最差>=0.95，并通过memcheck/racecheck才接受；负结果完整保留。可执行代码与parent完全相同的候选只记C级无变化证据，不以微小计时波动声称优化。

GPU0独占；官方Triton3.7.0 venv；不锁频、不主动清cache。公开Triton可以表达这些变换，但本次未进行源码回写实测。

r5生成前修订：以两个量化group的外循环展开替代原拟定output.cs候选，总预算仍3个。仅K为256倍数的既有case执行展开；奇数group计数（含K128、K1152）直接保留parent，避免新增尾处理改变数值或测量接口。此K-family适用规则在r5任何计时前冻结，不根据逐点性能选择。两个group各自加载scale/zero，绝不跨group共享量化参数；dot与accumulator顺序不变。原output.cs想法未实施，不能记为实测负结果。
