# LayerNorm 逐步消融的 NCU 机制核查

固定既有TTIR：baseline、abl_arithmetic、abl_grouped、abl_unroll、abl_ca、abl_cs、abl_final。只对既有验证通过版本采集硬件计数器，不生成候选、不修改源码/IR、不重新选优、不替换正式30点性能数字。

官方Triton3.7.0/H100、8warps/3stages/1CTA、FP16/FP32原数值合同，M4096、N=1024/2560/4096/15872。全部版本先跑原verify_pipeline与smoke数值检查，采集程序每个launch还要验证Y误差。GPU0独占；NCU base-clock/cache-control=all，采集LaunchStats/Occupancy/SpeedOfLight/WarpStateStats/MemoryWorkloadAnalysis。先核查GPU空闲，输出前缀必须新建。

以本次相邻消融对照说明逻辑张量宽度、寄存器驻留上限、实际occupancy、加载等待及内存流量的关联。NCU为C级机制线索；base-clock、cache flush和replay使其时长不能替代常规CUDA graph加速比。寄存器数是编译事实，occupancy/吞吐为代表点动态观测；没有单独证据的缓存命中、加载依赖及调度解释必须写为推测。整个多分支binary的静态指令总数不能当作单路径动态计数。源码diff仅表达对应算法改动，既有源码版本与历史手写TTIR性能不必一致。
