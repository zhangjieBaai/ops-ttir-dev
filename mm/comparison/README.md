# FlagGems Hopper / FlagTree 与教程 MM 对比

此目录是用户授权的跨实现、跨环境对比，独立于纯 TTIR 优化主实验，不修改其baseline、候选或协议。环境例外仅限本比较：FlagGems和教程主对比均使用已有`/usr/bin/python`的FlagTree0.7.0（实际triton3.6.0、torch2.11.0+cu130）；补充参照使用原venv的官方Triton3.7.0/torch2.12.0+cu130。

完整31个shape：M=N=K=256…4096、步长128，FP16、连续行主序、输出FP16。选择同一空闲H100 GPU0。输入由NumPy固定种子生成并转FP16，逐点保存CPU数据SHA256；两环境不能混用不同数据或旧计时结果。

主指标：预分配输出的CUDA graph设备时间，实际执行FlagGems目标文件的mm_out，包含其所有必要设备步骤；教程执行官方原始kernel、逐shape官方16配置autotune后冻结。FlagGems使用原实现的默认LibTuner选择（不缩候选集、不改源码），独立cache防止污染其他任务；保存实际入口、kernel名、完整配置、阶段hash、资源和选择耗时。首次选择与编译不计入kernel时间；正常计时不重新调参。A/A控制为同教程kernel的两次独立计时。

正确性先于计时，比较native FP16及禁TF32的FP32后转FP16，atol=1e-2、rtol=1e-3；输出每次NaN sentinel、检查有限性和输入未修改。另抽取最小/中等/最大尺寸做CPU FP64。若某点失败，保存错误并阻止该点性能结论。保留选中native cubin，禁止独立llc路径和二进制热替换。

配对随机顺序、每shape5轮、rep=20ms，保存全部样本；对相同冻结配置用seed123独立进程复测。每shape取配对加速比中位数，再在31点等权汇总算术/几何平均；按配对轮次2000次bootstrap给条件95%区间。A/A中位数偏离1超过3%、baseline配对轮次max/min超过1.15标记无效；保留所有原始样本和无效标记，不选最快值。阈值为本比较caller选择，原TTIR优化轮次标准不被改写。

补充比较官方upstream环境的原六组冻结教程baseline：在同卡上用相同输入和计时方法重新测量，按FlagTree/upstream/FlagTree/upstream进程块交错，分别保存；跨环境比值含torch/编译器/runtime差异，不能宣称只是kernel算法或TTIR改写收益，不替代同FlagTree环境的配对主结果。

不锁定/核实锁频，无主动cache flush，报告重复输入graph缓存状态与设备进程检查限制。原FlagGems源码保持不变；本比较不修改TMA或TLE实现。结果见 [report.md](report.md)，包含逐31点表格与曲线。

若两轮主对比出现A/A或baseline漂移超限点，只对各轮原先标记点做一次预先列定的7轮重测，配置/输入/阈值保持原样。保存repair_main/repair_repeat和替换来源映射；同时报告原始完整31点结果与明确链接的修复结果，不按加速比选样本，不循环重试直到通过。仍超限则保留不确定标记。补充跨环境参照不做这种有效性修复，不当作同环境稳定加速结论。

从仓库根目录重放（先确认设备空闲，每次写新目录）：

```bash
CUDA_VISIBLE_DEVICES=0 bash mm/comparison/run.sh
```

`worker.py` 负责实际编译、精度和计时；`analyze.py` 根据完整原始JSON核对输入、冻结cubin和来源后生成报告。初始 smoke 是流程测试，四个完整进程块和两次固定范围重测才是报告数据。编译缓存、原生阶段产物保留在本机忽略目录，实际输入/产物hash与配置进入Git证据。
