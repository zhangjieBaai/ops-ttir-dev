# 设备TMA独立工作区分组：未接受

本组是按实验要求另建的**匹配allocator/新工作区约定**实验，
不覆盖旧固定workspace分组。
实际启动的设备TMA候选数值检查通过，
但**racecheck报37条warning，
进程退出99**，
且完整六点主测/独立复测加速比仅 **0.703× / 0.700×**。
候选未通过验收，
保留baseline，
未改旧入口或optimized.ttir。


加速比方向为baseline耗时/candidate耗时。
0.70×对应candidate耗时约为baseline的1.43倍；
本次几何均值换算约增加42.3% / 42.8%耗时。
下面是**被拒绝候选的观测时间**，
不是一个已验证可用实现的收益证明。


## 匹配的执行条件

原baseline与candidate都通过本组同一个compile_native(path)，
相同options（64×128×64、group8、warps4、stages4、CTAs1）、官方Triton3.7.0、同一物理H100 GPU1、相同9参数和common.launch启动。
baseline是原教程TTIR的字节副本，
candidate是上一轮设备descriptor编译诊断的字节副本。
初始诊断在旧分组没有执行；
仅本组使用新allocator约定后才执行，
不重写旧历史。


两版都在同一个allocation_scope中注册相同的官方triton.set_allocator回调，
退出及异常路径恢复原allocator。
回调仅torch.empty(uint8)分配，
逐次检查真实launch stream与Torch当前stream/device一致及alignment满足要求。
native driver按metadata真实请求：baseline0，
candidate **384bytes/CTA、alignment128**；
没有额外手传workspace参数或二进制/metadata替换。
1408³时grid242，
对应92928bytes；
callback最大记录可核对。
baseline实际不需要该buffer，
不伪造相同资源使用。


候选TTIR在device内构造A/B/C三个tensor descriptor，
用descriptor_load、dot、descriptor_store。
相对FGhost descriptor，
它多了每CTA的设备描述符构造工作。
此差异是静态结构事实，
尚未通过受控消融隔离其全部耗时贡献，
不能将全部性能差额直接归因。
资源为90registers、0spill、98336bytes shared，
仍不能据此预测更快。


## 原始数据

| M=N=K | baseline µs | device-TMA µs | 主测比值 | 独立复测 |
|---:|---:|---:|---:|---:|
| 768 | 4.919 | 7.070 | 0.69542× | 0.69530× |
| 896 | 5.535 | 7.646 | 0.72380× | 0.69272× |
| 1024 | 5.943 | 8.327 | 0.71388× | 0.71359× |
| 1152 | 9.271 | 13.147 | 0.70555× | 0.70476× |
| 1280 | 10.804 | 14.728 | 0.73331× | 0.73413× |
| 1408 | 11.510 | 17.808 | 0.64906× | 0.66356× |

每点7轮主测、seed123独立进程5轮复测，
rep20ms，
配对随机顺序，
同一baseline kernel另做A/A；
全部计时控制有效。
设备graph replay包含描述符构造、fence、TMA/load/dot/store全部设备工作，
CPU allocator、JIT、调参、模块加载、输出分配不计入；
不能将该指标当作含分配的端到端时间。
未锁频或核实锁频，
无主动cache flush，
不声称持续独占或全31点收益。
四点初筛已明显回退，
因此没有继续尝试仅改变pid公式的第二候选；
未用预算强行增加同类试验。


## 数值、内存与同步必须分开

四个check进程各baseline37项+candidate37项，
共 **296次**数值检查通过，
包括全部六点、数值压力、padding、K/tile尾部和704×256×80非初始partial group；
native FP16、禁TF32 FP32、小尺寸CPU FP64，
atol=.01/rtol=.001，
NaN sentinel、canary、input不变与finite检查均执行。


- memcheck：工具日志0errors。

- racecheck：37条显示的hazard报告，
0errors、37warnings，
退出99。
源码标签指向device_descriptor_probe.ttir:36附近descriptor load；
cubin缓存命中仍可能保留首次编译的诊断文件调试标签，
实际新输入hash/产物hash一致。
每项报告可聚合多个底层hazards，
不能把37报告数当作底层访问对总数。

- 目前未确认是实际同步缺陷还是工具对TMA异步语义识别的限制；
**不能自行当作误报忽略**。
数值PASS不清除该warning，
不授予内存/同步资格，
也不接受候选。


最终研究入口 [api.py](api.py) 默认且仅加载已保留baseline，
核对本组contract和selection。
6点实际调用与不支持shape拒绝通过。
另一个明确标记为未获资格的candidate graph数值/allocator生命周期探针，
记录到capturing=True的真实stream和alignment，
退出恢复allocator，
见 [api_contract.json](results/api_contract.json)；
这不清除racecheck警告，
也不发布candidate为可用优化。


## 证据和复现

[plan.md](plan.md)、[contract.json](contract.json)声明新的匹配工作区组；
[engine.py](engine.py)拥有sharedcompiler/scopedallocator，
[experiment.py](experiment.py)负责pipeline/37项验证/配对计时；
[selection.json](results/selection.json)明确REJECTED_PERFORMANCE_AND_RACECHECK。
初次standalone导入顺序错误的setup日志保留，
实际数据以pipeline_v2.json与check_v2为起点；
后续api contract导入错误也保留，
最终入口以api_contract.json为准，
不将setup错误算作GPU算术失败。


核心证据为results中的pipeline_v2.json、check_v2/、full.json、repeat_check/、repeat.json、memcheck.log、racecheck.log、api_contract.json。
全fingerprints、allocator记录和source/IR/cubin hash保留；
阶段产物在父MM artifacts忽略目录中可再生。
旧baseline/protocol、六组映射、round1选择及别名保持原字节。
当前没有应晋升的TMA版本。


```bash
# 先确认空闲设备；执行研究候选会保留racecheck失败，不代表通过验收
CUDA_VISIBLE_DEVICES=1 bash mm/tma_workspace/replay.sh
# 仅从原始证据重新生成本报告，不运行kernel
/root/venvs/ir-ops-dev/bin/python mm/tma_workspace/report.py
```
