# Host descriptor TTIR 可行性原型

**编译、加载、设备执行与数值流程可行；同步检查尚未通过，未进入性能优化。** 最终结论见 [report.md](report.md)。本目录没有发布或替换现有 `optimized.ttir`。

这是一个明确改变host传参契约的独立实验，不属于旧固定9参数ABI的同轮结果。实际执行环境为官方Triton3.7.0/原生backend/runtime，加一个局部、可恢复的TTIR文件签名reader适配；不能称作原封不动的官方file入口支持host descriptor。原始编译后产物、metadata、native launcher及CUDA descriptor编码均未修改。

- `bridge.py`：仅声明且hash匹配的二维FP16输入可归一化，构造IRSource时折叠descriptor及其shape/stride字段；同输入raw/adapted的TTGIR、LLVM、PTX、cubin字节一致，并验证native实际展开顺序和数量。
- `make_prototypes.py`：FGhostIR只作能力参照；自己的host版本直接从已有device-TMA TTIR删除device描述符构造、增加descriptor入口；matched baseline保留原教程计算，加入相同descriptor+rawpointer参数envelope，双方无额外虚假GPU工作。
- `probe.py` / `final_checks.py`：数值、边界、压力、canary、输入不变、完整options、zero scratch、37项三路输出逐位相同及reader恢复/失败反例。
- `source_controls.py`：native源码控制不经过reader适配，仍重现race warning，不能据此证明适配器导致告警，也不能据此宣布误报。
- `make_sync_probes.py`：两个明确的同步诊断（不是正式候选）；一个未改机器码，另一个更改单循环stage后仍报warning。

原始三份输入在 `ir/`，来源和输入hash见 `results/prototype_inputs.json`；两个附加诊断用不可变的v2 registry，旧registry未覆盖。最早standalone导入错误保留，实际初始执行从 `small_v2` 开始。旧版bridge/probe源快照在results中，与当时运行hash对应；后续记录明确新的版本，不重标旧证据。

```bash
# 先确认GPU空闲；预期racecheck有非零状态，不能当作最终资格通过
CUDA_VISIBLE_DEVICES=1 bash mm/host_descriptor/replay.sh
# 已有证据的最终ABI与三路输出检查（新输出路径）
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 LD_LIBRARY_PATH= \
 /root/venvs/ir-ops-dev/bin/python mm/host_descriptor/final_checks.py --output /root/tmp/mm-host-final-new.json
```

计划为 [plan.md](plan.md)。未修改Triton安装文件、Python计算kernel、任何下游IR/binary或旧baseline/config/shape映射。全局IRSource构造器只在单线程实验的受锁context内临时包装；这不是多线程库级生产集成。原参数包和底层Tensor在设备启动/同步期间保活；本轮不承诺新stream/cache/生命周期场景均已验证。
