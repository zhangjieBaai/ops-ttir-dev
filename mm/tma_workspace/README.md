# 设备TMA独立工作区实验

[报告](report.md)记录实际结果：候选未通过性能或racecheck验收，保留baseline。该实验不替换原MM选择，不声称TMA收益；固定工作区的旧诊断历史仍保持“当时未执行”。

这是遵循实验要求第6节另建的匹配运行分组。两版共用native allocator设置，原教程baseline实际scratch0，设备TMA候选由原生driver按metadata申请384bytes/CTA。官方Triton3.7.0、配置和9参数入口、grid不变，所有设备描述符构造与访存工作计入graph设备时间。host分配不在此kernel指标中。

输入/选项身份见 [contract.json](contract.json)，规则见 [plan.md](plan.md)。`engine.py`负责sharedcompile与可恢复allocator，`experiment.py`负责验证/计时，`api.py`仅提供被保留的baseline入口；研究TMA候选只能从显式实验runner运行。`verify_api.py`核查baseline6点和未知shape拒绝，并单独记录一次未获资格候选的graph stream/对齐/恢复数值探针。

```bash
# GPU空闲时完整重放；racecheck失败会被保留为非零状态，不自动晋升
CUDA_VISIBLE_DEVICES=1 bash mm/tma_workspace/replay.sh
# 只复算已有记录的报告
/root/venvs/ir-ops-dev/bin/python mm/tma_workspace/report.py
```

证据在results/。`pipeline.log`和`check.log`为初始导入设置错误，实际完整数据从pipeline_v2.json/check_v2开始。`api_final.log`也是已修复的导入设置错误，最终入口证据是api_contract.json。错误记录保留，不计入数值失败或成功用例。候选数值PASS不清除racecheck的37条warning，不能作为已验证优化使用。
