# MM host TMA 全31点研究组

继承六点host descriptor实验的方法，将其推广到原六组配置、全部31个方阵点。

- 当前候选输入：[inputs_v4.json](inputs_v4.json)。初版输入注册与准备记录保留，见[执行说明](execution_notes.md)。
- 方法：`host_tma`保留原PID公式，`host_pid`改为pid%size；每组均有同host ABI的原教程普通load baseline。固定各组原tile/warps/stages，不重新autotune。
- 性能：描述符/输出提前准备的CUDA graph设备时间；原始main/repeat及一次固定失败范围重测全部保留。CPU调用walltime单列。
- 资格：数值与memcheck检查独立记录。racecheck warnings仍保留，所有选中候选均为`sync_qualified=false`的研究入口；未修改旧mm/optimized.py和旧选择。
- 结果：[报告](report.md)、[机器可读选择](results/selection.json)。公开研究入口：[api.py](api.py)，提供load、prepare_call、run_prepared、matmul，只接受31个冻结方阵点。

重放时必须选空闲H100并设置CUDA_VISIBLE_DEVICES，使用/root/venvs/ir-ops-dev/bin/python、PYTHONNOUSERSITE=1、LD_LIBRARY_PATH为空。每次输出必须为新文件，不覆盖冻结结果。先运行mm/verify_pipeline.py核对当前GPU、官方环境及31点源码/文件一致性，再运行experiment.py的check、bench、sanitize。该脚本当前绑定本组已记录pipeline结果，换GPU需要另建新实验组，不能把旧pipeline当作新GPU证明。

本次精确阶段顺序见[run_measurements.sh](run_measurements.sh)、[run_repairs.sh](run_repairs.sh)；它们含已存在的输出路径，不应原样执行覆盖历史文件。只复算报告不启动GPU：

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONNOUSERSITE=1 LD_LIBRARY_PATH= \
 /root/venvs/ir-ops-dev/bin/python mm/host_allshape_v1/analyze.py report
```

```python
# 仅研究用途，准备后的输入/输出地址、shape/stride和descriptor生命周期必须保持不变。
state = api.load()
y = api.matmul(a, b, state)
```
