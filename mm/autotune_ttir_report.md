# 官方 31 shape 的 autotune TTIR 数量实测

本次31个官方方阵 shape（M=N=K=256…4096，步长128）各自启用官方16配置 autotune，最终获选 **6份不同TTIR**（对应6套获选配置）。所有尝试的配置共有 **16份不同TTIR**；其中 **11份可加载执行**，另5份因shared-memory OutOfResources拒绝。5×31=155条拒绝记录保存在 configuration_inventory.json。

按TTIR完整文本SHA256去重，包含原始loc信息，没有去loc、归一化或修改IR。逐一核查31×16个shape/config组合：同一配置跨31个shape均得到同一个TTIR hash；因此并非31份，也不是31×16份不同内容。JIT编译缓存允许命中，统计的是不同内容，不宣称每次都重新执行了编译阶段。资源超限配置的TTIR通过warmup编译产物核查，随后在原生_init_handles阶段拒绝；它们不属于可运行配置。

| M=N=K | BM×BN×BK | warps | stages | TTIR SHA256前12位 |
|---|---|---:|---:|---|
| 256 | 32×64×32 | 2 | 5 | eb3f6b0c90f8 |
| 384, 512, 640 | 128×32×64 | 4 | 4 | 9998ed02ee84 |
| 768, 896, 1024, 1152, 1280, 1408 | 64×128×64 | 4 | 4 | af7b105bfb51 |
| 1536, 1664, 1792, 1920, 2048, 2560, 2688, 2816, 2944, 3072, 3200, 3328, 3456, 3584, 3712, 3840, 3968, 4096 | 128×256×64 | 8 | 3 | f72ab9f12b9e |
| 2176 | 64×256×32 | 4 | 4 | 9260dfaf694f |
| 2304, 2432 | 128×128×32 | 4 | 4 | f62a60ac3198 |

全部GROUP_SIZE_M=8，num_ctas=1，无activation。M/N/K和行stride仍为运行时参数；tile配置是编译期常量。同一配置下这些shape具有相同签名/对齐前提，所以能共享TTIR。

测量环境：指定venv、官方Triton3.7.0、H100 SM90、物理GPU6。每个shape使用全新autotune key，禁用autotune磁盘结果缓存，使用官方默认计时（未替换为CUDA graph计时），每次shape前后检查其他compute进程。获选kernel均执行并与native ATen FP16比较，31点全部通过，容差atol=1e-2、rtol=1e-3。获选数量/具体配置会随测量噪声及设备状态变化，本次统计不是永久固定映射，也不作为优化收益报告。

证据目录：[autotune_count_20261118T135029Z](results/runs/autotune_count_20261118T135029Z/summary.json)。保存31个shape逐点配置、原始autotune计时、获选TTIR/cubin哈希、完整16份TTIR、496条候选编译/加载记录、环境和日志。原fixed baseline及其协议哈希核查通过，未替换原baseline.json/ir/baseline.ttir，未开始TTIR优化。

复现（先确认GPU空闲，输出目录必须不存在）：

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONNOUSERSITE=1 TRITON_CACHE_AUTOTUNING=0 LD_LIBRARY_PATH= \
  /root/venvs/ir-ops-dev/bin/python mm/count_autotune_ttir.py \
  --output-dir mm/results/runs/autotune_count_NEW
```
