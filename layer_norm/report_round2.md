# LayerNorm TTIR 第二轮优化报告

采用 `r2_mixed_stream.ttir`，已更新 `ir/optimized.ttir`，与该候选字节一致。

证据等级 A（完整矩阵与独立复测）：30 个 shape、7轮：算术平均 **1.499×**，几何平均 **1.440×**；同轮上一版为 1.317× / 1.295×。独立进程 seed=123、5轮复测：**1.499× / 1.440×**。

相对parent有 22/30 个shape更快；最大回退为 3.63%（按同轮各版耗时中位数计算），不宣称逐点全部改善。

## 与 LLIR 的实际距离

B级补充观测（单进程3轮）：四版本同 GPU/同 CUDA graph 计时，用**当前原生 baseline 作为共同参考**：LLIR 成熟版算术/几何平均为 **1.452× / 1.412×**，本轮 TTIR 为 **1.500× / 1.440×**。新 TTIR/LLIR 优化版耗时的几何平均比值为 **0.981**（<1表示TTIR更快）。

LLIR 保留 cubin 来自旧离线后端，仅作为性能参考；主实验两份 TTIR 全部经过同一官方 Triton 3.7.0 原生链。该比较没有把旧 LLIR 的较慢 baseline 作为 TTIR 参考，也不把结果外推为层级本身的优劣。

## 保持不变的契约

H100 GPU7 / SM90；官方 Triton 3.7.0、独立 venv、无FlagTree。FP16 X/Y/W/B，FP32累加与Mean/Rstd，中心化方差。8 warps、1 CTA、3 stages、相同签名/grid=(M,1,1)/launcher。N/stride为16倍数、列连续、X/Y同行stride、无别名；主矩阵M=4096，N=1024..15872 step512。只编辑TTIR，无内联PTX、无下游手工修改，无maxnreg调参。

## 实际采用的变换

- 静态范围分支：1024/2048/3072/4096/6144/8192/10240/12288/14336/16384；其他正N保留动态分组循环。
- 将8192宽累加器改为2048宽（部分小N使用1024宽），在分组之间累加后再规约。输出也按分组处理，缩短中间值存活。
- mean加载显式使用TTIR `cacheModifier=ca`，与variance默认load区分，避免公共表达式合并把整行输入长时间保留。它是普通缓存load，不是volatile；实际PTX/SASS由原生后端生成。
- Y store使用TTIR `cacheModifier=cs`，减少流式输出对缓存的干扰；沿用倒数复用与rsqrt。

资源：从parent的80 registers/thread、n_spills=2，变为 **48 registers/thread、n_spills=0**，dynamic shared=32 B。仍使用后端默认规约，不宣称达成LLIR的两次barrier。

## 逐shape计时

下表baseline/parent/new是同轮全量配对测量；LLIR是补充比较中的耗时，单列参考，不能跨测量直接作单点精确归因。

| N | baseline µs | parent µs | new µs | new/base加速 | 独立复测加速 | LLIR参考 µs |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 31.119 | 12.845 | 9.795 | 3.177× | 3.174× | 11.886 |
| 1536 | 31.578 | 15.227 | 11.558 | 2.734× | 2.731× | 12.699 |
| 2048 | 31.910 | 17.056 | 12.230 | 2.609× | 2.604× | 13.263 |
| 2560 | 34.547 | 24.721 | 15.264 | 2.263× | 2.264× | 17.039 |
| 3072 | 36.072 | 29.133 | 20.077 | 1.797× | 1.797× | 20.916 |
| 3584 | 37.046 | 29.548 | 23.883 | 1.551× | 1.551× | 24.277 |
| 4096 | 38.115 | 31.088 | 26.321 | 1.448× | 1.448× | 26.422 |
| 4608 | 43.607 | 36.709 | 29.849 | 1.461× | 1.461× | 31.722 |
| 5120 | 46.044 | 39.131 | 31.892 | 1.444× | 1.444× | 33.451 |
| 5632 | 47.519 | 40.112 | 34.519 | 1.377× | 1.377× | 35.498 |
| 6144 | 48.971 | 41.174 | 37.304 | 1.313× | 1.313× | 38.139 |
| 6656 | 54.651 | 42.981 | 40.494 | 1.350× | 1.349× | 40.815 |
| 7168 | 57.274 | 44.815 | 43.190 | 1.326× | 1.326× | 43.392 |
| 7680 | 59.481 | 46.866 | 45.560 | 1.306× | 1.305× | 45.779 |
| 8192 | 61.392 | 49.157 | 48.647 | 1.262× | 1.262× | 48.170 |
| 8704 | 70.805 | 59.290 | 51.487 | 1.375× | 1.375× | 51.962 |
| 9216 | 72.876 | 60.798 | 54.145 | 1.346× | 1.346× | 54.563 |
| 9728 | 74.506 | 62.426 | 56.932 | 1.309× | 1.309× | 57.158 |
| 10240 | 76.398 | 64.171 | 59.478 | 1.284× | 1.285× | 59.598 |
| 10752 | 81.524 | 66.424 | 64.256 | 1.269× | 1.269× | 63.024 |
| 11264 | 84.423 | 68.067 | 66.826 | 1.263× | 1.263× | 65.764 |
| 11776 | 86.758 | 70.337 | 70.395 | 1.232× | 1.233× | 68.734 |
| 12288 | 88.406 | 73.199 | 73.117 | 1.209× | 1.210× | 71.553 |
| 12800 | 93.342 | 75.092 | 77.718 | 1.201× | 1.201× | 76.552 |
| 13312 | 96.190 | 77.775 | 80.598 | 1.193× | 1.194× | 79.163 |
| 13824 | 98.577 | 80.207 | 83.113 | 1.186× | 1.186× | 81.936 |
| 14336 | 100.170 | 83.252 | 85.853 | 1.167× | 1.167× | 84.293 |
| 14848 | 104.671 | 85.927 | 87.765 | 1.193× | 1.193× | 88.004 |
| 15360 | 107.265 | 88.978 | 91.804 | 1.168× | 1.168× | 91.048 |
| 15872 | 110.601 | 92.072 | 94.451 | 1.171× | 1.171× | 94.096 |

## NCU证据（C级机制线索）

`ncu_batch1` 与 `ncu_finalists` 均为 base clock、cache-control=all，采集 LaunchStats/Occupancy/SpeedOfLight/WarpStateStats/MemoryWorkloadAnalysis。每次被采集launch同时通过Y精度检查。NCU时长不替换graph benchmark。

| N | 版本 | NCU µs | 实际occupancy | DRAM Tbyte/s |
|---:|---|---:|---:|---:|
| 1024 | parent | 18.208 | 33.1% | 0.464 |
| 1024 | new | 13.952 | 55.0% | 0.606 |
| 4096 | parent | 31.360 | 35.4% | 1.613 |
| 4096 | new | 24.576 | 56.1% | 1.905 |
| 8704 | parent | 62.752 | 34.2% | 2.011 |
| 8704 | new | 50.144 | 56.6% | 2.417 |
| 15872 | parent | 89.664 | 33.4% | 2.723 |
| 15872 | new | 97.696 | 54.7% | 2.494 |

低寄存器路径的实际occupancy从约33%~35%升至约55%~57%。中等N时加载依赖隐藏和吞吐改善，但最大N的吞吐/时长仍有回退，不能仅凭occupancy判定更快。第一批volatile重载在大N的long-scoreboard/issue比值显著增加，指导后续改用普通ca加载。该比值是平均stall/issue统计，不是可直接消除的耗时百分比。

## 验证、负结果与测量有效性

- baseline和14个可编译候选/parent在完整142用例上验证；累计记录包含重复baseline，详见accuracy_all.json。额外覆盖新增范围边界±16、padding=0/16、normal/offset分布，共120例/版本，见boundary_accuracy.json。
- 最终版与baseline在120个新增分支边界/压力/padding用例上分别通过memcheck/racecheck；均0 errors，racecheck无warnings。
- affine初版的arith.negf在当前TTIR文件入口lowering失败；以新候选名修复，未覆盖失败IR。动态小分组循环、全输入复用和fold候选存在性能退步，全部保留。16候选预算用完后停止扩展，非无限搜索。
- 首次完整测量的N=1024出现baseline 31~68us大幅漂移，整轮排除并中止，原始partial数据保留。重跑采用所有版本相同的额外untimed graph预热(rep=100ms/版本)，独立复测同样预热。时钟未锁定，缓存未强制清理；报告为预分配重复CUDA graph kernel时间，不含编译、加载、CPU launch或分配。

## 复现与证据

候选生成：`make_round2.py`。原始结果位于`results/round2/`：candidates.json、accuracy_*.json、screen_*.json、full_stable.json、independent_repeat.json、llir_comparison.json、ncu_*.ncu-rep/csv/summary.json、memcheck.log、racecheck.log、selection.json。

```bash
CUDA_VISIBLE_DEVICES=7 bash layer_norm/test_performance.sh
# 直接复现本轮候选，结果写到新的输出路径：
/root/venvs/ir-ops-dev/bin/python layer_norm/make_round2.py
CUDA_VISIBLE_DEVICES=7 /root/venvs/ir-ops-dev/bin/python layer_norm/benchmark.py \
  tile_dispatch_math_v2 r2_mixed_stream --full --rounds 7 --rep 20 --stabilize-ms 100 \
  --validation layer_norm/results/round2/accuracy_finalists.json --output /root/tmp/new_round2_benchmark.json
```
