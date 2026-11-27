# INT4 窄加载、K mask 与量化组复用消融

复用第二轮冻结的 30 个 shape 和已有 TTIR。以 parent 为共同起点，将 narrow 与 full_k 组成 2×2 对照，再在 narrow_full_k 上加入 group_pair。每个 shape、每轮对两条路径的耗时直接相除，取中位数后在 shape 间求几何均值。最后一步在 BK128 上保持原实现。目的为估算条件收益，最终实现及晋升决策沿用第六轮。

从项目根目录执行以下命令，每次使用新的输出名，GPU 编号按设备空闲情况设置。

```bash
export CUDA_VISIBLE_DEVICES=7
P=/root/venvs/ir-ops-dev/bin/python
R=int4_gemm/round2
N=baseline,parent,r2_narrow,r2_full_k,r2_narrow_full_k,r2_group_pair
$P $R/verify_pipeline.py --output $R/results/ablation_v1_pipeline.json
$P $R/run.py --names "$N" --run ablation_v1_main --rounds 7 --rep 20 --seed 42
$P $R/verify_pipeline.py --output $R/results/ablation_v1_repeat_pipeline.json --seed 123
$P $R/run.py --names "$N" --run ablation_v1_repeat --rounds 5 --rep 20 --seed 123
```
