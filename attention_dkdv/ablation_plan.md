# Attention dK/dV 最终版消融契约

研究对象为已经冻结的第六轮有效选择，所有现存 baseline、候选与 optimized 文件不修改，不根据本轮结果重新择优。沿用 H100 / 官方 upstream Triton 3.7.0、FP16 连续 Q/K/V/dO 与输出、FP32 LSE/delta、无别名、N 为 128 倍数、4 warps / 5 stages / 1 CTA。只计已准备好统计量及预缩放 K 后的 dK/dV kernel，不能外推完整 attention backward。

固定第二轮扩展的全部 32 点：D64/128 × causal/noncausal，每组 `(B,H,N)` 为 `(1,4,128/256/512/1024/2048)`、`(2,8,1024/2048)`、`(4,8,2048)`。SM 锁为 1200 MHz，结束恢复自动频率。GPU 物理 4 主测，物理 5 独立复测；每卡独占且串行。

## 预先固定的对照

| 标签 | 内容 | 来源 |
|---|---|---|
| baseline | 原始 128-key 基线 | `round2/ir/<group>/baseline.ttir` |
| dv_only | 原始 128-key 基线上仅后移 dV | `ir/d128_*/r1_dv_late.ttir` |
| rows64 | CTA 串行处理两个 64-key 子块 | `round2/ir/d128_*/rows64_v3.ttir` |
| structure | rows64 + 非因果指针重算 / 因果子块静态展开 | `round2/ir/d128_*/rows64_remat_v3.ttir` / `rows64_unroll_v5.ttir` |
| rows64_dv | rows64 + dV 后移，移除最终版的指针重算或展开 | `ablation/ir/d128_*/rows64_dv_v1.ttir` |
| final | 冻结 r6 最终有效版本 | `multiround/r6/ir/<group>/optimized.ttir` |

D128 测以上六路；D64 最终版只有 dV 后移，测 baseline / final 两路。`make_ablation.py` 只对现存 rows64 TTIR 调用同一 dV 重排变换，生成新名字的不可变候选，并核对 structure 经相同变换逐字节等于 final。D64 final 必须等于首轮 r1_dv_late。不是修改 Python kernel 后重新导出候选。

逐项加入：baseline → rows64 → structure → final。最终移除 dV：structure → final；移除指针重算或展开且保留 dV：rows64_dv → final；联合移除 64-key 与依赖它的结构调整：dv_only → final。最后一项不能解释成纯 64-key 贡献。额外测量 rows64 → rows64_dv，观察 dV 重排是否依赖结构。条件贡献和交互不能相加。

## 数值门槛与性能协议

每次主实验先运行 `round2/verify_pipeline.py`，核对 32 点冻结源码/TTIR 入口 LLVM、PTX、cubin 及输出一致。随后一次性完成 44 输入上的 176 项候选检查，全部通过后才计时：32 正常 shape 加四组 N128 的 zero_dO、constant、large 压力。检查 dK/dV 有限、NaN sentinel 完整覆盖、输入不改，以及与源码基线的 int16 位型逐位一致；对原始 Q/K 的 FP64 数学参考固定 `rtol=0.03, atol=0.008`。

主测 seed20 / 7轮；独立进程及设备复测 seed123 / 5轮。每路初始以 CUDA event 验证实际预热≥500ms，每次测量前≥100ms；每轮所有版本及 baseline 副本随机交错，`do_bench_cudagraph(rep=20)`。A/A 每点配对比中位数须在 0.97..1.03。每点先计算同轮配对比值的中位数，再取各组或全矩阵几何均值。保留所有时延、顺序、实际预热时长、环境、输入哈希、TTIR/LLVM/PTX/cubin 与 .text 哈希、寄存器/spill/shared资源。

两次完整测量通过称 A 级性能证据；静态资源为 C 级。小差异只表述观测，不以寄存器下降证明因果。本补测没有重新运行 sanitizer；最终路线的 memcheck/racecheck 使用 r6 已存证据，新增 rows64_dv 只承担正确性与性能消融。旧 D128 加入实验与所有负结果完整保留在 `ablation/` 和历史轮目录。

## 复现

从仓库根目录执行，设备和输出目录使用实际空闲卡及新路径：

```bash
bash attention_dkdv/run_experiment.sh 4 20 7 attention_dkdv/results/runs/ablation_main_new
bash attention_dkdv/run_experiment.sh 5 123 5 attention_dkdv/results/runs/ablation_repeat_new
/root/venvs/ir-ops-dev/bin/python attention_dkdv/report_ablation.py \
  --main attention_dkdv/results/runs/ablation_main_new/measurement \
  --repeat attention_dkdv/results/runs/ablation_repeat_new/measurement \
  --output attention_dkdv/results/runs/ablation_summary_new
```

报告脚本输出 Markdown、CSV 与带来源哈希的 JSON。新目录必须不存在；不覆盖历史样本或旧候选。
