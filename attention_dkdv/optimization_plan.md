# Attention backward dK/dV 第一轮契约

源头：官方 Triton v3.7.0 `06-fused-attention.py`，完整快照和 SHA256 见 upstream/provenance.json。研究基线仅提取原 fused backward 中的 dK/dV 段，完整保留概率恢复、dP/dS 和 dK/dV 的四个 dot，去掉后半 dQ；与完整官方 backward 做逐位对照。不是全 attention backward 或 forward 的计时。冻结后只修改 TTIR。

FP16 连续 Q/K/V/dO/dK/dV，FP32 LSE(log2) 与 delta，输入有限且无别名；N 为128倍数，D为64或128。K 是原教程所需的预缩放 FP16 K（sm_scale / ln2），M 和 delta 已备好。前向、预缩放 K、delta 计算不计入设备时长。主机被测路径仅 launch；PyTorch 计算只用于测试和参考。sm_scale=1/sqrt(D)。

四个 IR 组：D64/128 × causal/noncausal；baseline 固定官方 num_warps=4,num_stages=5,num_ctas=1，BLOCK_M1=32,BLOCK_N1=128,BLK_SLICE_FACTOR=2。没有逐 shape autotune，不称最强 baseline。每组完整性能矩阵为 (B,H,N)=(1,4,256),(1,4,1024),(2,8,2048)，合计12点。另 N128 小边界、zero_dO、constant、large QK 压力验证；不支持 ragged N/padding/dropout/FP8。

先运行 verify_pipeline.py 检查 source/file LLVM/PTX/cubin 与实际输出，再生成候选；所有候选同 options、grid、ABI、编译入口。候选预算：整数非负 div/rem 改 unsigned、M/delta 加载前移与 dot 重排、dV 更新后移、load cache 策略及一项组合。先检查 TTIR 是否含对应结构；没有的记录未适用。

数值门槛事先固定：相对 FP64 未量化数学参考 dK/dV，torch.testing.assert_close(rtol=0.03,atol=0.008)；另与 baseline 逐位/误差报告。正常随机 std=0.5，large QK std=1.5；有些上游 FP16舍入误差可超数学参考门槛，失败必须保留而不以放宽阈值接受。小例 CPU FP64 autograd 交叉检查公式。输入不可改、输出NaN sentinel必须全部覆盖且有限。

初筛每点3对/rep=10ms，完整主测每点7对/rep=20ms，独立进程新seed复测5对/rep=20ms，CUDA graph replay，随机对内顺序，保存全部 samples/order。A/A 控制至少3对。组级选择按初筛几何均值，至少1.02×且组内最差>=0.95×；若没有候选过门槛仍完整测最有希望候选用于负结果，最终选择baseline。不能把12点结果外推其他矩阵。最终 memcheck/racecheck，记录warning和error分别计数。
