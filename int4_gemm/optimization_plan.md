# INT4 fused GEMM 首轮冻结协议

来源为本地 FlagGems `_weight_int4pack_mm_with_scales_and_zeros.py` 的完整设备计算段，
仅移除 `libentry` 和主机注册，`tle.program_id` 绑定到 upstream `tl.program_id`。
它本身融合解包、groupwise 反量化和 Tensor Core GEMM，不生成全量反量化显存中间量。
这是该实现的 TTIR 研究，非生产接口注册改动；不声称超过所有 INT4 库。

输入 A/scale/zero 为连续 FP16，packed 为 `[N,K/2]` uint8，低 nibble 对应偶 K。
`W[k,n]=FP16((float(q)-float(zero))*float(scale))`，C 为 FP16，FP32 dot 累积。
group 固定 128；K 为 128 的倍数、N 为 16 的倍数，M 可非对齐，无别名。
数值范围有限、输出无溢出；不支持 NaN/Inf。输出相对 FP32 ATen matmul 参考
`atol=0.035, rtol=0.003`，小用例再用 CPU FP64 验证。
候选必须和 baseline 逐元素一致（cast 位置保持舍入点；不重排 reduction）。

性能矩阵 `(M,N,K)`：`(16,4096,4096),(128,11008,4096),(1,256,128),
(32,1024,1024),(64,2048,4096),(128,4096,4096),(17,80,256)`。
每点先测试四组 `(BM,BN,BK,warps,stages)`：`(32,64,64,4,3)`、
`(16,64,64,4,3)`、`(32,128,64,4,3)`、`(32,64,128,4,3)`；
仅从该预定集合选择基线，并保存所有测量。基线冻结后候选配置相同。

直接修改 TTIR 的候选预算：共享 low/high 的 group/scale/zero；把 group 索引
标量化为 k0/group 后广播（BK 整除 group 且 loop 对齐）；非负索引除余变 unsigned；
cast 移至 join/reshape 后（可能妨碍 MMA fold）。最多组合共享与标量化及 unsigned。
不修改 Python kernel 生成候选，不改 TTGIR/LLVM/PTX/cubin，不内联 PTX。

先逐点核查源码→TTIR 文件入口 LLVM/PTX/cubin 字节一致、完整 options 一致，
再做全部数值/压力用例。初筛 3 配对轮、10ms CUDA graph，主测 7 轮、20ms，
独立进程复测 5 轮、20ms，保存顺序、全部样本、哈希与寄存器/shared/spill。
选择全矩阵候选，以主测几何平均至少 1.02 且最差点至少 0.95 为晋升标准，
复测不得重新选择。最终 memcheck/racecheck；失败阻止晋升。失败、退步保留。
计时包括单个 fused kernel，排除输入构造、参考、分配、编译、加载和 Python launch；
graph 重放缓存状态不代表冷缓存。所有结果仅针对此 7 点和 H100，未锁 GPU 时钟。
这是直接 TTIR 实验；没有 A/B/C 盲测，不能证明 IR 信息优于源码信息。
