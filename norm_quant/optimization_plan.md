# Residual RMSNorm + 动态 INT8 量化，r1

本轮是新建、明确指定的融合研究 baseline；参考本地 FlagGems `fused_add_rms_norm.py` 的残差/RMSNorm 数据流和 `act_quant.py` 的动态量化结构，但并非任何生产实现的原样提取。不以分离 PyTorch 算子作为性能基线。baseline 源码冻结后候选只修改 TTIR。

契约：连续 FP16 X/R/W，FP32 residual Z、逐行 scale，INT8 Q，无别名；`z=float(x)+float(r)`，`rho=rsqrt(sum(z*z)/N+1e-5)`，`y=(z*w)*rho`，`scale=amax(abs(y))/127`，全零行 scale=1。Q 是 `round-to-nearest-even(clamp(y/scale,-127,127))`。有限输入，abs(X/R)<=10000、abs(W)<=8，N<=16384。输出 Z 精确，scale vs FP64 rtol=3e-5/atol=1e-9，Q vs FP64 至多差1 code；重构误差 <=0.501*scale + 3e-5*amax(y) + 1e-7。这是容差量化合同，不声称逐位 INT8 等价。

冻结性能点 `(rows,N)`：`(1,4096),(128,1024),(128,4096),(2048,4096),(512,8192),(513,4097),(32,16384),(64,257)`。每点 baseline 在4/8 warps、3 stages中初选，保存原始计时后冻结映射。候选共享同一配置、ABI、grid 和单kernel。精度分布 normal/zero/constant/offset/cancellation/tiny/signed-weight；小行另对 CPU FP64。

候选：`factor_v1` 将 amax(y) 因式分解为 rho*amax(z*w)，两个统计量消除依赖；`joint_v1` 在 factor 基础上把 sum(z*z) 与 max(abs(z*w)) 合并成 tuple reduction；`joint_cancel_v1` 在量化比值中消去正 rho，保持 scale 合同；`stream_v1` 在 joint 基础上测试输出 cs cache hint。最多4种，保留失败和回退。浮点行为改变按上述预先给定合同验收。

环境固定 `/root/venvs/ir-ops-dev/bin/python`、upstream Triton3.7.0、H100 GPU6，禁止并发同卡任务。每次主实验先核验源码/冻结 TTIR 的 LLIR、PTX、cubin 完全一致和实际 options；候选从文件经同一路径编译。精度先于计时。预分配完整单kernel CUDA graph，随机顺序配对、A/A，初筛3轮/10ms，主测7轮/20ms，独立进程复测5轮/20ms，最终 memcheck/racecheck。主测前根据初筛全矩阵几何平均冻结一个候选；主测/复测同时>=1.02×且最差>=0.95×、A/A各点在[0.97,1.03]才晋升，否则保留baseline。报告限定于此8点、此研究baseline，不构成 A/B/C 源码组与IR组因果试验。
