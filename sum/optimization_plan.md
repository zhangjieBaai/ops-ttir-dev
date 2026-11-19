# Sum TTIR v1：冻结契约与实验协议

来源为 FlagGems commit ef8aea78fed8321c3e02a15bf31e847cd0579e4f 的
`src/flag_gems/ops/sum.py`，只研究 `sum()` 的非空、连续 FP32 全量两阶段归约。
两份 kernel 的计算函数体和 `ext.program_id` 原样提取；仅移除 libentry 主机缓存装饰器，
隔离导入避免加载系统 FlagTree。AST 验证和快照哈希证明提取关系。不是官方教程或 autotune 路径。

- 输入、中间结果、输出均 FP32；FP32 树形累加。输入有限且其绝对值总和不溢出 FP32。
  特殊值 NaN/Inf 另以分类一致性测试。禁止别名，地址至少 16 字节对齐。
- shape 为连续一维 `(n,)`，不支持空输入、非连续输入、dtype 参数、dim、out 或整数。
  只接受冻结映射列出的输入；测试与性能均走实际 TTIR 文件，不回退源码。
- 保留原 host 公式 `B=next_power_of_2(ceil(sqrt(n)))`、`S=ceil(n/B)`、
  `C=next_power_of_2(S)`，第一 kernel grid=(S,1,1)，第二 grid=(1,1,1)。
  原实现没有 autotune；按每个 shape 真实导出 TTIR，按内容和完整 options 合并重复组。
- 两阶段 options 均 num_warps=4, num_ctas=1, num_stages=3,
  enable_fp_fusion=True，保留官方默认其余配置；比较解析后全部 options 和 SM90 target。
  仅允许普通 TTIR 改写，不用内联 PTX，不改变签名/grid/工作区/编译配置，不融合 kernel。
- 使用 `/root/venvs/ir-ops-dev/bin/python`，官方 Triton 3.7.0、H100，GPU 7 隔离。
  编译缓存允许，禁止 override/ir_override 和 stage inspection hook。

## 矩阵与精度（候选之前冻结）

30 个性能 n，逐点等权：
`1, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 511, 1024, 4096,
16384, 65535, 65536, 65537, 262144, 1048575, 1048576, 1048577,
2097152, 4194304, 8388608, 16777215, 16777216, 16777217, 33554432, 67108864`。
范围包括单元素、小尺寸、整除属性、平方根分块跳变、尾部和大工作集。
它是自建性能集，并非上游性能集。

初筛 tuning n：`16, 127, 1024, 65535, 65536, 1048576, 1048577, 16777215,
16777216, 33554432, 67108864`；其余 19 点用于首轮 transfer 检查。
近邻边界组全部放 tuning 或 validation：65535/65536/65537 中 65537 为正确性重点，
因跨越 block 配置，仍披露全矩阵只有分阶段验证，不声称统计独立 holdout。
不在看到 validation 后继续形状选择；若改变候选，披露 exposure。

精度以测试侧 CPU FP64 sum 与 native ATen FP32 sum 为参考，最终输出和中间部分和都检查。
有限输入容差 `abs(error) <= 2e-5 + 2e-6 * sum(abs(x))`，两种参考均满足该界；
这是按输入 L1 范数缩放的归约误差标准，避免抵消接近零时不稳定的相对误差。
同时报告实际绝对误差、归一化误差，不把此界称为逐位一致。
压力集：全零、常数、交替正负、large offset、小量级、单脉冲；所有性能点随机输入。
额外 NaN、+Inf、-Inf、混合 +/-Inf 分类检查。每次重置 mid/out sentinel，
中间与输出尾部各 16 元素 canary，核查输入未写。最终 memcheck/racecheck 全矩阵。

## 顺序、预算与接受

先跑通源码→导出→文件重编译→执行→数值→基线计时，归档并提交 baseline 后才生成候选。
逐 shape 验证 TTIR 字节匹配、规范化 TTGIR、LLVM/PTX/cubin 字节一致和两个输出逐位一致。
候选先精度后性能；核查实际 TTIR 哈希后计时，保留所有失败和负结果。

第一批最多 5 个候选：i64→i32 安全索引（所有偏移 <2^31）、reshape 分组归约、
load .cg、load .cs，以及有意义的组合；编译或正确性失败保留，不计为性能失败。
若前三个有效新方向的几何平均收益都小于 1.05，暂停参数变体，检查产物与瓶颈再决定。
本轮最多两批共 8 个候选，GPU 测量预算 20 分钟，整体实现和审查预算 90 分钟。

主指标为预分配 FP32 mid/out 的两个 kernel 总 CUDA graph replay 设备时间；
使用同一 `triton.testing.do_bench_cudagraph`，不含 Python launch、分配、编译和加载。
缓存不清理，重复 graph，时钟不主动锁定，记录查询状态；不是应用端到端或冷缓存结果。
初筛 3 轮 rep=20ms；最终全部 30 点 7 轮、独立进程 seed=123 复测 7 轮。
每点含 baseline A/A 对照，基线/候选每轮随机对内顺序，保存所有 replay 样本和顺序。
报告逐点配对比中位数、最差值、算术/几何均值和 95% bootstrap 区间。
主测 A/A 中位数允许 [0.97,1.03]；超出则标记无效并调查，不挑选样本。
以全矩阵几何收益为选择标准，需主测与复测均 >1.01 且最差 >=0.95；否则保留 baseline。
允许诚实的无收益结果，禁止将初筛 B 级结果称为最终 A 级收益。
最后完成 sanitizer、pre-commit、KernelForge close review 和 Git 证据归档。
