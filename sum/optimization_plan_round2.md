# Sum round2：10%平均收益目标

2026-11-19；用户授权继续优化，目标平均加速比超过10%。继承v1的FP32全量sum数值语义、30个性能shape、
CPU FP64/native ATen参考、L1误差界、Triton3.7.0官方venv和SM90环境。
原baseline/source/manifest不可变，旧optimized别名/结果不改写。

## 两个明确区分的实验组

1. `r2_tail`：严格TTIR改写。保持原两阶段签名、grid、mid、options、映射，
   stage1完整CTA做无mask加载，只有尾部CTA使用原mask，stage2原样。
   与原baseline使用同一个compiler/launcher。单项正确性先于初筛。
2. `sum_launch_v2`：根据诊断另建的启动契约实验。原两阶段sum仍是逐shape匹配baseline，
   新候选对 `n<=16384` 使用一个CTA完成全量归约（一个kernel，输出直写out）；其他shape原两阶段不变。
   新kernel只从冻结stage1 TTIR直接重写张量宽度为next_power_of_2(n)，grid=(1,1,1)，
   第二指针绑定out，保留原constexpr特化形成的runtime参数个数。不给Python kernel改算法/重导出，
   不改TTGIR/LLVM/PTX/cubin，不用内联PTX。num_warps=4等完整options不变。
   **第二组改变启动序列和中间工作区语义，不能声称固定启动契约只改TTIR；收益单独报告。**
   有用输出为out；单kernel路径不承诺旧mid部分和，将验证mid完全未写、out尾部未写和输入未写。
   baseline与candidate共享输入/输出约定、编译入口/原生runtime/计时函数；各自实际完整设备序列全部计时。

先做第1组。如收益不足目标，完成第2组；不得把它的结果混成第1组收益。
新mapping独立保存在round2_manifest.json（shape→variant→文件/哈希/grid/参数选择/options），不改变v1映射。
原两阶段source/file baseline逐30点pipeline在本轮设备重验，所有候选检查完整解析options和target。

## 测试和选择

GPU运行前查空闲卡，显式CUDA_VISIBLE_DEVICES，UUID规范化设备锁，周期监控同卡其他进程/频率/功耗。
候选先精度：原83例+新路径边界（n=15/16/31/32/63/64/127/128/255/256/511/1024/4096/16384），
单kernel新增所有小shape压力与NaN/Inf用例；每次reset sentinel、mid/out canary、输入未写。
新路径映射外输入拒绝，不猜测编译配置。

第1组初筛使用v1已有11点，3轮rep20ms。第2组阈值事前固定16384，不由全矩阵逐点挑最快产生dispatch。
若首个单kernel版本有极端回退，可另命名新候选/阈值协议，披露已经看过的shape，不能称独立holdout。
小输入各特化对应新的TTIR文件与baseline对应关系均保存；重复组实际比对后才共享。

正式主测完整30点7轮rep20ms，baseline/candidate交错随机对内顺序，原triton.testing.do_bench_cudagraph，
保存全部replay值和A/A。相同输入、预分配输出、重复buffer、不清cache、不锁频，编译/加载/预热排除。
seed42主测、独立进程seed123复测7轮，各轮检查实际TTIR/cubin与accuracy/pipeline记录一致。
两个平均都报告：等权几何平均为主目标，算术平均辅助。主测与复测几何平均>1.10、最差>=.95、
A/A中位[.97,1.03]、无监测到同卡其他进程才达标。
bootstrap95%区间按逐点轮次重采样，保留局部退步，不能只取快点。
如果新路径达标但第1组不达标，报告只能说launch_v2达标，不能说原严格TTIR组达标。

主测后memcheck/racecheck覆盖全部30点和单kernel尾部/压力分类输入，失败不接受。
选中版本哈希同时绑定数值、pipeline、两次timing、sanitizer，修复旧select.py潜在未绑定hash的未来晋升风险，
使用独立round2入口，不改旧历史选择。最终pre-commit全量检查、KernelForge close review、Git仅提交sum范围。

## 预算和证伪

最多6个候选（tail、single16384、必要修复/组合），初筛3轮；最多两次完整新候选主测，最终1次独立复测。
若相同方向3次有效收益<5%，回看诊断，不穷举cache/寄存器变体。大shape保留baseline以避免带宽饱和下的无谓退步。
成功后停止搜索，不做不必要profiling。GPU预算20分钟，整体实现/审查90分钟；无收益、失败证据全部保留。
