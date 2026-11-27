# MM host TMA 全31点扩展 v1

用户要求把其余shape也用TMA优化。本组将原host TMA/PID方法推广到六个冻结配置组，在同一空闲H100 GPU7上重新验证、测量全部31点，不拼接历史GPU1结果。

固定官方 /root/venvs/ir-ops-dev/bin/python、Triton3.7.0。原tutorial、baseline、配置映射、旧选择/别名、既有bridge均不改。仅新建直接TTIR候选；使用既有局部descriptor签名reader，共享原生backend/launcher，不修改安装文件或下游产物。匹配baseline在原TTIR签名前加同样host descriptor字段，普通load计算体不变；新接口12逻辑/24物理参数，scratch为0。基线真实tensordesc metadata各自保留，不伪造资源。

每组固定原BM/BN/BK/GROUP/warps/stages，最多两个候选：host_tma（原PID公式）和host_pid（行号pid%size）；逻辑dot顺序及K分块不变。输入FP16、正16倍数、列stride1、padding0/16、无别名、偏移<2^31。descriptor store不保证保留逻辑输出区域外padding：测试保留NaN canary，任何覆盖均判失败。编译/资源失败原样保存，不静默改配置；若有硬件限制，仅另建明确版本和匹配新配置baseline，不混入本轮接受。

先fresh verify_pipeline验证31点原源码/文件LLVM/PTX/cubin字节与输出一致。每组基线/两候选验证全部本组性能点、tile/K/group边界、padding、数值压力，以及非初始3行尾组(11*BM,2*BN,BK+16)，对native FP16、禁TF32 FP32，小尺寸CPU FP64；atol=.01、rtol=.001，finite、NaN sentinel、padding canary、输入不变。主测seed42与独立进程seed123分别验证。失败候选不计时。

全部31点配对CUDA graph设备计时，描述符/输入输出提前准备，10次预热后同步；主测7轮、独立进程5轮、rep20ms；同轮随机label及baseline/candidate顺序；保存所有pairs、IR/产物/脚本哈希、环境与GPU UUID。A/A中位比偏离<=2%、baseline max/min<=1.10。若无效，仅对两个进程失败点并集做一次固定7轮重测，不按速度选轮次；失败记录保留，不宣称仍无效点稳定通过。

按完整配置组选择：两个进程geo>=1.02、CI95下界>1、最差点>=.95，计时控制全部有效；主测胜者冻结后独立复测，同时完整保留两候选数据，未达条件回退匹配baseline。不得把未测点写1×实测。记录各方法相对baseline及相对host_tma的直接配对（若未直接配对则仅标派生）。预留输出的原9参数基线设备计时也在同次测量配对保留，用于核查新接口基线影响；包含准备的walltime单列，不能宣称API端到端收益。

每组基线/可运行候选均执行memcheck/racecheck小尺寸边界组，工具实际错误阻止接受。继承此前TMA实验已披露的未解决racecheck warning语境，本次继续性能研究并保留所有warning及非零退出，不称同步资格通过；新数值、内存错误不能忽略。最终选择标sync_qualified=false，工具结果逐组登记。旧生产/研究入口不覆盖，新入口只分派31点与明确研究资格。

新增report.md含31点时间/比值和各组选择，更新all_report.md简明结果。不使用跨环境FlagTree作为纯TTIR收益。pre-commit --all-files；本次不运行外部通信或改变Git历史。
