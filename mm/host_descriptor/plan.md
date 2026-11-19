# Host descriptor 可行性先行实验 v1

用户明确要求先评估能否直接改TTIR为host descriptor并跑通流程，成功后才作为优化方向。旧optimized.ttir、baseline.json/绑定文件、六组配置/映射与旧选择不改。官方Triton3.7.0/指定venv、H100/空闲GPU1、FP16、64x128x64/group8/warps4/stages4/CTAs1，输入/数值标准沿用MM冻结契约。

## 阶段A：ABI集成原型，非收益排名

1. 保留原始直接文件编译失败复现：存储的FGhostTTIR可编译但IRSource.signature包含MLIR类型字符串和前端已展开的shape/stride，nativeCudaLauncher.expand_signature不认该形式；不把它误报GPUkernel失败。
2. 实验局部scoped IRSource.__init__签名适配：调用原生构造器先按原流程解析TTIR，再将已识别二维f16 descriptor及紧随的i32,i32,i64,i64字段折叠为`tensordesc<fp16[BM,BK]>`逻辑参数；标量/指针保持原样。只在CompiledKernel创建前调整输入源对象的host参数契约；原IR模块/字节/hash/make_ir、所有backend stages/options/target不变，原生metadata、binary/module/function/launcher均不替换或热改；退出恢复原IRSource.__init__，正常/异常路径验证。真实环境必须标“official3.7backend/runtime+本地签名reader适配”，不能写成未改动的file-source ABI路径。禁止后改ck.src.signature/ck.metadata，禁止stageoverride/inspection hooks。
3. ABI独立证据：raw/adapted编译同一输入时LLIR/PTX/cubin字节相同，唯一变化是编译前源对象的逻辑launch signature。nativeCudaLauncher与其CUDA tensor-map编码原封不动；host端使用官方TensorDescriptor并让其读取该kernel真实tensordesc_meta(swizzle/物理block_size)编码，不能手工套用FGmetadata。descriptor base/shape/stride/tile/输出对象必须实际匹配，保持底层tensor生命周期。
4. 最多3个输入原型：FGhostIR拷贝仅用于ABI能力；从既有device-TMA TTIR直接替换入口/去掉设备make_descriptor得到自己的host版本；统一签名envelope匹配baseline（原教程算术与指针load不变，增加同样hostdesc与rawptr参数）和candidate（同签名descriptorload/store）。检查unused descriptors是否能由原生lowering/metadata传参；若不支持不得复制candidate metadata给baseline或加入虚假额外工作来拖慢baseline。明确标记不能闭环之处，必要时停止接受优化结论，而不是绕开原生runtime。
5. 先编译/加载/1152和1280的小型数值验证，无workspace allocator、global_scratch须0；再37项扩展数值覆盖（含tail/padding/partialgroup/压力），nativeFP16/禁TF32FP32、小CPUFP64，atol.01 rtol.001，sentinel/canary/finite/输入不变；memcheck/racecheck必须保留退出码与警告。上轮device-TMA的37warnings不能因迁移host就当作清除。

## 阶段B：仅当流程闭环后推进性能

必须先有共同的host调用契约/匹配baseline、完整target/options与原生链证据，且数值和mem/race均通过，才将host-descriptor作为正式优化方向；不自动替换旧optimized。若数值能跑但race告警或匹配baseline缺口未解，只报告“集成可行但未具备优化验收条件”，不作为通过或收益。

最多2个本组功能候选（hostTMA及FGpid映射），不变配置与shape映射。6点组（768/896/1024/1152/1280/1408）完整配对7轮rep20ms，seed123独立进程5轮；A/A用同baseline，预分配输出与prepared descriptors，图设备时间包含全部GPU工作但host编码成本单独测；还测包括descriptor准备的调用口径，不能只报kernel收益。源文件、reader/hostwrapper、输入IR/产物/选项、descriptor元数据与实际参数展开、原始配对样本/顺序/环境hash、GPUUUID保存。全已暴露，不称独立holdout。

接受标准保持两轮geo>=1.02、bootstrap95%下界>1、最差点>=.95；A/A偏差<=2%、baseline max/min<=1.10，缺项/无效/工具失败不接受。未通过则旧选择不变。若闭环且有收益，提供独立host-descriptor入口和来源明确的host版TTIR，不重写旧baseline或把ABI变更收益标成旧固定ABI“只改TTIR”。

本地新实验组明确改变host传参契约，符合实验要求§6需另建分组规则。不得改已编译产物；reader适配为显式、受限、版本固定的ABI兼容实验，有失败保留与反例检查，不是偷偷更换编译器。KernelForge只供workflow/review；原生MM执行仍用本地adapter，不宣称框架qualification。最后precommit/独立close/Git归档，报告可行/不可行与限制；不承诺必有性能收益。
