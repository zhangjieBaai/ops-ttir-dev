"""Numbered source-level changes and bounded hardware explanations."""


def item(number, title, benefit, diff, mechanism):
    lines = [f"{number}. **{title}。** {benefit}", "", "   ```diff"]
    lines.extend("   " + line for line in diff.strip().splitlines())
    lines.extend(["   ```", ""])
    lines.extend("   " + line for line in mechanism.splitlines())
    return "\n".join(lines)


BASIC = {
    "mean": [
        (
            "扁平化向量状态并使用32位地址索引",
            "累计1.1008×。",
            """- lanes = tl.arange(0, 128)[:, None] * 8 + tl.arange(0, 8)[None, :]
- row = tl.program_id(0).to(tl.int64)
+ lanes = tl.arange(0, 1024)
+ row = tl.program_id(0).to(tl.int32)""",
            "128×8与1×1024覆盖相同的逻辑输出元素，扁平化改变编译器处理布局和索引的方式；"
            "在已验证的地址范围内使用i32，减少64位整数地址中间量及相关指令。"
            "两项按组合测得收益，不能将全部提升分配给扁平化；仅改变形状也可能得到相同机器码。",
        ),
        (
            "展开16行预加载并保留原累加顺序",
            "组合增量1.2815×、累计1.4106×。",
            """- for n in range(N):
-     acc += tl.load(X + row_base + n * K + lanes, mask, other=0).to(tl.float32)
+ for n0 in range(0, N, 16):
+     # 先预加载v0…v15，再按原来的n顺序累加
+     v0 = tl.load(X + row_base + n0 * K + lanes, mask0, other=0)
+     v1 = tl.load(X + row_base + (n0 + 1) * K + lanes, mask1, other=0)
+     # 同样预加载v2…v15，尾部均保留n0+j<N的掩码
+     acc = acc + v0.to(tl.float32)
+     acc = acc + v1.to(tl.float32)
+     # 再依次累加v2…v15""",
            "提前发出多个独立加载，为调度器提供更多可重叠的访存请求，减少逐行“加载后立即等待再累加”的依赖链影响。"
            "N<8保留原路径，单列尾块采用256×1的N方向归约，避免小输入承担展开成本。"
            "展开可能提高寄存器压力；当前增益包含展开、短N分支和尾列策略，没有分别量化三者。"
            "实际修改的18个向量路径点几何加速1.7743×，其余12点实现不变。",
        ),
    ],
    "var_mean": [
        (
            "短行使用单tile并复用输入",
            "单独1.1443×。",
            """- # 用循环分别加载输入，先求均值再求中心化平方和
+ # N<=128/512/1024分别选择WIDTH=128/512/1024
+ cols = tl.arange(0, WIDTH)
+ x = tl.load(X + row_base + cols, cols < N, other=0)
+ mean = tl.sum(x, 0) / N
+ centered = tl.where(cols < N, x - mean, 0.0)
+ variance = tl.sum(centered * centered, 0) / (N - correction)""",
            "一次加载的数据同时用于均值和方差，减少短行的重复访存、循环控制和多余向量位置。"
            "中心化方差公式保留，未改成更易发生抵消误差的E[x²]-E[x]²；长行仍使用原1024宽循环。",
        ),
        (
            "向编译器提供N为16倍数的对齐信息",
            "单独1.2909×。",
            """- @triton.jit(do_not_specialize=["M", "N", "correction"])
+ @triton.jit(do_not_specialize=["M", "correction"])
  def row_kernel(X, Y, N, ...):
      # N仍是运行时参数，合法输入共享N%16==0这一对齐类别
      ...""",
            "行stride可被16整除，使编译器能够保留行首对齐信息并选择更宽的向量加载，降低访存指令开销。"
            "这里展示允许N参与JIT对齐特化的写法；实验曾验证，简单在入口写tl.multiple_of(N,16)可能被丢弃，"
            "不能只凭提示出现在源码里就认定向量化已生效。",
        ),
        (
            "组合短行复用与对齐信息",
            "在短行版本上加入对齐信息，增量1.1721×、消融累计1.3416×。",
            """- # 只使用短行单tile分支
+ # 短行单tile分支 + 保留N的16整除对齐类别
  if N <= 128:
      short_row(..., WIDTH=128)
  elif N <= 512:
      short_row(..., WIDTH=512)
  elif N <= 1024:
      short_row(..., WIDTH=1024)""",
            "两种变化分别减少短行工作量、改善加载表达，组合后才形成最终选择。"
            "单项收益存在交互，不做百分比相加；此处消融与前表的晋升验证是不同会话，未混算平均。",
        ),
        (
            "缩窄长行循环的对照",
            "单独1.1658×，但加入最终组合仅0.9665×，耗时增加3.46%，未采用。",
            """- for off in range(0, N, 1024):
-     cols = off + tl.arange(0, 1024)
+ for off in range(0, N, 256):
+     cols = off + tl.arange(0, 256)""",
            "缩小累加向量可能降低活跃状态，但同时将长行循环次数增加到原来的四倍。"
            "在已经采用短行分派与对齐信息的上下文中，新增循环和加载代价超过收益，说明寄存器更省并非充分条件。",
        ),
    ],
    "mm": [
        (
            "以host TMA描述符搬运矩阵块",
            "完整31点相对匹配基线的几何加速1.0449×。",
            """- a = tl.load(A + rows[:, None] * stride_am + ks[None, :] * stride_ak, mask_a, other=0)
- b = tl.load(B + ks[:, None] * stride_bk + cols[None, :] * stride_bn, mask_b, other=0)
+ a = a_desc.load([pid_m * BM, k0])
+ b = b_desc.load([k0, pid_n * BN])
  acc = tl.dot(a, b, acc)
- tl.store(C + output_offsets, acc.to(tl.float16), mask_c)
+ c_desc.store([pid_m * BM, pid_n * BN], acc.to(tl.float16))""",
            "描述符在host提前准备，规则二维块搬运可由TMA相关硬件路径承担，减少逐元素地址和掩码构造，"
            "为Tensor Core计算供数。是否形成更好的流水仍取决于后端lowering和同步；当前设备计时不含描述符准备成本。"
            "racecheck仍有告警，所以不能把设备时间改善解释为已完成同步资格验收。",
        ),
        (
            "简化分组PID到矩阵行块的映射",
            "完整31点相对匹配基线的几何加速1.0492×。",
            """- pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
+ pid_m = first_pid_m + pid % group_size_m""",
            "在原固定分组映射的合法范围内，改写组内行号计算，给编译器简化整数取模及依赖链的机会。"
            "收益由整个编译结果决定，未证明某一条取模指令承担全部差异；两候选各自与基线配对，"
            "不能用两个整体均值相除作为独立确认的增量。",
        ),
        (
            "按完整配置组采用或回退",
            "384³…640³采用host TMA，768³…1408³采用host TMA+PID，其余保留基线。",
            """- # 将同一种TMA候选推广到全部配置组
+ # 按已冻结的配置组选择实现
+ selected = family_selection[family]
+ selected_kernel = kernels[selected]""",
            "不同矩阵块配置对应不同的寄存器、共享内存与搬运/计算比例，一种优化不必适合所有组。"
            "本项是选择策略，不是新的设备指令优化；前表仍保留全部TMA+PID候选的负结果及计时无效标记。",
        ),
    ],
    "sum": [
        (
            "小输入由单CTA完成第一级归约",
            "保留第二个拷贝kernel时为0.9991×。",
            """- # 多个CTA各写一个局部和
- sum_kernel_1[(num_blocks,)](X, partial, N, BLOCK, ...)
+ # n<=16384：一个CTA覆盖全部输入，但仍输出到partial
+ sum_kernel_1[(1,)](X, partial, N, triton.next_power_of_2(N), ...)
  sum_kernel_2[(1,)](partial, Y, 1, ...)""",
            "减少局部结果和第二级归约的数据量，但两个短kernel的启动及调度成本依然存在，"
            "因此仅改变第一级归约方式没有得到可靠整体收益。",
        ),
        (
            "直接写出结果并移除第二个kernel",
            "累计1.3569×；小输入15点1.8402×；直接配对增量1.3575×。",
            """- sum_kernel_1[(1,)](X, partial, N, BLOCK, ...)
- sum_kernel_2[(1,)](partial, Y, 1, ...)
+ sum_kernel_1[(1,)](X, Y, N, BLOCK, ...)""",
            "第一级已经得到全量和，直接写最终输出即可省去第二次kernel执行与中间结果读写。"
            "小输入计算短，移除这次启动/调度的相对收益更突出；n>16384保留原两阶段。",
        ),
        (
            "32位索引与分组归约对照",
            "i32为0.9995×，分组归约0.9997×，两者组合0.9972×，未加速。",
            """- pid = tl.program_id(0).to(tl.int64)
+ pid = tl.program_id(0).to(tl.int32)
- total = tl.sum(values, 0)
+ partial = tl.sum(tl.reshape(values, (4, BLOCK // 4), can_reorder=True), 0)
+ total = tl.sum(partial, 0)""",
            "i32尝试降低整数地址运算开销，分组尝试改变归约树与线程间通信。"
            "这两种变换在当前配置上没有可靠整体收益，不能因为逻辑上少了宽整数或改变了树形就推断硬件执行更快。",
        ),
        (
            "加载缓存策略对照",
            ".cg为0.9429×；TTIR中的加载.cs对照为1.0001×，均未采用。",
            """- values = tl.load(ptrs, mask, other=0)
+ values = tl.load(ptrs, mask, other=0, cache_modifier=".cg")""",
            ".cg让加载优先使用L2及以下缓存，可能减少L1干扰，也可能损失有效复用，本例出现回退。"
            "公开Triton的tl.load不接受.cs，不能伪造对应源码调用；该TTIR属性对照与默认加载的机器码一致，未形成有效优化。",
        ),
        (
            "完整CTA使用无mask加载",
            "相对原始基线0.9945×。",
            """- values = tl.load(X + offsets, offsets < N, other=0)
+ if (pid + 1) * BLOCK <= N:
+     values = tl.load(X + offsets)
+ else:
+     values = tl.load(X + offsets, offsets < N, other=0)""",
            "只有确认整块都在输入范围内时才去掉mask，尾块仍检查边界。"
            "意图是减少谓词和索引判断，但新分支及编译布局也有成本，当前矩阵未观察到整体改善。",
        ),
        (
            "在最终版上组合大输入无mask分支",
            "相对最终版增量0.9998×，未采用。",
            """- # 大输入沿用原来的带mask两阶段路径
+ # 大输入完整块走无mask加载，尾块和小输入最终路径保留
+ if full_block:
+     values = tl.load(X + offsets)""",
            "该对照检验无mask加载能否在已有单kernel小输入优化之外继续受益。"
            "结果接近1×，说明它没有补足当前最终版的主要性能瓶颈，不把这项加入正式组合。",
        ),
    ],
    "logsumexp": [
        (
            "缩减无效填充并合并在线指数",
            "属于最终组合，未单独分摊收益。",
            """- new_m = tl.maximum(m, x)
- z = z * tl.exp(m - new_m) + tl.exp(x - new_m)
+ new_m = tl.maximum(m, x)
+ q = tl.exp(-tl.abs(x - m))
+ z = tl.where(x > m, z * q + 1.0, z + q)""",
            "有限输入上两个指数中必有一个对应exp(0)，利用最大值的选择关系可将每次更新的两个指数改为一个，"
            "减少特殊函数计算；全-inf位置仍需保留单独保护。非内层按N范围缩小tile，减少填充位置的算术和归约工作。",
        ),
        (
            "在FP16上求最大值并为长行选择单块归约",
            "属于最终组合，未单独分摊收益。",
            """- x = tl.load(ptrs, mask, other=-float("inf")).to(tl.float32)
- maximum = tl.max(x, 0)
+ x16 = tl.load(ptrs, mask, other=-float("inf"))
+ maximum = tl.reduce(x16, 0, maximum_half).to(tl.float32)
+ x = x16.to(tl.float32)""",
            "输入本来就是FP16，maximum_half用tl.maximum选择较大值；在扩展为FP32前完成max，改变归约的数据宽度和中间状态。"
            "指数和求和仍用FP32。已验收长行由单块完成，减少循环与跨块状态更新，但超长输入保留原分块路径，避免无界寄存器扩张。",
        ),
        (
            "小输入合并多行并缩窄流式状态",
            "属于最终组合，未单独分摊收益。",
            """- row = tl.program_id(0)
- cols = tl.arange(0, WIDTH)
+ pid = tl.program_id(0)
+ if pid % 8 == 0:
+     rows = pid + tl.arange(0, 8)
+     cols = tl.arange(0, WIDTH)
+     # 二维加载8行，并沿列轴分别归约；保留行/列边界mask
+     maximum = tl.max(x, axis=1)""",
            "同一CTA处理多行可以改善短行的有效线程工作量；原grid中非活动CTA退出，避免重复写输出。"
            "单行小输入采用32-lane分片累加，在合法范围内将地址中间量缩为i32，尝试降低宽状态和地址运算成本。",
        ),
        (
            "组合上述改动",
            "完整30点累计1.1221×。",
            """- # 所有N使用原启发式对应的计算路径
+ # 运行时范围分派：小输入批处理/32-lane流，长行单块，超长保留原分块
+ if N <= one_tile_limit:
+     one_tile_path(...)
+ else:
+     original_online_path(...)""",
            "收益来自计算量、逻辑tile、状态更新与地址表达的组合，不是某一项寄存器或缓存指标的独立贡献。"
            "仍保留持平与回退shape；不同子集下的方法不能逐项相加为整体加速。",
        ),
    ],
    "vector_add": [
        (
            "输出写入采用.cs缓存策略",
            "完整矩阵几何平均1.0094×，N=2²²局部约1.1539×。",
            """- tl.store(Y + offsets, x + y, mask=offsets < N)
+ tl.store(Y + offsets, x + y, mask=offsets < N, cache_modifier=".cs")""",
            "流式输出提示让写出数据采用更易淘汰的缓存策略，可能减轻输出对输入或其他数据的缓存挤占。"
            "它不减少加法数量或有效输出字节，局部收益也不代表所有N都加速。"
            "完整矩阵未达原1.02×门槛，最终采用来自用户接受局部收益；不把其他设备上的约1.206×混入本会话。",
        ),
    ],
    "layer_norm/backward_dx": [
        (
            "小宽度组折叠冗余mask与中间状态",
            "1024/2048组采用，未独立分摊对全矩阵的贡献。",
            """- w = tl.load(W + cols, mask=mask)
- centered = tl.where(mask, x - mean, 0.0)
- wdy = tl.where(mask, w * dy, 0.0)
+ w = tl.load(W + cols, mask=mask, other=0.0)
+ centered = x - mean
+ wdy = w * dy
  # DY的无效位置也加载为0，最终输出store仍保留mask""",
            "让无效位置的W/DY在加载时为0，使其乘积自然为0，可以省去部分重复select及相关中间向量。"
            "这改变谓词计算与值存活，但不删除实际访存边界检查；在已验证的输入和表达式上合法。",
        ),
        (
            "4096组按复用特点设置缓存淘汰策略",
            "该配置组采用，未独立分摊对全矩阵的贡献。",
            """- x = tl.load(X + cols, mask=mask, other=0)
- dy = tl.load(DY + cols, mask=mask, other=0)
- w = tl.load(W + cols, mask=mask)
+ x = tl.load(X + cols, mask=mask, other=0, eviction_policy="evict_first")
+ dy = tl.load(DY + cols, mask=mask, other=0, eviction_policy="evict_first")
+ w = tl.load(W + cols, mask=mask, eviction_policy="evict_last")""",
            "X/DY按行流过，W被多行复用；分别提示优先淘汰流式输入、尽量保留权重，尝试改善缓存竞争。"
            "这里只说明策略和该配置组的验收结果，没有独立证明命中率变化解释了全部收益。",
        ),
        (
            "8192组保留基线",
            "该组不增加新优化。",
            """- # 强制采用其他宽度组的改动
+ # 8192组沿用原始baseline实现""",
            "不同宽度的归约、寄存器压力和访存比例不同。按完整配置组回退避免将其他组的局部收益外推到此组。",
        ),
        (
            "16384组合并统计归约并提取标量系数",
            "该配置组采用，完整矩阵仍未达到10%目标。",
            """- c1 = tl.sum(xhat * wdy, 0) / N
- c2 = tl.sum(wdy, 0) / N
- dx = (wdy - xhat * c1 - c2) * rstd
+ s1, s2 = tl.reduce((xhat * wdy, wdy), 0, sum_pair)
+ alpha = (s1 / N) * rstd
+ beta = (s2 / N) * rstd
+ dx = wdy * rstd - beta - xhat * alpha""",
            "sum_pair按分量相加，将两项统计放在联合归约中；把全行相同的乘法系数提前算成标量再广播。"
            "意图是共享归约遍历、减少向量端重复算术和长存活状态，浮点运算顺序变化已按原容差验证。"
            "各配置组的贡献未分别对全矩阵消融，最终平均仍使用前表实际选择路径的配对数据。",
        ),
    ],
}


def basic(operator):
    return "\n\n" + "\n\n".join(
        item(i, *entry) for i, entry in enumerate(BASIC[operator], 1)
    )


COMPLEX = {
    "int4_gemm": [
        (
            """- group_low = (k0 + 2 * kp) // qGroupSize
- group_high = (k0 + 2 * kp + 1) // qGroupSize
- scale_high = tl.load(S + group_high[:, None] * stride_sg + ns[None, :], mask_n)
+ group = (k0.to(tl.uint32) // qGroupSize.to(tl.uint32)).to(tl.int32)
+ scale_high = scale_low
+ zero_high = zero_low""",
            "一个byte中的两个nibble以及当前对齐K tile共享同一量化组，复用scale/zero并标量化组索引，"
            "减少重复地址和组参数计算；unsigned索引为非负范围的除余简化提供条件。"
            "这三项是首轮组合，没有将收益独立分配给某一个指令变化；FP32减乘与FP16舍入顺序保留。",
        ),
        (
            """- groups = tl.full((BK // 2, 1), group, tl.int32)
- scale = tl.load(S + groups * stride_sg + ns[None, :] * stride_sn, mask_n)
+ scale_row = tl.load(S + group * stride_sg + ns * stride_sn, ns < N)
+ scale = tl.broadcast_to(scale_row[None, :], (BK // 2, BN))""",
            "将重复的K×N逻辑参数加载改成一行N元素加载后广播，zero同样处理。"
            "既减少冗余参数索引和加载表达，也缩小需要维护的地址/参数向量，可能改善后端加载与寄存器分配。"
            "它不意味着硬件DRAM读取字节必然按逻辑张量大小同比减少，缓存和广播原本也可能消除部分重复。",
        ),
        (
            """- a = tl.load(a_ptrs, (ms[:, None] < M) & (ks[None, :] < K), other=0)
- packed = tl.load(w_ptrs, (ns[None, :] < N) & (byte_idx[:, None] < K // 2), other=0)
+ a = tl.load(a_ptrs, ms[:, None] < M, other=0)
+ packed = tl.load(w_ptrs, ns[None, :] < N, other=0)""",
            "K为128倍数且BK为64或128，当前K迭代不会越界，可以去掉恒真的K侧谓词。"
            "M/N尾部mask仍保留。该改动尝试减少比较和谓词组合，单独启用几乎无整体收益，"
            "在窄加载上下文中才取得表述的条件增益，说明它与加载lowering存在交互。",
        ),
        (
            """- for k0 in range(0, K, 64):
-     # 每次重新加载当前group的scale/zero并执行一次dot
+ for k0 in range(0, K, 128):
+     # 每个128-K组只加载一次scale/zero
+     for half in tl.static_range(2):
+         k_start = k0 + half * 64
+         # 加载A/packed，按共享参数反量化，依原顺序执行dot""",
            "两个64-K迭代属于同一量化组，将组参数保留到第二次dot使用，减少组参数重复加载与索引计算。"
            "两次dot的先后和FP16反量化舍入不变；BK128路径不需要重复合并，保持原状。"
            "复用会延长参数存活，因此收益需由完整融合kernel验证，不能仅按参数加载次数推算。",
        ),
        (
            """- group = k0.to(tl.uint32) // qGroupSize.to(tl.uint32)
+ group = k0.to(tl.uint32) >> 7""",
            "量化组大小固定128、k0非负，除128可用逻辑右移7位表达，降低运行时整数除法及其依赖开销。"
            "该特化只在既定group=128合同内合法，不扩展到任意分组大小。",
        ),
    ],
    "moe_routing": [
        (
            """- belongs = expert_group[:, None] == group_ids[None, :]
- expert_selected = tl.sum((belongs & group_selected[None, :]).to(tl.int32), 1) > 0
+ expert_selected = tl.reshape(
+     tl.broadcast_to(group_selected[:, None], (8, EXPERTS // 8)), (EXPERTS,)
+ )""",
            "专家按8个连续等长组排列，组选择结果可以直接广播到组内专家。"
            "省去专家×组关系矩阵及其归约，减少中间状态、比较指令和组轴通信；分组布局不连续时不能直接套用。",
        ),
        (
            """- best_score = tl.max(selection_scores, 0)
+ normalized = tl.where(selection_scores == 0.0, 0.0, selection_scores)
+ bits = normalized.to(tl.int32, bitcast=True)
+ keys = (bits ^ ((bits >> 31) | -2147483648)).to(tl.uint32)
+ best_key = tl.max(keys, 0)""",
            "将有限FP32分数编码为保持数值顺序的uint32，循环中用整数max维护选择状态，"
            "为后端使用整数归约指令提供机会；实际映射仍取决于布局和编译。"
            "±0先归一化，随后保留最小原始索引的tie规则，不因更换状态表示改变选中专家。",
        ),
        (
            """- for i in range(TOPK):
-     selected_raw = tl.load(scores_ptr + row_base + selected_idx)
-     selected_weight = 0.5 * libdevice.tanh(0.5 * selected_raw) + 0.5
-     # 循环内更新权重向量
+ # 循环只收集topk_indices，结束后从已计算的scored向量一次gather
+ topk_values = tl.gather(scored, topk_indices, axis=0)""",
            "原分数向量已经计算过sigmoid，选中后无需每轮标量重载再算一次tanh。"
            "把权重收集移出选择循环，减少重复访存、特殊函数计算和循环携带的权重状态；"
            "最终gather也可能引入布局开销，所以仍以完整两阶段结果判断。",
        ),
        (
            """- # 逐次max、计数和prefix处理边界tie，再确定4个组
+ ids = tl.arange(0, 8)
+ greater = group_scores[None, :] > group_scores[:, None]
+ earlier_tie = (group_scores[None, :] == group_scores[:, None]) & (ids[None, :] < ids[:, None])
+ rank = tl.sum((greater | earlier_tie).to(tl.int32), axis=1)
+ group_selected = rank < 4""",
            "只对8个组构造8×8的稳定排名，将多轮选择控制改为小规模并行比较/归约，减少顺序依赖。"
            "同分用小组索引优先，保持原语义。该方案用更多小矩阵比较换较短的控制链，不代表比较次数最少。",
        ),
        (
            """- max1 = tl.max(group_scores, 0)
- # 统计/处理唯一最大值，再归约max2
+ # top2_pair合并(h1,l1)、(h2,l2)：
+ # hi=max(h1,h2)，lo=max(min(h1,h2),max(l1,l2))
+ max1, max2 = tl.reduce((group_scores, tl.full_like(group_scores, -float("inf"))),
+                       axis=0, combine_fn=top2_pair)""",
            "一个归约树同时保留最高两个值，避免最高值、计数与次高值之间的多阶段依赖。"
            "但tuple状态和每次合并的算术更多，未必降低总硬件工作量；最终版移除该项仅约0.5%差异，"
            "小于加入时约1.9%，不把它解释成可独立相加的固定贡献。",
        ),
        (
            """- group_count = n_group
- group_width = num_experts_per_group
- token_stride = stride_scores_token
+ group_count = 8
+ group_width = EXPERTS // 8
+ token_stride = EXPERTS  # EXPERTS按256/512配置族固定""",
            "把已有合同中的组数、专家数和连续stride表达为编译期已知值，帮助常量折叠地址计算、边界判断和除余。"
            "源码可用constexpr参数表达，TTIR实验仍保留原ABI；这里只利用固定前提，没有宣称支持新的运行时布局。",
        ),
        (
            """- group = pid % 8
- # 一个CTA处理一个组
+ if pid % 4 == 0:
+     token = pid // 8
+     groups = pid % 8 + tl.arange(0, 4)
+     # 加载[4, EXPERTS//8]，沿axis=1分别归约4个组""",
            "每组只有32或64个专家，将4组放进同一CTA能给4warps提供更多有效工作，减少重复的CTA内公共开销。"
            "保留原grid和token边界检查，非活动CTA退出，第二阶段字节不变；不能说grid启动数本身减少为四分之一。",
        ),
    ],
    "norm_quant": [
        (
            """- amax = tl.max(tl.abs(u * rho), 0)
+ max_u = tl.max(tl.abs(u), 0)
+ amax = max_u * rho""",
            "rho为正，amax可因式分解，将向量端乘rho后的max改成先对u归约、最后做一次标量乘法。"
            "max不必等待RMS统计和rsqrt结果，解除两项统计的依赖，为后续合并归约创造条件；量化容差保持原合同。",
        ),
        (
            """- ss = tl.sum(z * z, 0)
- max_u = tl.max(tl.abs(u), 0)
+ # sum_max按分量合并：sum相加、max取较大值
+ ss, max_u = tl.reduce((z * z, tl.abs(u)), axis=0, combine_fn=sum_max)""",
            "将两个独立统计放在同一个归约表达式中，有机会共享数据遍历和跨线程归约的调度。"
            "tuple也增加携带状态，不能只凭一次tl.reduce就断言barrier或寄存器一定减少；性能以当前组合实测为准。",
        ),
        (
            """- tl.store(Z + offsets, z, mask)
- tl.store(Q + offsets, q, mask)
- tl.store(S + row, scale)
+ tl.store(Z + offsets, z, mask, cache_modifier=".cs")
+ tl.store(Q + offsets, q, mask, cache_modifier=".cs")
+ tl.store(S + row, scale, cache_modifier=".cs")""",
            "给三类输出流式写入提示，尝试降低只写不再读的数据对缓存的挤占。"
            "完整矩阵增量约0.28%，不足以单独宣称稳定优化，也没有隔离证明缓存命中率解释了全部变化。",
        ),
        (
            """- q_float = tl.div_rn(y, scale)
+ inv_scale = tl.div_rn(1.0, scale)
+ q_float = y * inv_scale""",
            "每行共用一个scale，把逐元素精确除法改成一次标量倒数与向量乘法，减少重复除法展开及其依赖。"
            "浮点舍入顺序变化已按既定合同检查：Q允许相对FP64最多差1 code，并同时限制scale和重构误差；"
            "不能将其称为INT8逐位等价。",
        ),
        (
            """  z = x.to(tl.float32) + r.to(tl.float32)
+ tl.store(Z + offsets, z, mask, cache_modifier=".cs")
  # 随后完成统计归约、scale与Q计算
- tl.store(Z + offsets, z, mask, cache_modifier=".cs")""",
            "Z只依赖残差加法，不必等量化完成再写。提前store有机会缩短z的存活范围，并将写出与后续统计计算穿插，"
            "改善寄存器压力与调度；是否具体来自哪项硬件变化尚未单独量化。输入输出无别名保证该提前写出合法。",
        ),
    ],
    "attention_dkdv": [
        (
            """- ppT = pT.to(tl.float16)
- dv += tl.dot(ppT, do)
  dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
  dsT = pT * (dpT - Di[None, :])
  dk += tl.dot(dsT.to(tl.float16), tl.trans(qT))
+ ppT = pT.to(tl.float16)
+ dv += tl.dot(ppT, do)""",
            "延后概率的FP16副本及dV消费位置，调整dK/dV两条dot路径同时活跃的中间值和调度机会。"
            "两个梯度accumulator各自的累加顺序不变，收益来自整个编译结果；这里只对D64的16点成立，不能外推D128。",
        ),
        (
            """- # 一个CTA持有[128,D]的dK/dV状态
- key_rows = start_key + tl.arange(0, 128)
+ for sub in range(2):
+     key_rows = start_key + sub * 64 + tl.arange(0, 64)
+     # 每次只计算、写出一个[64,D]子块""",
            "将同一CTA的输出状态拆成两个64-key子块，缩小dK/dV累加器与中间概率块，缓解宽矩阵带来的寄存器/spill压力。"
            "原CTA数和输出归属不变，但Q/dO可能读取更多次，不能描述为免费减半计算量。",
        ),
        (
            """- q_ptrs += QUERY_BLOCK * stride_q
- do_ptrs += QUERY_BLOCK * stride_do
+ q_ptrs = Q + base_q + query_rows[None, :] * stride_q
+ do_ptrs = DO + base_do + query_rows[:, None] * stride_do""",
            "非因果组按当前query坐标重算指针，减少循环携带的地址状态，为后端布局和寄存器复用提供不同表达。"
            "重算本身增加部分整数运算，收益不能简单解释成地址指令更少，须由编译后的资源和完整kernel耗时确认。",
        ),
        (
            """- ppT = pT.to(tl.float16)
- dv += tl.dot(ppT, do)
  dk += tl.dot(dsT.to(tl.float16), tl.trans(qT))
+ ppT = pT.to(tl.float16)
+ dv += tl.dot(ppT, do)""",
            "在64-key及指针重算已经存在的上下文中再后移dV，进一步调整两个dot路径的活跃中间值。"
            "这是D128非因果组的条件效应，不能与D64的收益相加或作为所有尺寸的统一收益。",
        ),
        (
            """- key_rows = start_key + tl.arange(0, 128)
+ for sub in range(2):
+     key_rows = start_key + sub * 64 + tl.arange(0, 64)
+     # 保留因果mask，依次计算两个64-key子块""",
            "因果组也缩窄输出累加状态，但必须保留query遍历和因果mask边界。"
            "减少同时活跃状态可能降低spill代价，同时仍付出重复读取Q/dO的成本，不能直接套用非因果组的加速比。",
        ),
        (
            """- for sub in range(2):
+ for sub in tl.static_range(2):
      # 按sub计算子块偏移及因果遍历范围""",
            "固定两个子块的偏移和边界在展开后对编译器可见，有机会简化控制与地址依赖并重新调度加载/dot。"
            "代码体积和活跃值也可能增加，实际效果是当前因果组的受控对照结果，不是静态展开总会加速。",
        ),
        (
            """- ppT = pT.to(tl.float16)
- dv += tl.dot(ppT, do)
  dk += tl.dot(dsT.to(tl.float16), tl.trans(qT))
+ ppT = pT.to(tl.float16)
+ dv += tl.dot(ppT, do)""",
            "在因果组的64-key与静态展开结构上再次调整dV消费顺序，保持各accumulator的数学累加顺序。"
            "64-key、展开和dV后移有交互，累计数字只针对标注的8点；全32点平均仍由前表单独计算。",
        ),
    ],
    "partial_topk": [
        (
            """- best = tl.max(values, 0)
+ bits = values.to(tl.int32, bitcast=True)
+ keys = (bits ^ ((bits >> 31) | -2147483648)).to(tl.uint32)
+ best_key = tl.max(keys, 0)
  # 该初版在每次K选择时都重新编码""",
            "有序整数key允许用uint32 max表达分数选择，但每轮重新编码增加位运算和状态转换，"
            "抵消了可能的归约收益。本项回退照实保留，不把“整数归约可用”当作已经加速的证明。",
        ),
        (
            """- for step in range(K):
-     keys = ordered_key(values)
+ values = tl.where(values == 0.0, 0.0, values)
+ keys = ordered_key(values)
+ for step in range(K):
      best_key = tl.max(keys, 0)
+     keys = tl.where(ids == best_id, 8388607, keys)""",
            "ordered_key按上一项的位运算定义，将编码移出循环并直接携带key状态，省去K轮重复的FP32到整数顺序表示转换。"
            "±0先归一化，已选元素置为-inf对应的有序key；最小原始index tie规则保持不变。",
        ),
        (
            """  keys = tl.where(ids == best_id, 8388607, keys)
- ids = tl.where(ids == best_id, 2147483647, ids)
+ # ids保持原值；移除由keys中的哨兵状态表示""",
            "索引本身不需要随每轮改写，固定ids可以减少循环携带向量及每轮select。"
            "短尾块耗尽后的-inf与INT_MAX保护仍保留，避免padding候选干扰全局稳定排序；不能只删更新而忽略哨兵语义。",
        ),
        (
            """- for step in range(K):
+ for step in tl.static_range(K):
      # 固定K=8/16，按原顺序执行选择和状态更新""",
            "展开固定K使迭代常量和状态传递对编译器可见，减少循环控制并提供跨迭代调度空间。"
            "选择仍有前后依赖，不是把K次选择并行化；代码和活跃状态也可能增大。",
        ),
        (
            """- values = tl.load(X + offsets, offsets < N, other=-float("inf"))
+ # 仅在合同证明本stage所有输入lane都有效时
+ values = tl.load(X + offsets)
  # N=65537等尾块路径继续使用mask和耗尽保护""",
            "利用完整tile、固定grid和至少K个有限候选的前提，删除恒真mask及不可能触发的部分耗尽判断，"
            "减少谓词/选择状态。尾块的保护继续保留，不将某个整齐shape的合法删除推广到全部N。",
        ),
        (
            """- # merge每轮在CHUNKS*K个候选上做选择
+ cursor = tl.full((CHUNKS_PAD,), 0, tl.int32)
+ head = tl.load(V + chunk_offsets * K, valid_chunk, other=-float("inf"))
+ # 每轮只比较各列表head，选中后推进该列表cursor并加载下一项""",
            "local输出每个chunk的有序K列表，因此未消费元素都不大于当前头部；只比较各列表head即可选择全局下一项。"
            "merge逻辑宽度由CHUNKS×K缩到约CHUNKS，但会新增cursor依赖和后续小加载。"
            "收益由完整local+merge两kernel计时确认，不适用于任意未排序workspace。",
        ),
        (
            """- best_key = tl.max(head_key, 0)
- best_id = tl.min(tl.where(head_key == best_key, head_id, 2147483647), 0)
+ packed = (head_key.to(tl.uint64) << 32) | (2147483647 - head_id).to(tl.uint64)
+ winner = tl.max(packed, 0)
+ best_id = 2147483647 - (winner & 0xFFFFFFFF).to(tl.int32)""",
            "高32位编码值顺序，低32位反向编码原始index，使一次uint64 max同时表达最大值和最小index优先。"
            "它以更宽状态和64位归约代价换掉第二次min归约，不能仅凭归约次数就断言更快。"
            "当前增量约0.48%，不单独宣称已隔离出稳定硬件收益；耗尽和无效头部仍保留保护。",
        ),
    ],
}


def complex_item(operator, index, title, benefit):
    diff, mechanism = COMPLEX[operator][index - 1]
    return item(index, title, benefit, diff, mechanism)
