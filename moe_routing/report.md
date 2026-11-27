# Grouped／biased MoE routing TTIR 优化

来源：FlagGems 面向 NVIDIA Hopper 的 grouped/biased top-k routing 实现。基线保留两个设备 kernel：第一阶段计算 8 个连续专家组的带 bias 最高两分之和，第二阶段选择 4 个组、8 个专家，并使用未加 bias 的 sigmoid 分数生成归一化权重。 上游源码：[Hopper grouped top-k](https://github.com/flagos-ai/FlagGems/blob/ef8aea78fed8321c3e02a15bf31e847cd0579e4f/src/flag_gems/runtime/backend/_nvidia/hopper/grouped_topk.py)（链接固定到实验使用版本）。

环境：官方Triton3.7.0，H100，FP32，E=256/512、各15个tokens点；固定4 warps / 3 stages / 1 CTA。保留原两次launch、grid和workspace，同分优先小索引。预分配CUDA graph设备计时，不含分配、编译或CPU launch，配对计时与A/A检查通过。

**最终平均加速比：算术平均1.547798×，几何平均1.542132×。** 覆盖全部30个shape，A级性能证据。

时间为同一会话中各轮耗时的中位数，单位µs；加速比为同轮基线/优化时间比的中位数，不必恰好等于两列时间中位数之比。算术和几何平均均对各shape加速比等权计算。

| Shape (tokens, experts) | 基线时间（µs） | 优化时间（µs） | 加速比 |
|---|---:|---:|---:|
| (16, 256) | 7.2097 | 4.9840 | 1.446831× |
| (16, 512) | 8.1360 | 5.4030 | 1.506052× |
| (32, 256) | 7.3303 | 5.0293 | 1.457458× |
| (32, 512) | 8.1488 | 5.4378 | 1.496652× |
| (64, 256) | 7.4358 | 5.1342 | 1.446605× |
| (64, 512) | 8.4042 | 5.5726 | 1.508341× |
| (128, 256) | 7.5852 | 5.3510 | 1.417901× |
| (128, 512) | 8.6964 | 5.9152 | 1.470293× |
| (256, 256) | 8.4789 | 5.8707 | 1.444325× |
| (256, 512) | 9.9418 | 6.6642 | 1.492190× |
| (384, 256) | 9.5275 | 6.4014 | 1.485543× |
| (384, 512) | 11.4328 | 6.8015 | 1.681049× |
| (512, 256) | 10.6345 | 7.1443 | 1.488445× |
| (512, 512) | 12.9920 | 7.6048 | 1.707727× |
| (768, 256) | 13.0143 | 8.7664 | 1.484500× |
| (768, 512) | 16.3654 | 9.4330 | 1.734621× |
| (1024, 256) | 15.5713 | 10.5566 | 1.475091× |
| (1024, 512) | 19.9710 | 11.4572 | 1.743117× |
| (1536, 256) | 21.0500 | 14.5252 | 1.449509× |
| (1536, 512) | 27.3418 | 15.7308 | 1.739504× |
| (2048, 256) | 26.5819 | 18.7143 | 1.419401× |
| (2048, 512) | 34.5225 | 19.8597 | 1.738951× |
| (3072, 256) | 37.7724 | 26.5627 | 1.421859× |
| (3072, 512) | 49.1983 | 28.1969 | 1.744832× |
| (4096, 256) | 48.6883 | 34.5900 | 1.407663× |
| (4096, 512) | 63.7468 | 36.4580 | 1.748499× |
| (6144, 256) | 70.1035 | 50.2065 | 1.396309× |
| (6144, 512) | 91.8943 | 52.7581 | 1.741805× |
| (8192, 256) | 91.9724 | 66.0312 | 1.392752× |
| (8192, 512) | 121.0521 | 69.3266 | 1.746112× |

优化方法与性能提升：

1. **membership比较/归约改为连续组broadcast/reshape。** 增量1.185503×、累计1.185503×。

   ```diff
   - belongs = expert_group[:, None] == group_ids[None, :]
   - expert_selected = tl.sum((belongs & group_selected[None, :]).to(tl.int32), 1) > 0
   + expert_selected = tl.reshape(
   +     tl.broadcast_to(group_selected[:, None], (8, EXPERTS // 8)), (EXPERTS,)
   + )
   ```

   专家按8个连续等长组排列，组选择结果可以直接广播到组内专家。省去专家×组关系矩阵及其归约，减少中间状态、比较指令和组轴通信；分组布局不连续时不能直接套用。

2. **再把专家分数改为有序uint32 key。** 增量1.059861×、累计1.256587×。

   ```diff
   - best_score = tl.max(selection_scores, 0)
   + normalized = tl.where(selection_scores == 0.0, 0.0, selection_scores)
   + bits = normalized.to(tl.int32, bitcast=True)
   + keys = (bits ^ ((bits >> 31) | -2147483648)).to(tl.uint32)
   + best_key = tl.max(keys, 0)
   ```

   将有限FP32分数编码为保持数值顺序的uint32，循环中用整数max维护选择状态，为后端使用整数归约指令提供机会；实际映射仍取决于布局和编译。±0先归一化，随后保留最小原始索引的tie规则，不因更换状态表示改变选中专家。

3. **再把原始权重gather移到选择循环结束后。** 增量1.080044×、累计1.357304×。

   ```diff
   - for i in range(TOPK):
   -     selected_raw = tl.load(scores_ptr + row_base + selected_idx)
   -     selected_weight = 0.5 * libdevice.tanh(0.5 * selected_raw) + 0.5
   -     # 循环内更新权重向量
   + # 循环只收集topk_indices，结束后从已计算的scored向量一次gather
   + topk_values = tl.gather(scored, topk_indices, axis=0)
   ```

   原分数向量已经计算过sigmoid，选中后无需每轮标量重载再算一次tanh。把权重收集移出选择循环，减少重复访存、特殊函数计算和循环携带的权重状态；最终gather也可能引入布局开销，所以仍以完整两阶段结果判断。

4. **再以8×8稳定排名选择专家组。** 增量1.035680×、累计1.405647×。

   ```diff
   - # 逐次max、计数和prefix处理边界tie，再确定4个组
   + ids = tl.arange(0, 8)
   + greater = group_scores[None, :] > group_scores[:, None]
   + earlier_tie = (group_scores[None, :] == group_scores[:, None]) & (ids[None, :] < ids[:, None])
   + rank = tl.sum((greater | earlier_tie).to(tl.int32), axis=1)
   + group_selected = rank < 4
   ```

   只对8个组构造8×8的稳定排名，将多轮选择控制改为小规模并行比较/归约，减少顺序依赖。同分用小组索引优先，保持原语义。该方案用更多小矩阵比较换较短的控制链，不代表比较次数最少。

5. **再用双值归约计算组内top2。** 增量1.019029×、累计1.432648×。

   ```diff
   - max1 = tl.max(group_scores, 0)
   - # 统计/处理唯一最大值，再归约max2
   + # top2_pair合并(h1,l1)、(h2,l2)：
   + # hi=max(h1,h2)，lo=max(min(h1,h2),max(l1,l2))
   + max1, max2 = tl.reduce((group_scores, tl.full_like(group_scores, -float("inf"))),
   +                       axis=0, combine_fn=top2_pair)
   ```

   一个归约树同时保留最高两个值，避免最高值、计数与次高值之间的多阶段依赖。但tuple状态和每次合并的算术更多，未必降低总硬件工作量；最终版移除该项仅约0.5%差异，小于加入时约1.9%，不把它解释成可独立相加的固定贡献。

6. **再特化两阶段固定整数和stride。** 增量1.051683×、累计1.506388×。

   ```diff
   - group_count = n_group
   - group_width = num_experts_per_group
   - token_stride = stride_scores_token
   + group_count = 8
   + group_width = EXPERTS // 8
   + token_stride = EXPERTS  # EXPERTS按256/512配置族固定
   ```

   把已有合同中的组数、专家数和连续stride表达为编译期已知值，帮助常量折叠地址计算、边界判断和除余。源码可用constexpr参数表达，TTIR实验仍保留原ABI；这里只利用固定前提，没有宣称支持新的运行时布局。

7. **再让每个活动CTA处理4组（最终版）。** 增量1.023639×、累计1.542132×。

   ```diff
   - group = pid % 8
   - # 一个CTA处理一个组
   + if pid % 4 == 0:
   +     token = pid // 8
   +     groups = pid % 8 + tl.arange(0, 4)
   +     # 加载[4, EXPERTS//8]，沿axis=1分别归约4个组
   ```

   每组只有32或64个专家，将4组放进同一CTA能给4warps提供更多有效工作，减少重复的CTA内公共开销。保留原grid和token边界检查，非活动CTA退出，第二阶段字节不变；不能说grid启动数本身减少为四分之一。


条件收益不能相加。增量相对前一版本，累计相对原始基线，各项存在依赖和交互。

最终版移除双值归约的条件差异约0.5%，不同于加入时约1.9%，不能将二者当成可相加的独立贡献。FP32 group scores/weights与baseline位型相同、indices精确，另对CPU参考验证权重。

全部时间、加速比和方法收益使用同一组原始配对样本；冻结最终选择不变。新增中间消融未全部重跑sanitizer，最终版沿用已有通过记录。
