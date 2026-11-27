"""Explain frozen LayerNorm ablations using source diffs and audited counters."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OP = ROOT / "layer_norm"


def read(path):
    return json.loads(path.read_text())


def render():
    summary = read(OP / "ncu_steps/results/summary.json")
    resources = read(OP / "results/ablation/resources.json")
    assert (
        summary["evidence_grade"] == "C" and summary["verified_profile_launches"] == 28
    )
    assert summary["clock_control"] == "base" and summary["cache_control"] == "all"
    for name, expected in summary["sources"].items():
        assert hashlib.sha256((OP / name).read_bytes()).hexdigest() == expected
    metrics = {(r["N"], r["variant"]): r["metrics"] for r in summary["records"]}

    def value(n, variant, metric):
        return metrics[(n, variant)][metric]["value"]

    occupancy = "sm__warps_active.avg.pct_of_peak_sustained_active"
    dram = "dram__bytes.sum.per_second"
    latency = "gpu__time_duration.sum"
    wait = "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"
    assert [
        resources[n]["registers"]
        for n in [
            "baseline",
            "abl_arithmetic",
            "abl_grouped",
            "abl_unroll",
            "abl_ca",
            "abl_cs",
            "abl_final",
        ]
    ] == [60, 60, 32, 80, 48, 48, 48]
    assert all(resources[n]["spills"] == 0 for n in summary["variants"])
    for n in (4096, 15872):
        for variant in summary["variants"]:
            assert metrics[(n, variant)][dram]["unit"] == "Tbyte/s"
            assert metrics[(n, variant)][latency]["unit"] == "us"
            assert metrics[(n, variant)][occupancy]["unit"] == "%"
            assert (
                value(n, variant, "launch__registers_per_thread")
                == resources[variant]["registers"]
            )
    base_ops = resources["baseline"]["sass_op_counts"]
    math_ops = resources["abl_arithmetic"]["sass_op_counts"]
    assert (base_ops["MUFU.RCP"], math_ops["MUFU.RCP"]) == (3, 1)
    assert (base_ops["MUFU.SQRT"], math_ops.get("MUFU.SQRT", 0)) == (1, 0)
    assert (base_ops.get("MUFU.RSQ", 0), math_ops["MUFU.RSQ"]) == (0, 1)
    return f"""

1. **复用1/N，并使用rsqrt。** 单独复用1/N为1.0017×，单独改rsqrt为1.0047×，两项合并1.0065×。

   ```diff
   - mean = tl.sum(_mean, axis=0) / N
   + inv_n = 1.0 / N
   + mean = tl.sum(_mean, axis=0) * inv_n
   - var = tl.sum(_var, axis=0) / N
   - rstd = 1.0 / tl.sqrt(var + eps)
   + var = tl.sum(_var, axis=0) * inv_n
   + rstd = tl.rsqrt(var + eps)
   ```

   均值和方差共用一个倒数，平方根后再取倒数改为直接求平方根倒数。编译产物的静态SASS中，MUFU.RCP从3处减到1处，MUFU.SQRT从1处变为0处，并出现1处MUFU.RSQ，支持特殊函数与算术依赖路径缩短的解释。
   寄存器仍为60个/线程、无spill，因此这一步没有提高寄存器约束下的驻留上限；少量标量算术改写的整体收益也较小。

2. **使用2048窄分组与三遍动态循环。** 在算术改写基础上，增量1.1962×、累计1.2041×。

   ```diff
   - BLOCK_SIZE = 8192
   + BLOCK_SIZE = 2048
   ```

   原版已使用三遍动态循环，因此源码中只需缩小启动时传入的BLOCK_SIZE；加载、累加和输出均按2048元素分块。对256线程CTA，逻辑向量均分由每线程约32元素降为约8元素，减少同时维护的中间状态和小N的无效掩码计算。
   这里是逻辑元素估算，不是物理寄存器公式。实测寄存器由60降到32，寄存器约束下的CTA/SM上限由4升到8；
   N=4096时NCU实际occupancy由{value(4096, 'abl_arithmetic', occupancy):.1f}%升到{value(4096, 'abl_grouped', occupancy):.1f}%，
   为更多warp并发与延迟隐藏提供条件。

   代价是块数和循环次数增加。N=15872时该版本实际occupancy虽达{value(15872, 'abl_grouped', occupancy):.1f}%，
   NCU耗时仍为{value(15872, 'abl_grouped', latency):.2f}µs，慢于原始基线的{value(15872, 'baseline', latency):.2f}µs。减少寄存器不保证所有尺寸都更快。

3. **按N范围静态展开。** 在2048分组基础上，增量1.0995×、累计1.3238×。

   ```diff
   - for off in range(0, N, BLOCK_SIZE):
   -     cols = off + tl.arange(0, BLOCK_SIZE)
   + # K为tl.constexpr，由设备内的N范围分支选择
   + for block in tl.static_range(K):
   +     cols = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
   ```

   求均值、方差和输出的三遍循环都作此修改。例如N≤4096选择2048宽、K=2，N≤6144选择2048宽、K=3；每块仍保留cols<N掩码，较大N保留动态回退。固定块数减少循环控制，
   并让编译器看见各次加载与计算之间的关系，从而改变指令调度和数据复用。

   展开也会增加代码体积与活跃值。本例寄存器从32升到80，
   N=15872的实际occupancy从{value(15872, 'abl_grouped', occupancy):.1f}%降到{value(15872, 'abl_unroll', occupancy):.1f}%，
   但NCU耗时从{value(15872, 'abl_grouped', latency):.2f}µs降到{value(15872, 'abl_unroll', latency):.2f}µs；
   long-scoreboard/issue比值从{value(15872, 'abl_grouped', wait):.2f}降到{value(15872, 'abl_unroll', wait):.2f}。
   该计数器反映等待全局/局部内存等结果依赖的统计，支持加载等待行为改善的解释，不能解释成同等比例的耗时下降。小N仍可能因资源开销退步，这是完整矩阵上的条件收益。

4. **均值阶段的输入加载使用.ca。** 在静态展开基础上，增量1.0667×、累计1.4123×。

   ```diff
   - a = tl.load(X + cols, mask=cols < N, other=0.0)
   + a = tl.load(X + cols, mask=cols < N, other=0.0,
   +             cache_modifier=".ca")
   ```

   只修改均值阶段，方差和输出阶段的X加载保留默认属性。.ca表示允许各级缓存，但默认加载也可能使用缓存；它不是“从不缓存变成缓存”。在该编译器和程序中，不同加载属性改变了跨阶段公共加载合并与值存活范围，
   使编译器更倾向于重新加载部分输入，而不是一直占用寄存器保存旧值。

   实际寄存器从80降到48，无spill，寄存器约束下的CTA/SM上限从3升到5。
   N=4096时NCU实际occupancy从{value(4096, 'abl_unroll', occupancy):.1f}%升到{value(4096, 'abl_ca', occupancy):.1f}%，
   DRAM吞吐从{value(4096, 'abl_unroll', dram):.3f}升到{value(4096, 'abl_ca', dram):.3f}TB/s，支持驻留能力和访存供给改善的解释。
   但N=15872仍从{value(15872, 'abl_unroll', latency):.2f}µs变为{value(15872, 'abl_ca', latency):.2f}µs，说明额外加载也有代价。
   收益不能全部归因于缓存命中率，.ca也不保证在任意程序中禁止加载合并。

5. **输出使用.cs。** 在均值加载.ca基础上，增量1.0141×、累计1.4322×。

   ```diff
   - tl.store(Y + cols, y, mask=mask)
   + tl.store(Y + cols, y, mask=mask, cache_modifier=".cs")
   ```

   仅修改Y写出，Mean和Rstd的标量store不改。.cs是流式缓存策略提示，意图是降低本kernel不再读取的输出数据对其他缓存数据的挤占。寄存器保持48、无spill，收益不是来自新的寄存器驻留档位。
   NCU中N=4096耗时由{value(4096, 'abl_ca', latency):.2f}µs降到{value(4096, 'abl_cs', latency):.2f}µs，
   但N=2560由{value(2560, 'abl_ca', latency):.2f}µs变为{value(2560, 'abl_cs', latency):.2f}µs；代表点没有一致改善。
   因此“减少缓存污染”是该策略的机制解释，尚未被隔离证明为全部收益，不能声称普遍提升L1/L2命中率或绕过所有缓存。

6. **N≤3072时使用1024分组。** 在以上改动基础上，增量1.0049×、累计1.4393×。

   ```diff
   # 设备分派示例：N≤3072时，_static_body的最后两项为块宽和固定块数
   - _static_body(X, Y, W, B, Mean, Rstd, stride, N, eps, 2048, 2)
   + _static_body(X, Y, W, B, Mean, Rstd, stride, N, eps, 1024, 3)
   ```

   N≤1024、N≤2048、N≤3072分别选择1024宽的1、2、3块，其他范围仍用2048。以N=2560为例，2048×2需处理4096个逻辑位置，1024×3只需3072个，减少25%的逻辑位置，但增加了一块；
   这不是实际DRAM流量减少25%的证明。寄存器仍为48，因此主要假设是无效掩码计算与块内工作量的折中，而非occupancy档位提高。
   该点NCU耗时从{value(2560, 'abl_cs', latency):.2f}µs降到{value(2560, 'abl_final', latency):.2f}µs，与该尺寸的收益方向一致。
   完整矩阵平均增量仅0.49%，不能解释成每个shape都提升0.49%。

"""
