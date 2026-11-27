"""Present every basic operator with one fixed timing dataset."""

import hashlib
import json
import math
import os
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    "layer_norm": dict(
        title="LayerNorm 前向 TTIR 优化",
        shape="M, N",
        path="layer_norm/results/ablation/full.json",
        candidate="abl_final",
        source="官方Triton v3.7.0 `05-layer-norm.py` 的前向实现，冻结BLOCK_SIZE=8192后导出TTIR直接优化。"
        "原始教程与baseline保持不变，两版均经同一Triton原生后端编译。",
        environment="H100，FP16输入输出、FP32统计与累加，8 warps / 3 stages / 1 CTA。"
        "M=4096，N=1024…15872、步长512；N/stride为16倍数，无别名。5轮、每轮10ms。",
        methods="\n\n1. 单独复用1/N为1.0017×，单独改rsqrt为1.0047×，两项合并1.0065×。\n"
        "2. 加入2048窄分组与三遍动态循环：增量1.1962×、累计1.2041×。\n"
        "3. 按N范围静态展开：增量1.0995×、累计1.3238×。\n"
        "4. 均值加载.ca：增量1.0667×、累计1.4123×。\n"
        "5. 输出.cs：增量1.0141×、累计1.4322×。\n"
        "6. N≤3072使用1024分组：增量1.0049×、累计1.4393×。\n\n"
        "条件收益不能相加，.ca同时影响load合并和存活范围，不仅是缓存效应。",
        scope="原数值合同和边界检查通过；当前最终版为r2_mixed_stream。性能限于预分配输出的kernel时间。",
    ),
    "mean": dict(
        title="Mean TTIR 优化",
        shape="M, N, K",
        path="mean/results/runs/round2/final/main.json",
        candidate="r2_final",
        source="FlagGems `mean_dim`，冻结原始启发式后导出TTIR。连续输入视为[M,N,K]，沿N取均值；"
        "只修改K≥1024的向量循环路径。源码/文件baseline的LLVM/PTX/cubin一致。",
        environment="官方Triton3.7.0/H100，FP16输入输出、FP32累加。向量路径BLOCK_SIZE_K=128、"
        "VEC_SIZE=8，8 warps / 3 stages / 1 CTA；其余路径/grid不变。30点，另有12个边界正确性点，7轮20ms。",
        methods="\n\n1. 将128×8的逻辑K状态扁平化为1×1024，并将地址索引i64缩为i32：累计1.1008×。\n"
        "2. 再将循环展开为16行预加载并按原顺序累加，N<8保留原路径，单列尾块采用256×1的N方向归约："
        "组合增量1.2815×、累计1.4106×。\n\n"
        "展开、短N分支和尾列归约按组合报告，条件收益不可相加。"
        "实际修改的向量路径18点几何均值1.7743×；其余12点实现不变。",
        scope="全部数值、编译链和memcheck/racecheck检查通过。固定30点均曾用于调优和回归，不能外推任意输入。",
    ),
    "var_mean": dict(
        title="var_mean TTIR 优化",
        shape="M, N",
        path="var_mean/results/runs/promotion_ac_v1/full.json",
        candidate="optimized",
        source="FlagGems `var_mean_welford_kernel` 行归约，固定BLOCK_N=1024，保留int64行索引语义。"
        "冻结基线采用先均值再中心化平方和的两遍算法。两版经相同Triton原生后端编译。",
        environment="官方Triton3.7.0/H100，连续FP32输入输出、最后一维按行归约，4 warps / 3 stages / 1 CTA。"
        "M∈{1,17,128,1024,4096}，N∈{128,512,1040,4096,8192,16384}，correction=1，7轮20ms。",
        methods="\n\n1. 短行按N≤128/512/1024使用对应宽度单tile，加载一次后复用输入（A）：单独1.1443×。\n"
        "2. 声明合同保证的N为16倍数（C）：单独1.2909×。\n"
        "3. 在短行优化A上加入整除属性C：增量1.1721×、消融累计1.3416×，最终采用A+C。\n"
        "4. 长行循环宽度1024→256（B）：单独1.1658×，但在A+C上加入B仅0.9665×、耗时增加3.46%，未采用。\n\n"
        "条件收益不能相加。此处消融与前述晋升验证属于不同会话，不混算平均。",
        scope="数值、编译链与memcheck/racecheck通过。支持连续FP32、1≤M≤4096、N为16倍数且≤16384、"
        "correction=0/1，无别名；不覆盖非连续、NaN/Inf或全张量两阶段路径。",
    ),
    "mm": dict(
        title="MM TTIR 优化",
        shape="M, N, K",
        path="mm/host_allshape_v1/results/main.json",
        candidate="host_pid",
        source="官方Triton v3.7.0矩阵乘教程，逐shape autotune后冻结六组配置。"
        "改用host descriptor/TMA，匹配基线使用普通指针load/store但保留同一新host ABI。",
        environment="H100，FP16输入输出、FP32累加，GROUP_M=8、1 CTA，tile/warps/stages按冻结配置族。"
        "M=N=K=256…4096、步长128，共31点；准备好的descriptor/输出的CUDA graph设备时间，7轮20ms。",
        methods="\n\n1. 将普通load/store改为host TMA：完整31点相对匹配基线的几何加速比1.0449×。\n"
        "2. 在host TMA上将组内行号改为pid%size：完整31点相对匹配基线的几何加速比1.0492×。\n"
        "3. 按配置组验收：384³…640³选择host TMA，768³…1408³选择host TMA+PID，其余保留基线。\n\n"
        "两候选各自与匹配基线配对，两个均值之比不能作为独立确认的PID因果增量，条件收益不能相加。"
        "本表继续展示全部host TMA+PID候选实测，包括回退点，不把回退路线伪装为1×实测。",
        scope="‡表示固定计时修补后仍有控制无效，该表整体为B级研究观测，不能声称全部31点稳定收益。"
        "数值和memcheck通过；racecheck仍有72 warnings、0 errors，未获同步资格。"
        "不计descriptor准备的端到端成本，最终选择仍按已有配置组决定。",
    ),
    "sum": dict(
        title="Sum TTIR 优化与消融",
        shape="n,",
        path="sum/ablation/results/main.json",
        candidate="baseline->single",
        source="FlagGems全量两阶段sum，按sqrt(n)选择分块后冻结TTIR；源码/文件baseline的LLVM/PTX/cubin一致。"
        "最终single对n≤16384使用一个CTA直接写out，其余保留原两阶段。",
        environment="官方Triton3.7.0/H100，连续一维FP32，4 warps / 3 stages / 1 CTA，30点，7轮20ms。"
        "完整设备序列计时；该优化改变输出绑定、启动配置和kernel数量。",
        methods="\n\n1. 小输入使用单CTA，但仍保留第二个拷贝kernel：0.9991×。\n"
        "2. 进一步直接写出并移除第二个kernel：累计1.3569×，小输入15点为1.8402×；"
        "保留与移除第二个kernel的直接配对增量为1.3575×。\n"
        "3. 将索引i64缩为i32：0.9995×；reshape分组归约：0.9997×；两项组合：0.9972×。\n"
        "4. 加载使用.cg：0.9429×；加载使用.cs：1.0001×。\n"
        "5. 完整CTA使用无mask加载：0.9945×。\n"
        "6. 在最终版增加大输入无mask分支：增量0.9998×，未采用。\n\n"
        "条件收益不能相加。除注明的直接配对增量和小输入子集外，其余数字均相对原始基线、覆盖完整30点。",
        scope="数值、编译链、memcheck/racecheck通过；n>16384路径没有算法改动，接近1×的变化为计时波动。"
        "不能称作保持原kernel数和固定启动契约下的纯TTIR收益。",
    ),
    "logsumexp": dict(
        title="Logsumexp TTIR 优化",
        shape="M, N, K",
        path="logsumexp/results/runs/r4_final/main_baseline.json",
        candidate="r4_focus",
        source="FlagGems `ops/logsumexp.py`，包含内层与非内层归约路径。"
        "按原启发式导出冻结TTIR再优化，两版共享Triton原生编译链和主机接口。",
        environment="官方Triton3.7.0/H100，连续FP16输入输出、FP32指数与累加。"
        "输入视为[M,N,K]，沿N归约，K=1为内层、K=64为非内层；原配置族固定，30点，7轮20ms。",
        methods="\n\n1. 缩减非内层归约的无效填充，并将在线递推中的两次指数计算合并为一次。\n"
        "2. 大尺寸内层归约先在FP16输入上求max，再转FP32计算指数和累加；长行采用单块归约，保留超长输入的分块路径。\n"
        "3. 小尺寸输入合并多行并采用32-lane向量流，将范围安全的地址中间量缩为i32。\n"
        "4. 组合以上改动：累计1.1221×。\n\n"
        "以上方法按组合验证，未独立分摊每项收益，也未把不同条件下的收益相加；"
        "回退或持平shape全部保留。",
        scope="51点数值/边界、编译链及memcheck/racecheck通过；限定已映射连续FP16、有限值/-inf、"
        "无别名，不覆盖NaN/+inf、任意stride、其他dtype或完整API。",
    ),
    "vector_add": dict(
        title="Vector add TTIR 优化",
        shape="N,",
        path="vector_add/results/runs/promote_cs_20261119/full.json",
        candidate="optimized",
        source="官方Triton向量加教程，固定commit `22459e3dbcef2b403aab7ff2a3981309a3537e39`，"
        "冻结源码导出TTIR。源码/文件入口的LLVM/PTX/cubin一致。",
        environment="官方Triton3.7.0/H100，FP32连续输入、独立buffer、16-byte对齐，N为16倍数且≤2²⁷。"
        "每CTA1024元素、4 warps，16点，9轮20ms。",
        methods="\n\n1. 将输出tt.store的cacheModifier设为.cs：完整矩阵几何平均1.0094×，局部N=2²²约1.1539×。\n\n"
        "这是唯一最终改动，完整矩阵平均收益小，未达到原全矩阵1.02×门槛，最终选择来自用户明确接受局部收益。"
        "不把此前其他设备上的约1.206×当作本会话数据。",
        scope="36个数值用例和A/A通过。当前TTIR/cubin与已有sanitizer证据相同，因此复用零错误记录，"
        "不将其称作本会话重新执行。无冷缓存或API端到端性能结论。",
    ),
    "layer_norm/backward_dx": dict(
        title="LayerNorm backward dX TTIR 优化",
        shape="M, N",
        path="layer_norm/backward_dx/results/round4/selected_benchmark.json",
        candidate="optimized",
        source="官方Triton v3.7.0 `_layer_norm_bwd_dx_fused`的dX计算段提取，"
        "保留两次规约及dX写出。双方都不计算dW/dB或使用梯度锁，不代表完整fused backward。",
        environment="官方Triton3.7.0/H100，FP16 X/DY/W/DX、FP32计算和保存统计量，"
        "N/stride为16倍数、N≤16384；按next-power-of-two固定5个family配置。30点，3轮20ms。",
        methods="\n\n1. 1024/2048宽度组采用已验收的r4_center_maskfold。\n"
        "2. 4096宽度组采用已验收的r4_evict。\n"
        "3. 8192宽度组保留原始基线。\n"
        "4. 16384宽度组采用joint_coefficients。\n\n"
        "以上按配置组组合选择，未独立分摊每项对完整矩阵的贡献。"
        "本表使用实际optimized别名直接启动的配对数据，"
        "不以搜索阶段各候选数字拼成时间。完整矩阵未达整体10%目标。",
        scope="基线/提取源码路径、实际optimized编译链、数值压力及计时控制有独立存证。"
        "仅计dX kernel，不包含统计量计算、前向、autograd、dW/dB或完整反向。",
    ),
}


def load(spec):
    path = ROOT / spec["path"]
    data = json.loads(path.read_text())
    records = data["records"]
    sources = {spec["path"]: hashlib.sha256(path.read_bytes()).hexdigest()}
    if spec["candidate"] == "host_pid":
        patch_path = ROOT / "mm/host_allshape_v1/results/repair_main.json"
        patch = json.loads(patch_path.read_text())
        replacements = {
            (r["family"], r["name"], tuple(r["shape"])): r for r in patch["records"]
        }
        records = [
            replacements.get((r["family"], r["name"], tuple(r["shape"])), r)
            for r in records
        ]
        sources[str(patch_path.relative_to(ROOT))] = hashlib.sha256(
            patch_path.read_bytes()
        ).hexdigest()
    rows = []
    invalid_controls = {
        tuple(r["shape"])
        for r in records
        if spec["candidate"] == "host_pid"
        and r.get("name") == "control"
        and not r["valid"]
    }
    for record in records:
        name = record.get("name", record.get("kernel"))
        if name != spec["candidate"]:
            continue
        shape = record.get("shape")
        if shape is None:
            shape = [record["M"], record["N"]] if "M" in record else [record["n"]]
        samples = record.get("pairs", record.get("samples"))
        baseline = st.median(s["baseline_us"] for s in samples)
        optimized = st.median(s["candidate_us"] for s in samples)
        speedup = st.median(s["baseline_us"] / s["candidate_us"] for s in samples)
        assert all(math.isfinite(v) and v > 0 for v in (baseline, optimized, speedup))
        rows.append(
            dict(
                shape=shape,
                baseline_us=baseline,
                optimized_us=optimized,
                speedup=speedup,
                valid=record.get("valid", True)
                and tuple(shape) not in invalid_controls,
            )
        )
    expected = (
        31
        if spec["candidate"] == "host_pid"
        else (
            16
            if spec["candidate"] == "optimized" and "vector_add" in spec["path"]
            else 30
        )
    )
    assert len(rows) == expected
    assert len({tuple(r["shape"]) for r in rows}) == expected
    if spec["candidate"] == "host_pid":
        rows.sort(key=lambda row: row["shape"])
    return rows, sources


def write_basics():
    sections, metadata = [], []
    for op, spec in SPECS.items():
        rows, sources = load(spec)
        methods = spec["methods"]
        if op == "layer_norm":
            from complex_suite.report_layernorm_methods import render

            methods = render()
        else:
            from complex_suite.report_method_details import basic

            methods = basic(op)
        speeds = [r["speedup"] for r in rows]
        arithmetic, geometric = st.mean(speeds), st.geometric_mean(speeds)
        grade = "B" if op == "mm" else "A"
        lines = [
            f"# {spec['title']}",
            "",
            f"来源：{spec['source']}",
            "",
            f"环境：{spec['environment']}预分配CUDA graph设备时间，不含分配、编译和CPU launch。",
            "",
            f"**最终平均加速比：算术平均{arithmetic:.6f}×，几何平均{geometric:.6f}×。** "
            f"共{len(rows)}个shape，{grade}级证据。",
            "",
            "时间是同一组配对样本各轮的中位数，单位µs；加速比为同轮基线/优化时间比的中位数。"
            "算术/几何平均按各shape加速比等权计算，未按耗时加权。",
            "",
            f"| Shape ({spec['shape']}) | 基线时间（µs） | 优化时间（µs） | 加速比 |",
            "|---|---:|---:|---:|",
        ]
        for row in rows:
            marker = "" if row["valid"] else "‡"
            lines.append(
                f"| {tuple(row['shape'])} | {row['baseline_us']:.4f} | "
                f"{row['optimized_us']:.4f} | {row['speedup']:.6f}×{marker} |"
            )
        lines.extend(
            [
                "",
                f"优化方法与性能提升：{methods}",
                "",
                "" if op == "layer_norm" else spec["scope"],
                "",
                f"证据：[原始配对样本]({spec['path']}) · [实验入口]({op}/README.md)。",
                "",
            ]
        )
        from complex_suite.report_public_presentation import public_section

        section = public_section(op, "\n".join(lines))
        sections.append(section)
        folder = ROOT / op
        local = re.sub(
            r"\]\(([^)]+)\)",
            lambda m: (
                m[0]
                if m[1].startswith("https://")
                else "](" + os.path.relpath(ROOT / m[1], folder) + ")"
            ),
            section,
        )
        (folder / "report.md").write_text(local)
        result = dict(
            operator=op,
            evidence_grade=grade,
            arithmetic=arithmetic,
            geometric=geometric,
            rows=rows,
            sources=sources,
        )
        (folder / "report_details.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        )
        metadata.append(result)
    return "\n".join(sections), metadata
