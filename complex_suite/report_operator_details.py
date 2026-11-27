"""Render five complete operator sections from the audited paired samples."""

import csv
import hashlib
import io
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG = {
    "int4_gemm": {
        "title": "INT4 反量化融合 GEMM TTIR 优化",
        "shape": "M, N, K",
        "source": "FlagGems commit `ef8aea78fed8321c3e02a15bf31e847cd0579e4f` 的 "
        "`_weight_int4pack_mm_with_scales_and_zeros.py`，冻结提取为 [tutorial.py](int4_gemm/tutorial.py)，"
        "[上游快照](int4_gemm/upstream/source.py.snapshot)保留原实现。基线已经融合INT4解包、"
        "groupwise反量化和Tensor Core GEMM，不使用先完整反量化再matmul作为性能基线。",
        "environment": "H100，A/scale/zero及输出FP16、packed权重uint8，FP32累加；group=128，"
        "K为128倍数、N为16倍数，M允许尾部。逐shape固定原tile/warps/stages，1 CTA，"
        "新增点沿用原配置族，非全面autotune。主测GPU0/7轮、独立复测GPU1/5轮，每轮20ms。",
        "paths": [
            f"int4_gemm/ablation/results/v2_{p}/results.json"
            for p in ("main", "repeat")
        ],
        "steps": [
            (
                "首轮组合：共享low/high组参数、标量化组索引、非负索引unsigned",
                "baseline",
                "parent",
                "全部30点",
            ),
            ("缩窄scale/zero加载后广播", "parent", "narrow", "全部30点"),
            ("再删除契约内恒真的K mask", "narrow", "narrow_full_k", "全部30点"),
            ("再加BK64的128-K组内参数复用", "narrow_full_k", "group_pair", "全部30点"),
            (
                "再将group128除法替换为右移7位（最终版）",
                "group_pair",
                "final",
                "全部30点",
            ),
        ],
        "note": "首轮三项是组合对照，未独立分摊其内部贡献。K mask单独启用只约0.2%平均增益，"
        "在窄加载后约3.9%，显示交互；完整加入/移除及2×2见[消融报告](int4_gemm/report_ablation.md)。"
        "保留原FP16反量化舍入和dot累加次序；参考FP32 matmul atol=0.035/rtol=0.003。",
    },
    "moe_routing": {
        "title": "Grouped／biased MoE routing TTIR 优化",
        "shape": "tokens, experts",
        "source": "FlagGems commit `ef8aea78fed8321c3e02a15bf31e847cd0579e4f` 的 "
        "`runtime/backend/_nvidia/hopper/grouped_topk.py`，"
        "[来源与提取哈希](moe_routing/upstream/provenance.json)。冻结原来的两个设备kernel，"
        "第一个计算8个连续专家组的带bias top2之和，第二个选4组、8个expert，"
        "用未加bias的sigmoid分数生成归一化权重。",
        "environment": "H100，FP32，E=256/512、各15个tokens点；固定4 warps / 3 stages / 1 CTA。"
        "保留原两次launch、grid和workspace，同分优先小索引。"
        "主测GPU2/5轮、独立复测GPU3/3轮，每轮10ms，计入完整两阶段。",
        "paths": [
            f"moe_routing/ablation_v1/results/{p}/measurement/performance.json"
            for p in ("main", "repeat")
        ],
        "steps": [
            (
                "membership比较/归约改为连续组broadcast/reshape",
                "baseline",
                "add_membership",
                "全部30点",
            ),
            (
                "再把专家分数改为有序uint32 key",
                "add_membership",
                "add_ordered",
                "全部30点",
            ),
            (
                "再把原始权重gather移到选择循环结束后",
                "add_ordered",
                "add_gather",
                "全部30点",
            ),
            ("再以8×8稳定排名选择专家组", "add_gather", "add_rank", "全部30点"),
            ("再用双值归约计算组内top2", "add_rank", "add_pair", "全部30点"),
            ("再特化两阶段固定整数和stride", "add_pair", "add_constants", "全部30点"),
            ("再让每个活动CTA处理4组（最终版）", "add_constants", "final", "全部30点"),
        ],
        "note": "最终版移除双值归约的条件差异约0.5%，不同于加入时约1.9%，不能将二者当成"
        "可相加的独立贡献。FP32 group scores/weights与baseline位型相同、indices精确，"
        "另对CPU参考验证权重。完整移除和两阶段交叉见[消融报告](moe_routing/report_ablation.md)。",
    },
    "norm_quant": {
        "title": "Residual RMSNorm＋动态 INT8 TTIR 优化",
        "shape": "rows, N",
        "source": "明确指定的融合研究基线 [frozen_baseline.py](norm_quant/frozen_baseline.py)，"
        "参考FlagGems同一commit的 `fused_add_rms_norm.py` 和 `act_quant.py`，"
        "[来源与快照哈希](norm_quant/upstream/provenance.json)。不是生产kernel原样提取。"
        "单kernel融合残差加法、RMSNorm、amax与INT8量化，写出FP32 residual Z、scale和INT8 Q。",
        "environment": "H100，连续有限FP16 X/R/W、FP32统计和Z/scale、INT8 Q，无别名，N≤16384。"
        "每点固定原4/8 warps、3 stages、1 CTA；配置只做过有限预调优。"
        "GPU6串行主测7轮、独立进程复测5轮，每轮20ms。最终版仍为reciprocal_early。",
        "paths": [
            f"norm_quant/ablation_v1/results/split_20261124_{p}/results.json"
            for p in ("main", "repeat")
        ],
        "steps": [
            ("利用正rho将amax(abs(rho*u))因式分解", "baseline", "factor", "全部30点"),
            ("再将sum(z²)/max(abs(u))合为tuple归约", "factor", "joint", "全部30点"),
            ("再给Z/Q/scale输出加.cs", "joint", "stream", "全部30点"),
            ("再将量化逐元素除法改为复用标量倒数", "stream", "reciprocal", "全部30点"),
            ("再提前写出Z（最终版）", "reciprocal", "final", "全部30点"),
        ],
        "note": "因式分解是联合统计的前提。.cs加入时平均仅约0.2%–0.3%，不单独宣称稳定收益。"
        "Z精确，scale对FP64 rtol=3e-5/atol=1e-9，Q允许最多差1 code并检查重构误差，"
        "不声称INT8逐位等价。倒数/早写Z完整2×2及最终移除见"
        "[消融报告](norm_quant/report_ablation.md)。",
    },
    "attention_dkdv": {
        "title": "Attention 反向 dK／dV TTIR 优化",
        "shape": "B, H, N, D, causal",
        "source": "官方Triton v3.7.0 `06-fused-attention.py` 教程，"
        "[冻结上游与提取记录](attention_dkdv/upstream/provenance.json)。"
        "从完整backward提取截至dQ之前的dK/dV计算，baseline每CTA负责128个key；"
        "输入LSE/delta与预缩放K均已准备好，只计dK/dV kernel。",
        "environment": "H100，SM固定1200MHz，FP16输入、FP32 dot累加，"
        "4 warps / 5 stages / 1 CTA。D64/128、因果/非因果共32点。"
        "主测GPU4/7轮/seed20、独立复测GPU5/5轮/seed123，每轮20ms；结束恢复自动频率。",
        "paths": [
            f"attention_dkdv/results/ablation_v2/{p}/measurement/benchmark.json"
            for p in ("main", "repeat")
        ],
        "steps": [
            (
                "dV的cast/dot后移到dK之后（D64最终版）",
                "baseline",
                "final",
                "D64全部16点",
            ),
            ("每CTA串行处理两个64-key子块", "baseline", "rows64", "D128非因果8点"),
            ("再依据query坐标重算指针", "rows64", "structure", "D128非因果8点"),
            ("再后移dV（该组最终版）", "structure", "final", "D128非因果8点"),
            ("每CTA串行处理两个64-key子块", "baseline", "rows64", "D128因果8点"),
            ("再静态展开两个子块", "rows64", "structure", "D128因果8点"),
            ("再后移dV（该组最终版）", "structure", "final", "D128因果8点"),
        ],
        "note": "方法表各行明确标注适用子集；其累计加速是该子集相对原baseline，不能当作全32点收益。"
        "下表及最终算术/几何均值均覆盖全部32点。64-key与结构调整存在依赖，联合移除不冒充"
        "纯单项贡献。当前dK/dV保持基线逐位一致，另有小N CPU FP64数学参考；"
        "不含forward、K预缩放、delta准备或完整backward。完整交互见"
        "[消融报告](attention_dkdv/report_ablation.md)。",
    },
    "partial_topk": {
        "title": "大 N Partial Top-K TTIR 优化",
        "shape": "rows, N, K",
        "source": "两阶段研究基线 [frozen_baseline.py](partial_topk/frozen_baseline.py)，"
        "采用FlagGems `ops/topk.py:topk_stage1_kernel` 的分块重复选择思路，"
        "[来源与快照哈希](partial_topk/upstream/provenance.json)。每1024输入先保留K项，"
        "merge在局部候选上再次选全局Top-K；不是生产实现原样提取或全排序基线。",
        "environment": "H100，连续有限FP32，K=8/16、chunk1024，4 warps / 3 stages / 1 CTA，"
        "无逐点autotune。保留两kernel、workspace、ABI/grid，计入完整local＋merge。"
        "主测GPU7/7轮、独立复测GPU1/5轮，每轮20ms；使用已修正的v2完整会话。",
        "paths": [
            f"partial_topk/ablation_v1/results/split_v2_20261124_{p}/results.json"
            for p in ("main", "repeat")
        ],
        "steps": [
            (
                "FP32 max改为每轮重算有序uint32 key（负结果）",
                "baseline",
                "ordered_recompute",
                "全部30点",
            ),
            (
                "再把key编码移到循环外、携带key状态",
                "ordered_recompute",
                "carry",
                "全部30点",
            ),
            ("再固定原始ids，去掉ids循环状态", "carry", "static_ids", "全部30点"),
            ("再静态展开K次选择", "static_ids", "static_unroll", "全部30点"),
            (
                "再用有限输入/完整tile合同简化mask和耗尽判断",
                "static_unroll",
                "parent",
                "全部30点",
            ),
            ("merge再改成已排序局部列表的头部归并", "parent", "heads", "全部30点"),
            (
                "头部选择再用packed uint64稳定比较（最终版）",
                "heads",
                "final",
                "全部30点",
            ),
        ],
        "note": "每轮重算key的回退照实保留；packed头部比较相对uint32版本约0.4%–0.5%，"
        "不单独宣称可靠收益。indices严格精确、value数值精确，±0等价；头部归并依赖局部列表有序。"
        "完整local/merge 2×2及移除对照见[消融报告](partial_topk/report_ablation.md)。",
    },
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def subset(shape, scope):
    if scope.startswith("D64"):
        return shape[3] == 64
    if scope.startswith("D128"):
        return shape[3] == 128 and bool(shape[4]) == ("非因果" not in scope)
    return True


def load_runs(op, spec, evidence):
    summary_path = ROOT / evidence["summary_path"]
    assert sha(summary_path) == evidence["summary_sha256"]
    summary = read(summary_path)
    for name, expected in summary.get(
        "sources", summary.get("source_sha256", {})
    ).items():
        assert sha(ROOT / op / name) == expected, (op, name)
    runs = []
    for name in spec["paths"]:
        raw = read(ROOT / name)
        assert raw.get("passed", True) and raw.get("aa_passed", True)
        rows = {}
        for row in raw.get("points", raw.get("records", [])):
            timing = row.get("timing", row)
            samples = [s.get("us", s.get("times_us")) for s in timing["samples"]]
            assert len(samples) == raw["rounds"]
            assert all(
                math.isfinite(v) and v > 0
                for sample in samples
                for v in sample.values()
            )
            if op == "attention_dkdv":
                aa = st.median(s["baseline"] / s["baseline_copy"] for s in samples)
            else:
                aa = st.median(a["ratio"] for a in timing["aa"])
            assert 0.97 <= aa <= 1.03
            shape = tuple(row["shape"])
            assert shape not in rows
            rows[shape] = samples
        assert len(rows) == evidence["shape_count"]
        runs.append(rows)
    assert runs[0].keys() == runs[1].keys()
    return runs


def ratios(rows, before, after, scope="全部30点"):
    return [
        st.median(s[before] / s[after] for s in samples)
        for shape, samples in rows.items()
        if subset(shape, scope)
    ]


def render(op, spec, evidence):
    runs = load_runs(op, spec, evidence)
    speeds = [ratios(r, "baseline", "final") for r in runs]
    averages = [
        dict(arithmetic=st.mean(v), geometric=st.geometric_mean(v)) for v in speeds
    ]
    for avg, key in zip(averages, ("main", "repeat")):
        assert math.isclose(avg["geometric"], evidence[key], rel_tol=1e-12)
    lines = [
        f"# {spec['title']}",
        "",
        f"来源：{spec['source']}",
        "",
        f"环境：官方Triton3.7.0，{spec['environment']}"
        "计时为预分配CUDA graph设备时间，不含分配、编译或CPU launch；"
        "每次在同一卡内交错比较baseline和候选，A/A通过。",
        "",
        f"**最终平均加速比：主测算术平均{averages[0]['arithmetic']:.6f}×、"
        f"几何平均{averages[0]['geometric']:.6f}×；独立复测算术平均"
        f"{averages[1]['arithmetic']:.6f}×、几何平均{averages[1]['geometric']:.6f}×。** "
        f"覆盖全部{evidence['shape_count']}个shape，为本矩阵内A级性能证据。",
        "",
        "每个shape先取同轮baseline耗时/最终版耗时的中位数；算术平均是这些加速比的等权平均，"
        "几何平均是它们乘积的n次方根，均不是所有耗时求和后的比值。"
        "耗时列是各自样本的中位数，单位µs，因此其相除不必恰好等于配对比值中位数。",
        "",
        f"| Shape ({spec['shape']}) | 主测基线 µs | 主测最终 µs | 主测加速比 | "
        "复测基线 µs | 复测最终 µs | 复测加速比 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    details = []
    shapes = sorted(
        runs[0],
        key=(lambda s: (s[3], s[4], *s[:3])) if op == "attention_dkdv" else None,
    )
    for shape in shapes:
        cells, phases = [], {}
        for phase, rows in zip(("main", "repeat"), runs):
            samples = rows[shape]
            base = st.median(s["baseline"] for s in samples)
            final = st.median(s["final"] for s in samples)
            speed = st.median(s["baseline"] / s["final"] for s in samples)
            cells.extend([f"{base:.4f}", f"{final:.4f}", f"{speed:.6f}×"])
            phases[phase] = dict(baseline_us=base, final_us=final, paired_speedup=speed)
        lines.append(f"| {shape} | " + " | ".join(cells) + " |")
        details.append(dict(shape=shape, **phases))
    lines.extend(
        [
            "",
            "优化方法与性能提升：按下表顺序逐项加入。增量加速和提升百分比均相对前一版本，"
            "累计加速相对原始baseline；全部由同会话各shape的同轮时间比中位数求几何平均，"
            "不乘历史轮次均值。百分比是吞吐提升，不是耗时下降，各项有交互，不能相加。",
            "",
            "| 优化方法 | 统计范围 | 主测增量／提升 | 复测增量／提升 | 主测累计加速 | 复测累计加速 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    methods = []
    for label, before, after, scope in spec["steps"]:
        inc = [st.geometric_mean(ratios(r, before, after, scope)) for r in runs]
        cumulative = [
            st.geometric_mean(ratios(r, "baseline", after, scope)) for r in runs
        ]
        changes = [f"{v:.6f}× / {(v - 1) * 100:+.2f}%" for v in inc]
        lines.append(
            f"| {label} | {scope} | {changes[0]} | {changes[1]} | "
            f"{cumulative[0]:.6f}× | {cumulative[1]:.6f}× |"
        )
        methods.append(
            dict(
                method=label,
                before=before,
                after=after,
                scope=scope,
                incremental=dict(main=inc[0], repeat=inc[1]),
                cumulative=dict(main=cumulative[0], repeat=cumulative[1]),
            )
        )
    lines.extend(
        [
            "",
            spec["note"],
            "",
            "本节全部耗时、最终均值和方法增益统一取自本次固定r6最终版的消融主/复测，"
            "不混用首轮小矩阵或历史r6正式会话。新增中间消融未全部重跑sanitizer；"
            "最终版沿用r6已有通过记录。",
            "",
            f"证据：[主测原始样本]({spec['paths'][0]}) · "
            f"[独立复测]({spec['paths'][1]}) · [详细数据与均值]({op}/report_details.json) · "
            f"[实验入口]({op}/README.md)。",
            "",
        ]
    )
    data = dict(
        operator=op,
        evidence_grade="A",
        shape_count=len(shapes),
        averages=dict(main=averages[0], repeat=averages[1]),
        shapes=details,
        methods=methods,
        sources={p: sha(ROOT / p) for p in spec["paths"]},
    )
    return single_section(op, spec, runs[0]), data


def single_section(op, spec, rows):
    """Present one fixed timing session with three performance columns."""
    speedups = ratios(rows, "baseline", "final")
    environment = re.sub(r"主测GPU.*$|GPU6串行主测.*$", "", spec["environment"])
    lines = [
        f"# {spec['title']}",
        "",
        f"来源：{spec['source']}",
        "",
        f"环境：官方Triton3.7.0，{environment}"
        "预分配CUDA graph设备计时，不含分配、编译或CPU launch，配对计时与A/A检查通过。",
        "",
        f"**最终平均加速比：算术平均{st.mean(speedups):.6f}×，"
        f"几何平均{st.geometric_mean(speedups):.6f}×。** 覆盖全部{len(rows)}个shape，A级性能证据。",
        "",
        "时间为同一会话中各轮耗时的中位数，单位µs；加速比为同轮基线/优化时间比的中位数，"
        "不必恰好等于两列时间中位数之比。算术和几何平均均对各shape加速比等权计算。",
        "",
        f"| Shape ({spec['shape']}) | 基线时间（µs） | 优化时间（µs） | 加速比 |",
        "|---|---:|---:|---:|",
    ]
    for shape in sorted(
        rows, key=(lambda s: (s[3], s[4], *s[:3])) if op == "attention_dkdv" else None
    ):
        samples = rows[shape]
        base = st.median(s["baseline"] for s in samples)
        final = st.median(s["final"] for s in samples)
        speed = st.median(s["baseline"] / s["final"] for s in samples)
        lines.append(f"| {shape} | {base:.4f} | {final:.4f} | {speed:.6f}× |")
    lines.extend(
        [
            "",
            "优化方法与性能提升：",
            "",
        ]
    )
    for index, (label, before, after, scope) in enumerate(spec["steps"], 1):
        from complex_suite.report_method_details import complex_item

        inc = st.geometric_mean(ratios(rows, before, after, scope))
        cumulative = st.geometric_mean(ratios(rows, "baseline", after, scope))
        scope_text = f"（{scope}）" if scope != "全部30点" else ""
        lines.append(
            complex_item(
                op,
                index,
                label + scope_text,
                f"增量{inc:.6f}×、累计{cumulative:.6f}×。",
            )
        )
        lines.append("")
    lines.extend(
        [
            "",
            "条件收益不能相加。增量相对前一版本，累计相对原始基线，各项存在依赖和交互。",
            "",
            spec["note"],
            "",
            "全部时间、加速比和方法收益使用同一组原始配对样本；冻结最终选择不变。"
            "新增中间消融未全部重跑sanitizer，最终版沿用已有通过记录。",
            "",
            f"证据：[原始配对样本]({spec['paths'][0]}) · "
            f"[实验入口]({op}/README.md)。",
            "",
        ]
    )
    from complex_suite.report_public_presentation import public_section

    return public_section(op, "\n".join(lines))


def write_operator(op, evidence=None):
    if evidence is None:
        evidence = next(
            x
            for x in read(ROOT / "complex_suite/final_ablation_summary.json")[
                "operators"
            ]
            if x["operator"] == op
        )
    section, data = render(op, CONFIG[op], evidence)
    # Sections link from the repository root; standalone reports use relative links.
    local = re.sub(
        r"\]\(([^)]+)\)",
        lambda m: m[0] if m[1].startswith("https://") else "](" + "../" + m[1] + ")",
        section,
    )
    (ROOT / op / "report.md").write_text(local)
    (ROOT / op / "report_details.json").write_text(
        json.dumps(public_data(data), indent=2, ensure_ascii=False) + "\n"
    )
    return section, data


def public_data(data):
    return dict(
        operator=data["operator"],
        evidence_grade=data["evidence_grade"],
        arithmetic=data["averages"]["main"]["arithmetic"],
        geometric=data["averages"]["main"]["geometric"],
        rows=[
            dict(
                shape=r["shape"],
                baseline_us=r["main"]["baseline_us"],
                optimized_us=r["main"]["final_us"],
                speedup=r["main"]["paired_speedup"],
            )
            for r in data["shapes"]
        ],
        methods=[
            dict(
                method=m["method"],
                scope=m["scope"],
                incremental=m["incremental"]["main"],
                cumulative=m["cumulative"]["main"],
            )
            for m in data["methods"]
        ],
        sources={
            p: h
            for p, h in data["sources"].items()
            if p == CONFIG[data["operator"]]["paths"][0]
        },
    )


def main():
    all_report = ROOT / "all_report.md"
    from complex_suite.report_basic_details import write_basics

    tail, basic_data = write_basics()
    evidence = read(ROOT / "complex_suite/final_ablation_summary.json")
    sections, operators = [], []
    for item in evidence["operators"]:
        section, data = write_operator(item["operator"], item)
        sections.append(section)
        operators.append(data)
    assert sum(d["shape_count"] for d in operators) == 152
    intro = "# TTIR 算子优化实验\n\n" + (
        "本报告覆盖13项算子实验、379个测试shape，先展示基础算子，最后展示新增的五个复杂算子。"
        "每项包含基线来源、各shape的基线时间、优化时间与加速比，以及优化方法收益和最终算术、几何平均加速比。"
        "所有结果均由原始配对样本计算，不混合不同实验矩阵，也不计算跨算子总平均。\n\n"
        "**TTIR优化的方法与思路**\n\n"
        "从计算、数据流和访存入手，直接改写TTIR，再由编译器生成设备代码：\n\n"
        "1. **减少重复计算**：复用倒数、提取公共系数、合并统计与归约。\n"
        "2. **控制中间状态**：缩小分块、缩短值存活时间，在寄存器占用与重复加载之间取舍。\n"
        "3. **简化循环与分支**：按尺寸分派、静态展开，利用已验证前提删除无效掩码和冗余判断。\n"
        "4. **改善访存与复用**：窄加载后广播、组内共享数据、调整缓存策略，必要时采用TMA。\n"
        "5. **重组算法与调度**：改写循环状态、批量处理短行、优化归并；允许调整启动方式时，减少小输入的kernel数量。\n"
        "6. **用实验确认收益**：先验证数值，再做配对计时和逐项消融，结合寄存器与NCU指标分析原因。\n\n"
        "下一次优化新算子，可直接使用[TTIR 优化决策模型（流程图、候选选择与任务卡）]"
        "(knowledge/ttir-optimization-workflow.md)。\n\n"
    )
    from complex_suite.report_public_presentation import overview

    table = overview(basic_data + [public_data(d) for d in operators])
    all_report.write_text(intro + table + tail + "\n" + "\n".join(sections))
    (ROOT / "complex_suite/operator_details_summary.json").write_text(
        json.dumps(
            dict(
                unique_shapes=152 + sum(len(d["rows"]) for d in basic_data),
                operators=[public_data(d) for d in operators] + basic_data,
            ),
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["operator", "shape", "baseline_us", "optimized_us", "speedup"])
    for op in operators:
        for row in op["shapes"]:
            values = row["main"]
            writer.writerow(
                [
                    op["operator"],
                    str(tuple(row["shape"])),
                    values["baseline_us"],
                    values["final_us"],
                    values["paired_speedup"],
                ]
            )
    for op in basic_data:
        for row in op["rows"]:
            writer.writerow(
                [
                    op["operator"],
                    str(tuple(row["shape"])),
                    row["baseline_us"],
                    row["optimized_us"],
                    row["speedup"],
                ]
            )
    (ROOT / "complex_suite/operator_details.csv").write_text(stream.getvalue())
    from complex_suite.report_table_presentation import normalize_current

    normalize_current()
    print(
        "Rendered 13 operators, 379 shapes and one set of baseline/optimized time and speedup columns"
    )


if __name__ == "__main__":
    main()
