"""Audit and report the final Norm+INT8 ablation."""

import sys
from pathlib import Path

OP = Path(__file__).resolve().parent
sys.path.insert(0, str(OP.parent))
from complex_suite.ablation_reporting import render  # noqa: E402

if __name__ == "__main__":
    render(
        OP,
        "Residual RMSNorm＋动态INT8",
        {
            "baseline": "原始融合研究基线",
            "factor": "amax因式分解，分离归约",
            "joint": "再联合sum/max",
            "stream": "再加三处输出.cs",
            "reciprocal": "再复用标量倒数",
            "early_z": "stream上只提前写Z",
            "final": "倒数＋提前写Z，当前最终版",
            "final_no_cs": "最终版去.cs",
            "final_separate": "最终版去联合统计，保留因式分解",
        },
        [
            ("baseline", "factor", "因式分解"),
            ("factor", "joint", "联合统计"),
            ("joint", "stream", "输出.cs"),
            ("stream", "reciprocal", "复用倒数"),
            ("reciprocal", "final", "提前写Z"),
            ("stream", "early_z", "无倒数时提前写Z"),
            ("early_z", "final", "已提前写Z时复用倒数"),
        ],
        [
            ("final_no_cs", "final", "移除.cs"),
            ("final_separate", "final", "移除联合统计"),
            ("early_z", "final", "移除倒数复用"),
            ("reciprocal", "final", "移除提前写Z"),
        ],
        "因式分解解除amax对rho的依赖，是联合统计的前提；没有伪造只移除因式分解的独立开关。"
        "stream/reciprocal/early_z/final构成完整2×2，可见早写Z与倒数的条件效应。"
        "Z必须精确，scale及重构误差受原FP64容差约束，Q最多相差1 code，不能称INT8逐位等价。"
        "这些变换可用公开Triton表达；当前最终版没有源码回写实测，第一轮8点源码结果不外推。",
    )
