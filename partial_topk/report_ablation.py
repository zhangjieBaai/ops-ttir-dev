"""Audit and report the final Top-K chain and stage factorial."""

import sys
from pathlib import Path

OP = Path(__file__).resolve().parent
sys.path.insert(0, str(OP.parent))
from complex_suite.ablation_reporting import render  # noqa: E402

if __name__ == "__main__":
    render(
        OP,
        "大N Partial Top-K",
        {
            "baseline": "原两阶段研究基线",
            "ordered_recompute": "每轮重算有序整数key",
            "carry": "循环外编码，key携带状态",
            "static_ids": "再固定原始ids",
            "static_unroll": "再静态展开K",
            "parent": "r5合同特化的有效parent",
            "heads": "以有序块头部归并，uint32 max＋min-id",
            "final_local_only": "最终local＋baseline merge",
            "final_merge_only": "baseline local＋最终merge",
            "final": "最终local＋packed uint64头部merge",
        },
        [
            ("baseline", "ordered_recompute", "整数key但每轮重算，保留负结果"),
            ("ordered_recompute", "carry", "把key移到循环外"),
            ("carry", "static_ids", "固定ids"),
            ("static_ids", "static_unroll", "展开K循环"),
            ("static_unroll", "parent", "合同特化"),
            ("parent", "heads", "头部归并"),
            ("heads", "final", "packed稳定比较"),
        ],
        [
            ("final_merge_only", "final", "移除整个local组合"),
            ("final_local_only", "final", "移除整个merge组合"),
            ("heads", "final", "移除packed头部选择"),
            ("baseline", "final_local_only", "仅local组合的独立效应"),
            ("baseline", "final_merge_only", "仅merge组合的独立效应"),
        ],
        "阶段交叉包含完整2×2，两kernel计时和workspace协议一致。两种local均输出已排序候选列表，"
        "因此可以接头部归并；不能推广到任意未排序workspace。整个local/merge移除是组合条件效应。"
        "头部归并和packed比较可用公开Triton表达，但r6没有源码回写性能实测。"
        "indices严格一致，value数值精确；±0等价，不声称保留零符号位。",
    )
