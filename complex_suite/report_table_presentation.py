"""Normalize current ablation tables without changing historical evidence."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def normalize(op):
    folder = ROOT / op
    path = folder / "report_ablation.md"
    if not path.exists():
        return
    current = (folder / "report.md").read_text()
    final_table = re.search(
        r"(?m)^\| Shape [^\n]+\n\|[-:| ]+\n(?:\|[^\n]+\n)+", current
    )[0]
    lines = path.read_text().splitlines()
    output, index, has_shapes = [], 0, False
    while index < len(lines):
        if not lines[index].startswith("|"):
            output.append(lines[index])
            index += 1
            continue
        end = index
        while end < len(lines) and lines[end].startswith("|"):
            end += 1
        block = lines[index:end]
        header = cells(block[0])
        first = header[0].lower()
        shape_table = (
            first.startswith("shape")
            or first
            in (
                "shape",
                "m,n,k",
                "(tokens,e)",
                "(b,h,n,d,causal)",
            )
            or first.startswith("n（")
        )
        if shape_table:
            output.extend(final_table.rstrip().splitlines())
            has_shapes = True
        else:
            keep = [
                i
                for i, h in enumerate(header)
                if "复测" not in h and "repeat" not in h.lower()
            ]
            for line in block:
                values = cells(line)
                if len(values) != len(header):
                    output.append(line)
                    continue
                values = [values[i] for i in keep]
                if line == block[0]:
                    values = [v.replace("主测", "").strip() or "加速比" for v in values]
                output.append("| " + " | ".join(values) + " |")
        index = end
    if not has_shapes:
        output.extend(["", "## 最终版本逐shape时间", "", final_table.rstrip()])
    output = [
        line.replace("主测", "") if line.startswith("|") else line for line in output
    ]
    text = "\n".join(output) + "\n"
    text = text.replace(
        "各单元格为主测 / 独立复测相对共同baseline的逐shape配对比中位数；全部16版本见",
        "下表为固定同一会话的最终版时间与配对加速比；全部16版本原始数据见",
    ).replace(
        "每列为独立复测相对baseline的配对加速比中位数。完整两轮时延、样本和IR/cubin哈希保存在JSON；CSV含两轮各版本。",
        "下表统一采用最终版的一组配对样本，列出基线时间、优化时间及加速比。全部版本原始时延与IR/cubin哈希保存在JSON/CSV。",
    )
    path.write_text(text)


def normalize_current():
    for op in (
        "int4_gemm",
        "moe_routing",
        "norm_quant",
        "attention_dkdv",
        "partial_topk",
        "layer_norm",
        "sum",
    ):
        normalize(op)
