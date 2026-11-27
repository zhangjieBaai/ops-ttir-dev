"""Keep the rejected v1 and give packed vector temporaries unique SSA names."""

import json
import re

from generate_r6 import ROOT, digest, freeze, save


def main():
    manifest = json.loads((ROOT.parent / "round2/expanded/manifest.json").read_text())
    records = []
    for item in manifest["records"]:
        for stage in ("local", "merge"):
            folder = ROOT / "r6/ir" / item["group"] / stage
            text = (folder / "merge_heads_packed.ttir").read_text()
            if stage == "merge":
                pattern = re.compile(
                    r"^    (%h\d+)_reverse =.*?^    %h\d+_packed =[^\n]*\n",
                    re.M | re.S,
                )

                def replace(match):
                    return re.sub(
                        re.escape(match[1]) + r"_(reverse|high|low|shifted|packed)\b",
                        match[1] + r"_next_\1",
                        match[0],
                    )

                text, count = pattern.subn(replace, text)
                assert count == item["shape"][2] - 1
                text = re.sub(r"(%h\d+)_packed\b", r"\1_next_packed", text)
            path = folder / "merge_heads_packed_v2.ttir"
            freeze(path, text)
            records.append(
                {
                    "group": item["group"],
                    "stage": stage,
                    "name": "merge_heads_packed_v2",
                    "sha256": digest(path),
                }
            )
    save(ROOT / "r6/candidates_v2.json", records)
    freeze(
        ROOT / "r6/parser_correction.md",
        "# Packed头部归并v2\n\n"
        "v1的标量解码与向量pack临时值重用了hN_high/low名称，MLIR解析失败。"
        "v2仅将向量pack区域临时SSA名加next前缀；不改变结构或候选数。"
        "v1文件及初筛失败记录保留，最终选择使用包含两种有效候选的完整screen_v2。\n",
    )


if __name__ == "__main__":
    main()
