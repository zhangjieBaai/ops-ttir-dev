"""Freeze algorithm additions and the complete final local/merge factorial."""

import json
import sys
from pathlib import Path

OP = Path(__file__).resolve().parent
sys.path[:0] = [str(OP.parent), str(OP.parent / "complex_suite"), str(OP)]
from make_ordered import ordered  # noqa: E402

from complex_suite.support import digest, freeze  # noqa: E402


def main():
    original = json.loads((OP / "round2/expanded/manifest.json").read_text())
    names = [
        "baseline",
        "ordered_recompute",
        "carry",
        "static_ids",
        "static_unroll",
        "parent",
        "heads",
        "final_local_only",
        "final_merge_only",
        "final",
    ]
    rows = []
    for item in original["records"]:
        paths, hashes = {n: {} for n in names}, {n: {} for n in names}
        for stage in ("local", "merge"):
            expanded = OP / "round2/expanded/ir" / item["group"] / stage
            latest = OP / "multiround/r6/ir" / item["group"] / stage
            base = (expanded / "baseline.ttir").read_text()
            selected = (latest / "optimized.ttir").read_text()
            versions = dict(
                baseline=base,
                ordered_recompute=ordered(base),
                carry=(expanded / "parent.ttir").read_text(),
                static_ids=(expanded / "static_ids.ttir").read_text(),
                static_unroll=(expanded / "static_unroll_v2.ttir").read_text(),
                parent=(latest / "parent.ttir").read_text(),
                heads=(latest / "merge_heads.ttir").read_text(),
                final_local_only=selected if stage == "local" else base,
                final_merge_only=selected if stage == "merge" else base,
                final=selected,
            )
            for name in names:
                path = OP / "ablation_v1/ir" / item["group"] / stage / f"{name}.ttir"
                freeze(path, versions[name])
                paths[name][stage] = str(path.relative_to(OP.parent))
                hashes[name][stage] = digest(path)
        rows.append(
            dict(
                shape=item["shape"],
                group=item["group"],
                options=item["options"],
                paths=paths,
                sha256=hashes,
            )
        )
    freeze(
        OP / "ablation_v1/manifest.json",
        json.dumps(dict(names=names, records=rows), indent=2) + "\n",
    )


if __name__ == "__main__":
    main()
