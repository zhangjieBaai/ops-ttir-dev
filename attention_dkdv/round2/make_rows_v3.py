"""Keep prior failures; separate head and key ranges in row subdivision."""

import hashlib
import json
import re
from pathlib import Path

from make_candidates import immutable

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / "active_manifest.json").read_text())
    immutable(ROOT / "manifest_v2.json", json.dumps(manifest, indent=2) + "\n")
    for g, row in manifest.items():
        new = {}
        for name, oldhash in row["candidates"].items():
            if name == "remat_ptrs_v1":
                new[name] = oldhash
                continue
            assert name.endswith("_v2")
            path = ROOT / "ir" / g / (name + ".ttir")
            text = path.read_text()
            width = 64 if name.startswith("rows64") else 32
            if g.startswith("d64_"):
                text = text.replace(
                    "%offs_n = tt.make_range {end = 128 : i32,",
                    f"%offs_n = tt.make_range {{end = {width} : i32,",
                )
            else:
                text = re.sub(
                    r"(tt.expand_dims %offs_k \{axis = 0 : i32\} : )tensor<\d+xi32>",
                    r"\1tensor<128xi32>",
                    text,
                )
            assert text != path.read_text()
            nextname = name.replace("_v2", "_v3")
            immutable(path.with_name(nextname + ".ttir"), text)
            new[nextname] = hashlib.sha256(text.encode()).hexdigest()
        row["candidates"] = new
    immutable(ROOT / "manifest_v3.json", json.dumps(manifest, indent=2) + "\n")
    (ROOT / "active_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
