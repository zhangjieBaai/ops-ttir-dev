"""Retain parse-failing v1 and repair explicit SCF induction-variable type."""

import json
from pathlib import Path

from make_candidates import immutable

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for g, row in manifest.items():
        new = {}
        for name, oldhash in row["candidates"].items():
            if name == "remat_ptrs_v1":
                new[name] = oldhash
                continue
            path = ROOT / "ir" / g / (name + ".ttir")
            text = path.read_text().replace(
                "step %r2_width {", "step %r2_width : i32 {"
            )
            assert text != path.read_text()
            nextname = name.replace("_v1", "_v2")
            immutable(path.with_name(nextname + ".ttir"), text)
            import hashlib

            new[nextname] = hashlib.sha256(text.encode()).hexdigest()
        row["candidates"] = new
    (ROOT / "active_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
