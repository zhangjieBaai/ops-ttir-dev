"""Repair numeric SSA names in explicit unrolling without changing v4 artifacts."""

import hashlib
import json
import re
from pathlib import Path

from make_candidates import immutable

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / "manifest_v4.json").read_text())
    selection = {"tested_candidate": {}}
    for g, row in manifest.items():
        old = ROOT / "ir" / g / "rows64_unroll_v4.ttir"
        text = re.sub(r"%(\d+)_sub", r"%r2_num_\1_sub", old.read_text())
        assert text != old.read_text()
        name = "rows64_unroll_v5"
        immutable(old.with_name(name + ".ttir"), text)
        del row["candidates"]["rows64_unroll_v4"]
        row["candidates"][name] = hashlib.sha256(text.encode()).hexdigest()
        selection["tested_candidate"][g] = name
    immutable(ROOT / "manifest_v5.json", json.dumps(manifest, indent=2) + "\n")
    (ROOT / "active_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    immutable(ROOT / "unroll_v5_only.json", json.dumps(selection, indent=2) + "\n")


if __name__ == "__main__":
    main()
