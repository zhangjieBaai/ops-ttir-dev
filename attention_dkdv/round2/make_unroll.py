"""Sixth candidate: replace fixed two-subtile outer loop with explicit SSA copies."""

import hashlib
import json
import re
from pathlib import Path

from make_candidates import immutable

ROOT = Path(__file__).resolve().parent


def unroll(text):
    begin = text.index("    scf.for %r2_slice =")
    body_start = text.index("\n", begin) + 1
    end = text.index("    }\n    tt.return", body_start)
    body = text[body_start:end]
    defined = set(re.findall(r"(%[A-Za-z_0-9]+)(?::\d+)?\s*=", body))
    output = []
    for index, offset in enumerate((0, 64)):
        output.append(f"    %r2_fixed_{index} = arith.constant {offset} : i32\n")

        def replace(m):
            old = m[0]
            if old == "%r2_slice":
                return f"%r2_fixed_{index}"
            if old in defined:
                return old + f"_sub{index}"
            return old

        copy = re.sub(r"%[A-Za-z_0-9]+", replace, body)
        output.append(
            "\n".join(
                line[2:] if line.startswith("  ") else line
                for line in copy.splitlines()
            )
            + "\n"
        )
    return text[:begin] + "".join(output) + text[end + 6 :]


def main():
    manifest = json.loads((ROOT / "active_manifest.json").read_text())
    selection = {"tested_candidate": {}}
    for g, row in manifest.items():
        path = ROOT / "ir" / g / "rows64_unroll_v4.ttir"
        text = unroll((path.parent / "rows64_v3.ttir").read_text())
        immutable(path, text)
        row["candidates"]["rows64_unroll_v4"] = hashlib.sha256(
            text.encode()
        ).hexdigest()
        selection["tested_candidate"][g] = "rows64_unroll_v4"
    immutable(ROOT / "manifest_v4.json", json.dumps(manifest, indent=2) + "\n")
    (ROOT / "active_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    immutable(ROOT / "unroll_only.json", json.dumps(selection, indent=2) + "\n")


if __name__ == "__main__":
    main()
