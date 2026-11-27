"""Compare normalized structure and actual machine instructions, no performance claims."""

import difflib
import re
import subprocess

from common_port import ROOT, digest, save


def main():
    result = {}
    for group in ("d64_c0", "d64_c1", "d128_c0", "d128_c1"):
        entries = {}
        normalized = {}
        for route in ("baseline", "ir", "source"):
            folder = ROOT / "artifacts" / group / route
            sass = subprocess.check_output(
                ["cuobjdump", "--dump-sass", str(folder / "kernel.cubin")], text=True
            )
            (folder / "kernel.sass").write_text(sass)
            instructions = re.findall(
                r"/\*[0-9a-f]+\*/\s+((?:@!?P\d+\s+)?[A-Z][A-Z0-9_.]*)", sass
            )
            entries[route] = dict(
                sass_sha256=digest(sass),
                sass_instruction_count=len(instructions),
                opcode_counts={
                    prefix: sum(prefix in inst for inst in instructions)
                    for prefix in ("HGMMA", "LDL", "STL", "BAR")
                },
            )
            text = (folder / "kernel.ttir").read_text()
            # Raw difference files retained without claiming arbitrary SSA renaming is harmless.
            normalized[route] = text
        entries["source_ir_raw_ttir_equal"] = normalized["source"] == normalized["ir"]
        diff = "".join(
            difflib.unified_diff(
                normalized["ir"].splitlines(keepends=True),
                normalized["source"].splitlines(keepends=True),
                fromfile="direct_ir",
                tofile="source_export",
            )
        )
        path = ROOT / "results" / f"{group}_source_vs_ir.diff"
        path.write_text(diff)
        entries["raw_diff_sha256"] = digest(diff)
        result[group] = entries
    save(ROOT / "results/structural_analysis.json", result)


if __name__ == "__main__":
    main()
