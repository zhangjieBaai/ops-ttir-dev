"""Static evidence only; instruction counts are not execution-time attribution."""

import json
import re
import subprocess
from pathlib import Path

from common import ROOT, digest, save


def main():
    result = {}
    for group in ("d64_c0", "d64_c1", "d128_c0", "d128_c1"):
        result[group] = {}
        for name in ("baseline", "r1_dv_late"):
            folder = ROOT / "artifacts" / group / name
            metadata = json.loads((folder / "metadata.json").read_text())
            ptx = (folder / "kernel.ptx").read_text()
            sass = subprocess.check_output(
                ["cuobjdump", "--dump-sass", str(folder / "kernel.cubin")], text=True
            )
            (folder / "kernel.sass").write_text(sass)
            instructions = re.findall(
                r"/\*[0-9a-f]+\*/\s+((?:@!?P\d+\s+)?[A-Z][A-Z0-9_.]*)", sass
            )
            counts = {
                prefix: sum(prefix in instruction for instruction in instructions)
                for prefix in ("HGMMA", "LDL", "STL", "BAR", "SHFL")
            }
            result[group][name] = dict(
                n_regs=metadata["n_regs"],
                n_spills=metadata["n_spills"],
                shared_bytes=metadata["shared"],
                cubin_sha256=metadata["hashes"]["cubin"],
                sass_sha256=digest(sass),
                sass_static_instruction_count=len(instructions),
                sass_counts=counts,
                ptx_bar_sync=ptx.count("bar.sync"),
                ptx_wgmma_mentions=ptx.count("wgmma"),
            )
    save(ROOT / "results/static_artifacts.json", result)
    import triton.backends.nvidia.compiler as compiler
    import triton.compiler.compiler as core
    from triton._C import libtriton

    ptxas = Path(compiler.get_ptxas(90).path)
    save(
        ROOT / "results/toolchain.json",
        dict(
            files={
                str(p): digest(p)
                for p in (
                    Path(libtriton.__file__),
                    Path(compiler.__file__),
                    Path(core.__file__),
                    ptxas,
                )
            },
            ptxas_version=subprocess.check_output([str(ptxas), "--version"], text=True),
            sanitizer_version=subprocess.check_output(
                ["compute-sanitizer", "--version"], text=True
            ),
        ),
    )


if __name__ == "__main__":
    main()
