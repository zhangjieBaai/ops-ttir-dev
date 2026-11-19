"""Direct TTIR prototype inputs; no Python computation kernel is changed."""

# flake8: noqa: E402 -- standalone MM owner bootstrap.

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT as MM
from common import save_json, sha256

ROOT = Path(__file__).resolve().parent
FAMILY = "bm64_bn128_bk64_g8_w4_s4"


def descriptors():
    fields = []
    for name, m, n in [("ad", 64, 64), ("bd", 64, 128), ("cd", 64, 128)]:
        fields += [
            f"%{name}: !tt.tensordesc<tensor<{m}x{n}xf16>>",
            f"%{name}_s0: i32",
            f"%{name}_s1: i32",
            f"%{name}_st0: i64",
            f"%{name}_st1: i64",
        ]
    return ", ".join(fields)


def store(path, value):
    if path.exists():
        assert path.read_text() == value
    else:
        with path.open("x") as f:
            f.write(value)


def main():
    source = MM / "ir" / FAMILY / "baseline.ttir"
    base = source.read_text()
    probe = (
        MM / "results/runs/mm_opt_r2_20261119/device_descriptor_probe.ttir"
    ).read_text()
    # Remove only device descriptor construction. Scalars/offset math and dot order remain.
    lines = [
        line
        for line in probe.splitlines()
        if not any(
            f"{name} =" in line
            for name in ["%one64", "%sa", "%sb", "%sc", "%da", "%db", "%dc"]
        )
    ]
    host = "\n".join(lines) + "\n"
    host = host.replace("%da[", "%ad[").replace("%db[", "%bd[").replace("%dc[", "%cd[")
    header = next(l for l in host.splitlines() if "tt.func public" in l)
    # Prototype with descriptor+rawtensor envelope; rawpointerargs remain for matched baseline.
    extended = header.replace("@matmul_kernel(", f"@matmul_kernel({descriptors()}, ", 1)
    host = host.replace(header, extended, 1)
    # Matched baseline keeps the identical public IR signature; descriptors are unused.
    baseline_header = next(l for l in base.splitlines() if "tt.func public" in l)
    import re

    def no_loc(line):
        line = re.sub(r' loc\("[^"\n]*"\(#loc\w*\)\)', "", line)
        return re.sub(r" loc\(#loc\w*\)", "", line)

    new_header = no_loc(baseline_header).replace(
        "@matmul_kernel(", f"@matmul_kernel({descriptors()}, ", 1
    )
    baseline = base.replace(baseline_header, new_header, 1)
    store(ROOT / "ir/envelope_baseline.ttir", baseline)
    store(ROOT / "ir/envelope_host_v1.ttir", host)
    # Actual known host descriptor export, used solely to validate ABI bridging.
    record = json.loads(
        (MM / "comparison/results/flagtree_main/shape_1152.json").read_text()
    )
    kernel = record["flaggems_kernels"][0]
    fg = (
        MM
        / "comparison/results/flagtree_main/artifacts"
        / kernel["stages"]["cubin"]
        / "kernel.ttir"
    )
    store(ROOT / "ir/fg_host_reference.ttir", fg.read_text())
    save_json(
        ROOT / "results/prototype_inputs.json",
        dict(
            inputs={p.name: sha256(p) for p in (ROOT / "ir").glob("*.ttir")},
            old_baseline_sha256=sha256(source),
            old_probe_sha256=sha256(
                MM / "results/runs/mm_opt_r2_20261119/device_descriptor_probe.ttir"
            ),
        ),
    )


if __name__ == "__main__":
    main()
