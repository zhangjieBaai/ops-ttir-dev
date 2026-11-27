"""Create immutable final-state controls without modifying existing candidates."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "attention_r6_transform", ROOT / "multiround/r6_generate.py"
)
transform = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transform)


def variants(d, causal):
    group = f"d{d}_c{int(causal)}"
    r2 = ROOT / "round2/ir" / group
    r6 = ROOT / "multiround/r6/ir" / group
    paths = {
        "baseline": r2 / "baseline.ttir",
        "final": r6 / "optimized.ttir",
    }
    if d == 128:
        structure = "rows64_unroll_v5" if causal else "rows64_remat_v3"
        paths.update(
            rows64=r2 / "rows64_v3.ttir",
            structure=r2 / f"{structure}.ttir",
            rows64_dv=ROOT / "ablation/ir" / group / "rows64_dv_v1.ttir",
            dv_only=ROOT / "ir" / group / "r1_dv_late.ttir",
        )
        assert paths["structure"].read_bytes() == (r6 / "parent.ttir").read_bytes()
        assert (
            transform.dv_late(paths["structure"].read_text())
            == paths["final"].read_text()
        )
    else:
        assert (
            paths["final"].read_bytes()
            == (ROOT / "ir" / group / "r1_dv_late.ttir").read_bytes()
        )
    return paths


def main():
    for causal in (False, True):
        group = f"d128_c{int(causal)}"
        source = ROOT / "round2/ir" / group / "rows64_v3.ttir"
        target = ROOT / "ablation/ir" / group / "rows64_dv_v1.ttir"
        transform.immutable(target, transform.dv_late(source.read_text()))
    manifest = {
        f"d{d}_c{int(c)}": {
            name: str(path.relative_to(ROOT)) for name, path in variants(d, c).items()
        }
        for d in (64, 128)
        for c in (False, True)
    }
    transform.immutable(
        ROOT / "ablation/manifest_v2.json", json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
