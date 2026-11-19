"""Extract only the original dX statements; do not count removal of dW/dB as a speedup."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "upstream/05-layer-norm.py.txt"


def main():
    text = SOURCE.read_text()
    begin = text.index("@triton.jit\ndef _layer_norm_bwd_dx_fused(")
    end = text.index("\n\n@triton.jit\ndef _layer_norm_bwd_dwdb", begin)
    original = text[begin:end]
    (ROOT / "upstream/fused_backward.py.txt").write_text(original + "\n")
    body1 = original[
        original.index("    # Map the program id") : original.index(
            "    # Offset locks"
        )
    ]
    body2 = original[
        original.index("    # Load data to SRAM") : original.index(
            "    # Accumulate partial sums"
        )
    ]
    code = (
        '"""Tutorial dX computation only. Full fused dW/dB/lock work is excluded from both versions."""\n\n'
        "import triton\nimport triton.language as tl\n\n\n@triton.jit\n"
        "def layer_norm_bwd_dx(DX, DY, X, W, Mean, Rstd, stride, N, BLOCK_SIZE_N: tl.constexpr):\n"
        + body1
        + body2
    )
    path = ROOT / "tutorial.py"
    if path.exists():
        assert path.read_text() == code, "Frozen extracted source changed"
    else:
        path.write_text(code)
    provenance = dict(
        source=str(SOURCE),
        source_sha256=hashlib.sha256(text.encode()).hexdigest(),
        upstream_url="https://github.com/triton-lang/triton/blob/v3.7.0/python/tutorials/05-layer-norm.py",
        original_kernel="_layer_norm_bwd_dx_fused",
        extracted_kernel="layer_norm_bwd_dx",
        retained_body_sha256=hashlib.sha256((body1 + body2).encode()).hexdigest(),
        excluded="DW/DB/Lock parameters and their address setup; all partial gradient accumulation and locks",
        classification="Tutorial dX section extraction; not unchanged full fused backward",
    )
    (ROOT / "upstream/provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
