"""One-time exact extraction; refuses to replace a frozen source."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
source = ROOT / "upstream/06-fused-attention.py"
text = source.read_text()
start = text.index("@triton.jit\ndef _attn_bwd_preprocess")
end = text.index("\n\nclass _attention")
original = text[start:end]
wrapper_start = original.index("@triton.jit\ndef _attn_bwd(")
wrapper = original[wrapper_start : original.index("    # THIS BLOCK DOES DQ:")]
wrapper = wrapper.replace("def _attn_bwd(", "def _attn_bwd_dkdv_only(", 1)
content = (
    '"""Frozen exact official backward functions plus dK/dV-only extraction."""\n'
    "import triton\nimport triton.language as tl\n\n" + original + "\n\n" + wrapper
)
path = ROOT / "frozen_source.py"
assert not path.exists(), "Frozen source already exists"
path.write_text(content)
(ROOT / "upstream/provenance.json").write_text(
    json.dumps(
        dict(
            url="https://raw.githubusercontent.com/triton-lang/triton/v3.7.0/python/tutorials/06-fused-attention.py",
            tag="v3.7.0",
            upstream_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            extracted_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            extraction=(
                "Exact preprocess, dkdv helper, dq helper and full bwd; "
                "additional renamed bwd wrapper truncated before THIS BLOCK DOES DQ. "
                "No dK/dV computational edits."
            ),
        ),
        indent=2,
    )
    + "\n"
)
