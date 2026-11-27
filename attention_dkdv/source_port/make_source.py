"""Create the faithful source port once, before formatting and freezing."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
original = (ROOT.parent / "frozen_source.py").read_text()
start = original.index("@triton.jit\ndef _attn_bwd_dkdv(")
end = original.index("# the main inner-loop logic for computing dQ")
helper = original[start:end]
start = original.index("@triton.jit\ndef _attn_bwd_dkdv_only(")
wrapper = original[start:]
old = "        # Compute dV.\n        ppT = pT\n        ppT = ppT.to(tl.float16)\n        dv += tl.dot(ppT, do)\n"
assert helper.count(old) == 1
helper = helper.replace(old, "")
anchor = "        dk += tl.dot(dsT, tl.trans(qT))\n"
assert helper.count(anchor) == 1
helper = helper.replace(anchor, anchor + old)
path = ROOT / "frozen_port.py"
assert not path.exists(), "Source port already exists; do not overwrite frozen source."
path.write_text(
    '"""Faithful r1_dv_late source port: dV cast/dot after dK, no retuning."""\n\n'
    "import triton\nimport triton.language as tl\n\n" + helper + "\n" + wrapper
)
