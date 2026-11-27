"""Run the existing full-matrix validator and timer on frozen ablations."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "int4_ablation_protocol", ROOT.parent / "round2/run.py"
)
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)
protocol.ROOT = ROOT
protocol.BASE = ROOT / "frozen"
protocol.NAMES = json.loads((protocol.BASE / "candidates.json").read_text())["names"]

if __name__ == "__main__":
    protocol.main()
