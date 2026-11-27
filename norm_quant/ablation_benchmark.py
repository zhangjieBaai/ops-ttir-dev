"""Operator-local pipeline, numerical checks and paired ablation timing."""

import sys
from pathlib import Path

OP = Path(__file__).resolve().parent
sys.path[:0] = [str(OP.parent), str(OP.parent / "complex_suite")]
from complex_suite.ablation_support import run  # noqa: E402

if __name__ == "__main__":
    run(OP)
