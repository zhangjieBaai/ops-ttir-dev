"""Render complete per-shape timings, method gains and both final means."""

import sys
from pathlib import Path

OP = Path(__file__).resolve().parent
sys.path.insert(0, str(OP.parent))
from complex_suite.report_operator_details import write_operator  # noqa: E402

if __name__ == "__main__":
    write_operator(OP.name)
