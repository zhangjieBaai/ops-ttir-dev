"""Reuse the project's upstream-Triton environment guard without duplicating it."""

import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parents[1] / "layer_norm/environment.py"
_spec = importlib.util.spec_from_file_location("ttir_environment_guard", _path)
_guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_guard)
check_environment = _guard.check_environment
verify_distribution_files = _guard.verify_distribution_files
