"""Research-only 31-shape host ABI selection; unresolved synchronization warnings remain."""

# flake8: noqa: E402 -- standalone MM experiment bootstrap.
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
from experiment import compile_native, frozen, invoke, prepare, sha256


def load():
    manifest = frozen()
    selection = json.loads((ROOT / "results/selection.json").read_text())
    for name, digest in selection["source_sha256"].items():
        assert sha256(ROOT / "results" / name) == digest
    kernels = {}
    for family, choice in selection["families"].items():
        assert sha256(ROOT / choice["path"]) == choice["sha256"]
        ck, _ = compile_native(family, choice["selected"], manifest["families"][family])
        main = json.loads((ROOT / "results/main.json").read_text())
        expected = main["families"][family]["resources"][choice["selected"]]["stages"][
            "cubin"
        ]
        from common import digest

        assert digest(ck.asm["cubin"]) == expected
        kernels[family] = ck
    return dict(
        manifest=manifest, selection=selection, kernels=kernels, sync_qualified=False
    )


def prepare_call(state, a, b, c):
    shape = (a.shape[0], b.shape[1], a.shape[1])
    if not (
        shape[0] == shape[1] == shape[2]
        and str(shape[0]) in state["manifest"]["shape_to_family"]
    ):
        raise ValueError("Only the 31 frozen square MM shapes are supported")
    family = state["manifest"]["shape_to_family"][str(shape[0])]
    return family, prepare((a, b, c), state["manifest"]["families"][family])


def run_prepared(state, prepared):
    family, args = prepared
    invoke(state["kernels"][family], args)


def matmul(a, b, state):
    c = torch.empty((a.shape[0], b.shape[1]), dtype=a.dtype, device=a.device)
    run_prepared(state, prepare_call(state, a, b, c))
    return c
