"""Load the reviewed six-family selection; preserve frozen shape-to-family dispatch."""

import json

import torch
from common import ROOT, compile_ir, config_for_shape, frozen, launch, sha256


def load_kernels():
    manifest = frozen()
    selection = json.loads((ROOT / "results/round1_selection.json").read_text())
    assert selection["baseline_manifest_sha256"] == sha256(ROOT / "baseline.json")
    kernels = {}
    for family, config in manifest["families"].items():
        chosen = selection["families"][family]
        assert sha256(ROOT / chosen["optimized_path"]) == chosen["optimized_sha256"]
        assert sha256(ROOT / chosen["source_path"]) == chosen["source_sha256"]
        ck = compile_ir(family, "optimized")
        assert ck.src.src == (ROOT / chosen["optimized_path"]).read_text()
        kernels[family] = ck
    return manifest, selection, kernels


def matmul(a, b, state):
    manifest, selection, kernels = state
    family, config = config_for_shape((a.shape[0], b.shape[1], a.shape[1]), manifest)
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float16)
    launch(kernels[family], (a, b, c), config)
    return c
