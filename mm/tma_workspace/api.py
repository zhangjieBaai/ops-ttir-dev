"""Explicit research entry for the separate workspace contract; baseline remains default."""

import hashlib
import json

import torch
from engine import (
    CONFIG,
    ROOT,
    NativeScratchAllocator,
    allocation_scope,
    compile_native,
    launch_native,
)


def load():
    # The TMA candidate failed the separate group's performance/racecheck gates.
    # It remains executable only through the explicit experiment runner.
    name = "baseline"
    contract = json.loads((ROOT / "contract.json").read_text())
    selection = json.loads((ROOT / "results/selection.json").read_text())
    assert selection["selected"] == name and not selection["promoted"]
    assert (
        hashlib.sha256((ROOT / "ir" / f"{name}.ttir").read_bytes()).hexdigest()
        == contract["input_sha256"][name]
    )
    return dict(
        name=name, kernel=compile_native(name), allocator=NativeScratchAllocator()
    )


def matmul(a, b, state):
    shape = (a.shape[0], b.shape[1], a.shape[1])
    if not (shape[0] == shape[1] == shape[2] and shape[0] in CONFIG["square_sizes"]):
        raise ValueError("Research entry supports only the six frozen square shapes")
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
    with allocation_scope(state["allocator"]):
        launch_native(state["kernel"], (a, b, c))
    return c
