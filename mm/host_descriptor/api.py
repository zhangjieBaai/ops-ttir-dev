"""Separate selected host ABI entry; unresolved race warnings remain explicitly disclosed."""

# flake8: noqa: E402 -- standalone MM owner bootstrap.

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from bridge import compile_file
from common import digest, runtime_args, sha256
from probe import CONFIG, ROOT, descriptors


def load():
    selection = json.loads((ROOT / "results/performance_v2/selection.json").read_text())
    assert (
        selection["status"] == "PERFORMANCE_SELECTED_UNDER_EXPLICIT_USER_RACE_EXCEPTION"
    )
    assert selection["sync_qualified"] is False
    path = ROOT / selection["selected_alias"]
    registry = ROOT / "results/prototype_inputs_v5.json"
    assert sha256(path) == selection["selected_sha256"] == selection["source_sha256"]
    assert sha256(ROOT / selection["source_path"]) == selection["source_sha256"]
    assert sha256(registry) == selection["registry_sha256"]
    kernel, audit = compile_file(path, CONFIG["options"], registry)
    measurements = ROOT / "results/performance_v2/main.json"
    assert sha256(measurements) == selection["main_sha256"]
    expected = json.loads(measurements.read_text())["resources"][selection["selected"]][
        "stages"
    ]["cubin"]
    assert (
        digest(kernel.asm["cubin"]) == expected
    ), "Selected native binary differs from measured artifact"
    return dict(
        kernel=kernel,
        selection=selection,
        audit=audit,
        sync_qualified=False,
        warning_exception=selection["warning_exception"],
    )


def prepare(a, b, c):
    shape = (a.shape[0], b.shape[1], a.shape[1])
    if not (shape[0] == shape[1] == shape[2] and shape[0] in CONFIG["square_sizes"]):
        raise ValueError("Selected host ABI supports only the six frozen group shapes")
    raw = runtime_args((a, b, c))
    desc = descriptors(a, b, c)
    arguments = (
        *desc,
        a,
        b,
        c,
        raw[3],
        raw[4],
        raw[5],
        a.stride(0),
        b.stride(0),
        c.stride(0),
    )
    grid = ((a.shape[0] + 63) // 64 * ((b.shape[1] + 127) // 128), 1, 1)
    # Descriptor objects retain their tensors. Shapes/strides/addresses must not
    # be mutated while a prepared call or captured graph is in use.
    return grid, arguments


def run_prepared(state, prepared):
    grid, args = prepared
    state["kernel"][grid](*args)


def matmul(a, b, state):
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
    run_prepared(state, prepare(a, b, c))
    return c
