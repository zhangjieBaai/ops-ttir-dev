"""Check the retained entry and native allocator behavior during graph capture."""

import argparse
from pathlib import Path

import api
import torch
from common import allocate, save_json, sha256
from engine import (
    CONFIG,
    ROOT,
    NativeScratchAllocator,
    allocation_scope,
    compile_native,
    launch_native,
)
from experiment_round1 import idle
from triton.runtime import _allocation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    idle()
    before = _allocation._allocator.get()
    state = api.load()
    assert state["name"] == "baseline"
    records = []
    for n in CONFIG["square_sizes"]:
        a, b, _, _ = allocate(n, n, n)
        actual = api.matmul(a, b, state)
        torch.testing.assert_close(actual, torch.mm(a, b), atol=0.01, rtol=0.001)
        assert _allocation._allocator.get() is before
        records.append(dict(shape=n, entry="baseline", passed=True))
    a, b, _, _ = allocate(128, 128, 128)
    try:
        api.matmul(a, b, state)
    except ValueError:
        pass
    else:
        raise AssertionError("Unmapped shape must be rejected")
    # This explicitly labeled candidate capture is a numerical/lifetime probe,
    # not public acceptance: its racecheck warning status remains unresolved.
    ck = compile_native("device_tma_v1")
    allocator = NativeScratchAllocator()
    ts = allocate(1152, 1152, 1152)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with allocation_scope(allocator):
        with torch.cuda.graph(graph):
            launch_native(ck, ts)
        graph.replay()
        torch.cuda.synchronize()
    assert _allocation._allocator.get() is before
    torch.testing.assert_close(ts[2], torch.mm(ts[0], ts[1]), atol=0.01, rtol=0.001)
    assert any(x["capturing"] for x in allocator.samples)
    assert all(
        x["requested_stream"] == x["actual_stream"] and x["aligned"]
        for x in allocator.samples
    )
    idle()
    save_json(
        args.output,
        dict(
            passed=True,
            public_entry_records=records,
            unmapped_shape_rejected=True,
            allocator_restored=True,
            candidate_graph_numerical_probe=True,
            candidate_qualified=False,
            allocator_capture_samples=allocator.samples,
            api_sha256=sha256(ROOT / "api.py"),
        ),
    )
    print(
        "PASS: retained baseline6points/refusal; candidate graph stream/alignment/restoration verified; "
        "no TMA qualification granted"
    )


if __name__ == "__main__":
    main()
