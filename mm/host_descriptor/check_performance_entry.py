"""Exercise the selected host ABI under the recorded warning exception."""

# flake8: noqa: E402 -- standalone MM owner bootstrap.

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import api
import torch
from bridge import compile_file
from common import allocate, dump_kernel, save_json, sha256
from experiment_round1 import idle
from probe import CONFIG, ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists()
    idle()
    state = api.load()
    source = state["selection"]["source_path"]
    original, _ = compile_file(
        ROOT / source, CONFIG["options"], ROOT / "results/prototype_inputs_v4.json"
    )
    assert original.asm["cubin"] == state["kernel"].asm["cubin"]
    records = []
    torch.manual_seed(123)
    for n in CONFIG["square_sizes"]:
        a, b, c, _ = allocate(n, n, n)
        out = api.matmul(a, b, state)
        torch.testing.assert_close(out, torch.mm(a, b), atol=0.01, rtol=0.001)
        prepared = api.prepare(a, b, c)
        api.run_prepared(state, prepared)
        torch.testing.assert_close(c, out, rtol=0, atol=0)
        records.append(dict(shape=n, allocated_and_prepared_entries_passed=True))
    a, b, c, _ = allocate(128, 128, 128)
    try:
        api.prepare(a, b, c)
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown group shape must be rejected")
    a, b, c, _ = allocate(1152, 1152, 1152)
    prepared = api.prepare(a, b, c)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        api.run_prepared(state, prepared)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(c, torch.mm(a, b), atol=0.01, rtol=0.001)
    idle()
    save_json(
        args.output,
        dict(
            passed=True,
            sync_qualified=False,
            warning_exception=state["warning_exception"],
            records=records,
            unknown_shape_refused=True,
            graph_entry_numeric_passed=True,
            selected_alias_cubin_matches_source=True,
            artifacts=dump_kernel(state["kernel"], "host_descriptor/selected_host"),
            api_sha256=sha256(ROOT / "api.py"),
            selection_sha256=sha256(ROOT / "results/performance_v2/selection.json"),
        ),
    )
    print(
        "PASS: selected source/cubin, six allocated+prepared calls, graph replay and unknown shape refusal; race waiver remains"
    )


if __name__ == "__main__":
    main()
