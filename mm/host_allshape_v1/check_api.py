"""Validate the selected research entry, dispatch, graph replay and rejection."""

# flake8: noqa: E402 -- standalone experiment entry.
import argparse
from pathlib import Path

import torch
from api import load, matmul, prepare_call, run_prepared
from experiment import allocate, idle, save_json, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    idle()
    state = load()
    records = []
    for n in range(256, 4097, 128):
        a, b, c, _ = allocate(n, n, n)
        reference = torch.mm(a, b)
        result = matmul(a, b, state)
        torch.testing.assert_close(result, reference, atol=0.01, rtol=0.001)
        prepared = prepare_call(state, a, b, c)
        run_prepared(state, prepared)
        torch.testing.assert_close(c, reference, atol=0.01, rtol=0.001)
        if n == 1024:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run_prepared(state, prepared)
            c.fill_(float("nan"))
            graph.replay()
            torch.testing.assert_close(c, reference, atol=0.01, rtol=0.001)
        records.append(
            dict(
                n=n,
                family=prepared[0],
                selected=state["selection"]["families"][prepared[0]]["selected"],
                passed=True,
            )
        )
    tensors = allocate(272, 272, 272)
    try:
        prepare_call(state, *tensors[:3])
    except ValueError:
        rejected = True
    else:
        raise AssertionError("Unknown shape accepted")
    torch.cuda.synchronize()
    idle()
    save_json(
        args.output,
        dict(
            records=records,
            graph_replay=True,
            unknown_rejected=rejected,
            sync_qualified=False,
            api_sha256=sha256(Path(__file__).with_name("api.py")),
        ),
    )
    print(
        "PASS: 31 allocating/prepared calls, graph replay and unknown-shape rejection"
    )


if __name__ == "__main__":
    main()
