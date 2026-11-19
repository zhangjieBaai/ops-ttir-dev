"""Profile one steady graph node using the frozen native-TTIR kernel path."""

import argparse
import json
from pathlib import Path

import torch
from common import (
    ROOT,
    allocate,
    compile_ir,
    kernel_info,
    launch,
    receipt,
    save_json,
    sha256,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    ck = compile_ir(args.name)
    previous = json.loads((ROOT / "results/runs/r1_20261119/accuracy.json").read_text())
    assert previous["passed"]
    assert previous["kernels"][args.name] == sha256(ROOT / f"ir/{args.name}.ttir")
    torch.manual_seed(941)
    x, y, out, _ = allocate(args.n)
    reference = x + y
    out.fill_(float("nan"))
    launch(ck, x, y, out)
    assert torch.equal(out.view(torch.int32), reference.view(torch.int32))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(20):
            launch(ck, x, y, out)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(32):
            launch(ck, x, y, out)
    # Repeated buffers, as in the primary metric. NCU's cache-control remains
    # an explicit independent diagnostic factor selected by the parent command.
    for _ in range(8):
        graph.replay()
    torch.cuda.synchronize()
    before = receipt()
    torch.cuda.cudart().cudaProfilerStart()
    graph.replay()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    assert torch.equal(out.view(torch.int32), reference.view(torch.int32))
    props = torch.cuda.get_device_properties(0)
    save_json(
        args.output,
        dict(
            **before,
            name=args.name,
            n=args.n,
            dtype="float32",
            layout="contiguous, 16-byte aligned, no aliases",
            ttir_path=f"ir/{args.name}.ttir",
            ttir_sha256=sha256(ROOT / f"ir/{args.name}.ttir"),
            artifact=kernel_info(ck, args.name),
            correctness_before_after=True,
            correctness_receipt_sha256=sha256(
                ROOT / "results/runs/r1_20261119/accuracy.json"
            ),
            graph_nodes=32,
            graph_warmup_replays=8,
            profiled_node="17th kernel in final graph (NCU launch-skip=16, count=1)",
            sm_count=props.multi_processor_count,
            device_properties=str(props),
            gpu_snapshot_after=receipt()["gpu_snapshot"],
        ),
    )


if __name__ == "__main__":
    main()
