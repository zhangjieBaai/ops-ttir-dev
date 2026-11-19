"""One hash-gated launch per selected N/kernel for diagnostic NCU collection."""

import argparse
import json
import os

import torch
from common import ROOT, allocate, block_for, compile_ir, digest, launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    validation = json.load(open(args.validation))
    assert validation["passed"]
    names = [
        "baseline",
        "r3_parent",
        "r3_centered_stream" if args.stream else "r3_group_center2048",
    ]
    kernels = {}
    for n in [1024, 2048] if args.stream else [8192, 8704, 15872]:
        block = block_for(n)
        ts = allocate(4096, n)
        for name in names:
            key = (name, block)
            assert validation["kernels"][f"{name}/b{block}"] == digest(
                ROOT / f"ir/b{block}/{name}.ttir"
            )
            if key not in kernels:
                kernels[key] = compile_ir(name, block)
            print("PROFILE", name, n, flush=True)
            launch(kernels[key], ts)
            torch.cuda.synchronize()


if __name__ == "__main__":
    main()
