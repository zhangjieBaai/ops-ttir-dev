"""C1/hash-gated NCU launches for resource and traffic diagnosis."""

import argparse
import json
import os

import torch
from common import ROOT, allocate, block_for, compile_ir, digest, launch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validation", required=True)
    args = p.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    check = json.load(open(args.validation))
    assert check["passed"]
    kernels = {}
    for n in [3072, 8192, 15872]:
        b = block_for(n)
        ts = allocate(4096, n)
        for name in ["baseline", "r4_parent", "r4_evict", "r5_rows8_tile256"]:
            assert check["kernels"][f"{name}/b{b}"] == digest(
                ROOT / f"ir/b{b}/{name}.ttir"
            )
            if (name, b) not in kernels:
                kernels[name, b] = compile_ir(name, b)
            print("PROFILE", n, name, flush=True)
            launch(kernels[name, b], ts)
            torch.cuda.synchronize()


if __name__ == "__main__":
    main()
