"""One verified launch per named TTIR and shape for Nsight Compute collection."""

import argparse

import torch
from common import EPS, allocate, compile_ir, launch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("names", nargs="+")
    p.add_argument("--ns", nargs="+", type=int, default=[1024, 4096, 8704, 15872])
    args = p.parse_args()
    kernels = {name: compile_ir(name) for name in args.names}
    torch.manual_seed(42)
    for n in args.ns:
        ts = allocate(4096, n)
        ts[0].mul_(0.5).add_(-2.3)
        ref = torch.nn.functional.layer_norm(ts[0], (n,), ts[2], ts[3], EPS)
        for name, ck in kernels.items():
            launch(ck, ts)
            error = (ts[1].float() - ref.float()).abs().max().item()
            assert error < 1e-2, (name, n, error)
            print(name, n, error, flush=True)


if __name__ == "__main__":
    main()
