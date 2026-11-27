"""Run the unchanged multi-round measurement implementation for round6."""

import argparse

from multiround_runner import run

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--op", required=True, choices=["norm_quant", "partial_topk"])
    p.add_argument("--run", required=True)
    p.add_argument("--names", required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--rep", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--round", type=int, choices=[6], default=6)
    run(p.parse_args())
