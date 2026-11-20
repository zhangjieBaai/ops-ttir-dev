"""Sanitizer subset covers all eight factor variants and all short-row branches."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402
from common import (  # noqa: E402
    CONTRACT,
    ROOT,
    allocate,
    compile_ir,
    identity,
    launch,
    provenance,
    save,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    manifest = json.loads((ROOT / "ablation_v1/manifest.json").read_text())
    kernels = {bits: compile_ir(v["name"]) for bits, v in manifest["variants"].items()}
    torch.manual_seed(42)
    records = []
    for n in [16, 128, 144, 512, 528, 1024, 1040, 16384]:
        for distribution in ["normal", "offset"]:
            ts = allocate(17, n, distribution)
            for correction in [0.0, 1.0]:
                refs = torch.var_mean(
                    ts[0].cpu().double(), dim=-1, correction=correction
                )
                for bits, ck in kernels.items():
                    ts[3].fill_(float("nan"))
                    ts[4].fill_(float("nan"))
                    launch(ck, ts, correction)
                    for actual, ref, label in zip(ts[1:3], refs, ["variance", "mean"]):
                        torch.testing.assert_close(
                            actual.cpu().double(), ref, **CONTRACT["tolerance"][label]
                        )
                    assert all(
                        torch.isnan(s[:16]).all() and torch.isnan(s[-16:]).all()
                        for s in ts[3:]
                    )
                    records.append(
                        dict(
                            bits=bits,
                            shape=[17, n],
                            distribution=distribution,
                            correction=correction,
                            passed=True,
                        )
                    )
    save(
        args.output,
        dict(
            passed=True,
            provenance=provenance(),
            script_sha256=__import__("hashlib")
            .sha256(Path(__file__).read_bytes())
            .hexdigest(),
            identities={
                bits: identity(manifest["variants"][bits]["name"], ck)
                for bits, ck in kernels.items()
            },
            records=records,
        ),
    )
    print("PASS", len(records), "sanitizer numerical checks")


if __name__ == "__main__":
    main()
