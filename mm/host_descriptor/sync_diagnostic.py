"""Small, explicitly separated reproductions of the TMA barrier hazard."""

# flake8: noqa: E402 -- diagnostic bootstrap; FlagTree mode never imports common.gate.

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import torch
import triton
from source_controls import native_host_control
from triton.tools.tensor_descriptor import TensorDescriptor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["tiny-upstream", "flagtree-control"])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    visible = os.environ["CUDA_VISIBLE_DEVICES"]
    assert len(visible.split(",")) == 1, "Isolate one idle GPU"
    occupants = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            visible,
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader",
        ],
        text=True,
    )
    assert not [
        line
        for line in occupants.splitlines()
        if int(line.split(",", 1)[0]) != os.getpid()
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    if args.mode == "tiny-upstream":
        from bridge import compile_file
        from common import allocate
        from probe import CONFIG, launch

        ck, _ = compile_file(HERE / "ir/envelope_host_v1.ttir", CONFIG["options"])
        ts = allocate(64, 128, 64)
        launch(ck, ts, "envelope_host_v1")
        torch.cuda.synchronize()
        torch.testing.assert_close(ts[2], torch.mm(ts[0], ts[1]), atol=0.01, rtol=0.001)
        rows.append(dict(shape=[64, 128, 64], numerical_pass=True))
    else:
        assert importlib.metadata.version("flagtree") == "0.7.0"
        for n in [1152, 1280]:
            torch.manual_seed(73 + n)
            a = torch.randn((n, n), device="cuda", dtype=torch.float16)
            b = torch.randn_like(a)
            c = torch.empty_like(a)
            descriptors = [
                TensorDescriptor(x, list(x.shape), list(x.stride()), block)
                for x, block in [(a, [64, 64]), (b, [64, 128]), (c, [64, 128])]
            ]
            ck = native_host_control[((n + 63) // 64 * ((n + 127) // 128),)](
                *descriptors,
                n,
                n,
                n,
                BM=64,
                BN=128,
                BK=64,
                GM=8,
                num_warps=4,
                num_stages=4,
                num_ctas=1,
                enable_fp_fusion=True,
            )
            torch.testing.assert_close(c, torch.mm(a, b), atol=0.01, rtol=0.001)
            rows.append(dict(shape=[n, n, n], numerical_pass=True))
    # Selected native artifact, no edits and no replacement.
    for ext in ["ptx", "cubin"]:
        value = ck.asm[ext]
        (args.output / f"kernel.{ext}").write_bytes(
            value if isinstance(value, bytes) else value.encode()
        )
    result = dict(
        mode=args.mode,
        records=rows,
        triton=triton.__version__,
        triton_path=triton.__file__,
        torch=torch.__version__,
        source_sha256=hashlib.sha256(
            (HERE / "source_controls.py").read_bytes()
        ).hexdigest(),
        ptx_sha256=hashlib.sha256(ck.asm["ptx"].encode()).hexdigest(),
        cubin_sha256=hashlib.sha256(ck.asm["cubin"]).hexdigest(),
        claim="Diagnostic only: inspect external sanitizer status separately; no timing or primary-env qualification",
    )
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
