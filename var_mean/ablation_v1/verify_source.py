"""Compile/run source probes and retain actual TTIR attributes and product hashes."""

import argparse
import importlib
import inspect
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402
import triton.language as tl  # noqa: E402
from common import (  # noqa: E402
    CONTRACT,
    OPTIONS,
    ROOT,
    allocate,
    digest,
    environment,
    hash_data,
    provenance,
    save,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-module", default="source_probes_v3")
    args = parser.parse_args()
    source_module = importlib.import_module(args.source_module)
    source_variant = getattr(source_module, "source_variant", None)
    environment.check_environment()
    torch.manual_seed(42)
    manifest = __import__("json").loads(
        (ROOT / "ablation_v1/manifest.json").read_text()
    )
    cases = [
        (17, n, d)
        for n in [16, 128, 144, 512, 528, 1024, 1040, 4096, 16384]
        for d in ["normal", "constant", "offset", "lowvar", "alternating", "outlier"]
    ]
    records = []
    products = {}
    for bits in manifest["variants"]:
        a, b, c = map(int, bits)
        if hasattr(source_module, "select_variant"):
            source_variant = source_module.select_variant(bool(c))
        for m, n, distribution in cases:
            ts = allocate(m, n, distribution)
            for correction in [0.0, 1.0]:
                ts[3].fill_(float("nan"))
                ts[4].fill_(float("nan"))
                ck = source_variant[(m,)](
                    *ts[:3],
                    m,
                    n,
                    correction,
                    A=bool(a),
                    B=bool(b),
                    C=bool(c),
                    **OPTIONS,
                )
                refs = torch.var_mean(
                    ts[0].cpu().double(), dim=-1, correction=correction
                )
                errors = []
                for output, ref, label in zip(ts[1:3], refs, ["variance", "mean"]):
                    actual = output.cpu().double()
                    torch.testing.assert_close(
                        actual, ref, **CONTRACT["tolerance"][label]
                    )
                    errors.append(float((actual - ref).abs().max()))
                assert all(
                    torch.isnan(s[:16]).all() and torch.isnan(s[-16:]).all()
                    for s in ts[3:]
                )
                records.append(
                    dict(
                        bits=bits,
                        shape=[m, n],
                        distribution=distribution,
                        correction=correction,
                        max_abs_var=errors[0],
                        max_abs_mean=errors[1],
                        passed=True,
                    )
                )
                if bits not in products:
                    folder = ROOT / f"artifacts/ablation_{args.source_module}/{bits}"
                    folder.mkdir(parents=True, exist_ok=True)
                    for ext, data in ck.asm.items():
                        path = folder / f"kernel.{ext}"
                        (
                            path.write_bytes(data)
                            if isinstance(data, bytes)
                            else path.write_text(data)
                        )
                    ck._init_handles()
                    ttir = ck.asm["ttir"]
                    products[bits] = dict(
                        products={s: hash_data(v) for s, v in ck.asm.items()},
                        options=ck.metadata._asdict(),
                        n_regs=ck.n_regs,
                        n_spills=ck.n_spills,
                        divisibility_lines=[
                            line for line in ttir.splitlines() if "divisibility" in line
                        ],
                        int64_cast_lines=[
                            line for line in ttir.splitlines() if "extsi" in line
                        ],
                        n_is_runtime=bool(re.search(r"%N: i32", ttir)),
                        actual_ttir_path=str(folder / "kernel.ttir"),
                    )
    docs = dict(
        multiple_of=inspect.getsource(tl.multiple_of),
        assume=inspect.getsource(tl.assume),
    )
    save(
        args.output,
        dict(
            passed=True,
            provenance=provenance(),
            source_module=args.source_module,
            source_sha256=digest(inspect.getfile(source_module)),
            verifier_sha256=digest(__file__),
            records=records,
            variants=products,
            pinned_local_documentation=docs,
            language_core_sha256=digest(inspect.getfile(tl.multiple_of)),
        ),
    )
    print("PASS source probes", len(records))


if __name__ == "__main__":
    main()
