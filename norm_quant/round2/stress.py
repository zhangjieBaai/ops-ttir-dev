"""Additional extreme legal-input checks for the round2 quantization rewrite."""

import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
sys.path.insert(0, str(ROOT.parent))
from complex_suite.support import compile_file, digest, environment, save  # noqa: E402


def main():
    spec = importlib.util.spec_from_file_location(
        "original_norm", ROOT.parent / "experiment.py"
    )
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    target = ROOT / "expanded/results/extreme_stress.json"
    assert not target.exists()
    manifest = json.loads((ROOT / "expanded/manifest.json").read_text())
    groups = {}
    for item in manifest["records"]:
        groups.setdefault(item["shape"][1], item)
    result = {"environment": environment(), "records": []}
    for n, item in groups.items():
        for mode in (
            "max_amplitude",
            "zero_weight",
            "min_subnormal",
            "cancellation",
            "spike",
        ):
            torch.manual_seed(713)
            ts, stores = old.allocate(2, n)
            x, r, w = ts[:3]
            if mode == "max_amplitude":
                x.fill_(10000)
                r.fill_(10000)
                w.copy_(torch.linspace(-8, 8, n, device="cuda").half())
            elif mode == "zero_weight":
                w.zero_()
            elif mode == "min_subnormal":
                x.fill_(2**-24)
                r.zero_()
                w.fill_(2**-24)
            elif mode == "cancellation":
                x.fill_(10000)
                r.fill_(-10000)
                w.fill_(8)
            else:
                x.fill_(2**-24)
                r.zero_()
                x[:, 0] = 10000
                w.fill_(8)
            inputs = [t.clone() for t in ts[:3]]
            z, y, scale, q = old.ref(ts, cpu=True)
            for name in ("baseline", "parent", "reciprocal_early"):
                path = ROOT / "expanded/ir" / item["group"] / f"{name}.ttir"
                ck = compile_file(path, item["options"])
                ts[3].fill_(float("nan"))
                ts[4].fill_(-128)
                ts[5].fill_(float("nan"))
                old.launch(ck, ts)
                torch.cuda.synchronize()
                zz, qq, ss = [t.cpu() for t in ts[3:]]
                torch.testing.assert_close(zz, z.float(), rtol=0, atol=0)
                torch.testing.assert_close(ss.double(), scale, rtol=3e-5, atol=1e-9)
                assert bool(((qq >= -127) & (qq <= 127)).all())
                diff = (qq.short() - q.short()).abs()
                assert diff.max().item() <= 1
                err = (qq.double() * ss.double()[:, None] - y).abs()
                limit = (
                    0.501 * scale[:, None]
                    + 3e-5 * y.abs().amax(-1, keepdim=True)
                    + 1e-7
                )
                assert bool((err <= limit).all())
                assert all(torch.equal(a, b) for a, b in zip(inputs, ts[:3]))
                assert bool(
                    torch.isnan(stores[0][-32:]).all()
                    and torch.isnan(stores[2][-32:]).all()
                )
                assert bool((stores[1][-32:] == -128).all())
                result["records"].append(
                    {
                        "N": n,
                        "mode": mode,
                        "name": name,
                        "ir_sha256": digest(path),
                        "max_code_error": diff.max().item(),
                        "reconstruction_max_abs": err.max().item(),
                    }
                )
            save(target, result)
    result["passed"] = True
    save(target, result)
    print("PASS", len(result["records"]), "CPU FP64 extreme checks", flush=True)


if __name__ == "__main__":
    main()
