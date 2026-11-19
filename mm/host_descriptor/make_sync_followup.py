"""Direct TTIR-only barrier diagnostics for mbarrier initialization/first-use hazards."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    source = (ROOT / "ir/envelope_host_v1.ttir").read_text()
    registry = json.loads((ROOT / "results/prototype_inputs_v2.json").read_text())
    variants = {}
    # GPU dialect barrier is also emitted by Triton's tl.debug_barrier.
    prefix = source.replace(
        "      %ok = arith.muli", "      gpu.barrier\n      %ok = arith.muli", 1
    )
    variants["host_barrier_prefix"] = prefix
    old = "    }\n    %out"
    variants["host_barrier_serial"] = prefix.replace(
        old, "    } {tt.num_stages = 1 : i32}\n    %out", 1
    )
    variants["host_barrier_edges"] = source.replace(
        "      %ok = arith.muli", "      gpu.barrier\n      %ok = arith.muli", 1
    ).replace("      scf.yield %next", "      gpu.barrier\n      scf.yield %next", 1)
    for name, value in variants.items():
        p = ROOT / "ir" / f"{name}.ttir"
        if p.exists():
            assert p.read_text() == value
        else:
            p.write_text(value)
        registry["inputs"][p.name] = hashlib.sha256(value.encode()).hexdigest()
    p = ROOT / "results/prototype_inputs_v3.json"
    if p.exists():
        assert json.loads(p.read_text()) == registry
    else:
        p.write_text(json.dumps(registry, indent=2) + "\n")
    print("Generated three correctness-only TTIR barrier diagnostics; no stageB timing")


if __name__ == "__main__":
    main()
