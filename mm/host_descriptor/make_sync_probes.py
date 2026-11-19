"""Rebuild two correctness-only scheduling probes; neither is a performance finalist."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    source = (ROOT / "ir/envelope_host_v1.ttir").read_text()
    registry = json.loads((ROOT / "results/prototype_inputs.json").read_text())
    for name, attribute in [
        ("host_no_acc_multibuffer", "tt.disallow_acc_multi_buffer"),
        ("host_serial", "tt.num_stages = 1 : i32"),
    ]:
        old = "    }\n    %out"
        assert source.count(old) == 1
        value = source.replace(old, "    } {" + attribute + "}\n    %out")
        path = ROOT / "ir" / f"{name}.ttir"
        if path.exists():
            assert path.read_text() == value, "Never overwrite a named probe"
        else:
            with path.open("x") as f:
                f.write(value)
        registry["inputs"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    output = ROOT / "results/prototype_inputs_v2.json"
    if output.exists():
        assert json.loads(output.read_text()) == registry
    else:
        with output.open("x") as f:
            f.write(json.dumps(registry, indent=2) + "\n")
    print("PASS: synchronization-only probe identities; no kernel launched")


if __name__ == "__main__":
    main()
