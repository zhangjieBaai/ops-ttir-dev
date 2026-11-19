"""Freeze two performance candidates; original prototypes and their registries stay immutable."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    source = ROOT / "ir/envelope_host_v1.ttir"
    value = source.read_text()
    old = "%rm = arith.remsi %local, %size"
    assert value.count(old) == 1
    value = value.replace(old, "%rm = arith.remsi %pid, %size")
    path = ROOT / "ir/host_pid_v2.ttir"
    if path.exists():
        assert path.read_text() == value
    else:
        with path.open("x") as f:
            f.write(value)
    registry = json.loads((ROOT / "results/prototype_inputs.json").read_text())
    registry["inputs"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    registry["performance_group"] = dict(
        plan_sha256=hashlib.sha256(
            (ROOT / "optimization_plan_v2.md").read_bytes()
        ).hexdigest(),
        prior_registry_sha256=hashlib.sha256(
            (ROOT / "results/prototype_inputs.json").read_bytes()
        ).hexdigest(),
        user_exception="User explicitly permits unresolved race warnings; no synchronization qualification is granted",
        candidates=["envelope_host_v1", "host_pid_v2"],
        baseline="envelope_baseline",
    )
    target = ROOT / "results/prototype_inputs_v4.json"
    if target.exists():
        assert json.loads(target.read_text()) == registry
    else:
        with target.open("x") as f:
            f.write(json.dumps(registry, indent=2) + "\n")
    print("Frozen two candidate identities and explicit warning exception")


if __name__ == "__main__":
    main()
