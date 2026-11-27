"""Freeze the final Norm+INT8 dataflow chain and controlled removals."""

import importlib.util
import json
import sys
from pathlib import Path

OP = Path(__file__).resolve().parent
sys.path[:0] = [str(OP.parent), str(OP.parent / "complex_suite"), str(OP)]
from make_candidates import candidates  # noqa: E402

from complex_suite.support import digest, freeze  # noqa: E402


def main():
    spec = importlib.util.spec_from_file_location(
        "round2_candidates", OP / "round2/make_candidates.py"
    )
    r2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r2)
    spec = importlib.util.spec_from_file_location(
        "multiround_candidates", OP / "multiround/generate.py"
    )
    mr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mr)
    original = json.loads((OP / "round2/expanded/manifest.json").read_text())
    rows = []
    for item in original["records"]:
        base_path = OP / "round2/expanded/ir" / item["group"] / "baseline.ttir"
        first = candidates(base_path.read_text())
        stream = first["stream_v1"]
        final = r2.early_z(r2.reciprocal(stream))
        selected = OP / "multiround/r6/ir" / item["group"] / "optimized.ttir"
        assert final == selected.read_text(), item["group"]
        versions = dict(
            baseline=base_path.read_text(),
            factor=first["factor_v1"],
            joint=first["joint_v1"],
            stream=stream,
            reciprocal=r2.reciprocal(stream),
            early_z=r2.early_z(stream),
            final=final,
            final_no_cs=final.replace(" cacheModifier = cs", ""),
            final_separate=mr.statistics_form(final, "separate_float"),
        )
        paths, hashes = {}, {}
        for name, text in versions.items():
            path = OP / "ablation_v1/ir" / item["group"] / f"{name}.ttir"
            freeze(path, text)
            paths[name] = dict(kernel=str(path.relative_to(OP.parent)))
            hashes[name] = dict(kernel=digest(path))
        rows.append(
            dict(
                shape=item["shape"],
                group=item["group"],
                options=item["options"],
                paths=paths,
                sha256=hashes,
                selected_sha256=digest(selected),
            )
        )
    manifest = dict(names=list(versions), records=rows)
    freeze(OP / "ablation_v1/manifest.json", json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
