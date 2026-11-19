"""Extract provenance check and immutable, per-shape two-stage TTIR export."""

import ast
import json

from common import (
    OPTIONS,
    PERF_SIZES,
    ROOT,
    allocate,
    digest,
    gate,
    launch_source,
    save_json,
    sha256,
    write_once,
)


def verify_extraction():
    provenance = json.loads((ROOT / "upstream/provenance.json").read_text())
    for name, expected in provenance["files"].items():
        path = ROOT / name if (ROOT / name).exists() else ROOT / "upstream" / name
        assert sha256(path) == expected
    pairs = [
        ("upstream/sum.py.txt", "tutorial.py", ["sum_kernel_1", "sum_kernel_2"]),
        (
            "upstream/triton_lang_extension.py.txt",
            "frozen_extension.py",
            ["program_id"],
        ),
    ]
    for original, extracted, names in pairs:
        trees = [ast.parse((ROOT / p).read_text()) for p in (original, extracted)]
        for name in names:
            nodes = [
                next(
                    n
                    for n in t.body
                    if isinstance(n, ast.FunctionDef) and n.name == name
                )
                for t in trees
            ]
            for node in nodes:
                node.decorator_list = []
            assert ast.dump(nodes[0]) == ast.dump(nodes[1]), name


def main():
    gate()
    verify_extraction()
    assert not (ROOT / "baseline.json").exists(), "Baseline is immutable"
    groups, shapes = {}, {}
    for n in PERF_SIZES:
        tensors = allocate(n)
        stages = launch_source(tensors)
        ids = []
        for index, ck in enumerate(stages):
            text = ck.asm["ttir"]
            group = f"s{index + 1}_{digest(text)[:16]}"
            path = f"ir/{group}/baseline.ttir"
            write_once(ROOT / path, text)
            runtime = [i for i in range(3) if (i,) not in ck.src.constants]
            config = dict(
                path=path,
                sha256=digest(text),
                options=OPTIONS,
                runtime_indices=runtime,
                signature=ck.src.signature,
                constants={str(k): v for k, v in ck.src.constants.items()},
            )
            assert group not in groups or groups[group] == config
            groups[group] = config
            ids.append(group)
        shapes[str(n)] = ids
        print(f"Export {n}: {ids}", flush=True)
    names = [
        "tutorial.py",
        "frozen_extension.py",
        "upstream/provenance.json",
        "optimization_plan.md",
        "common.py",
    ]
    save_json(
        ROOT / "baseline.json",
        dict(
            experiment="sum_ttir_v1",
            groups=groups,
            shapes=shapes,
            frozen_files={name: sha256(ROOT / name) for name in names},
        ),
    )


if __name__ == "__main__":
    main()
