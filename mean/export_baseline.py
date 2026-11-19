"""Export every frozen point using exact original heuristic configurations."""

import ast
import json

import triton
from common import (
    ALL_SHAPES,
    PERF_SHAPES,
    ROOT,
    allocate,
    digest,
    gate,
    key,
    launch_source,
    normalized,
    original_config,
    save_json,
    sha256,
)
from triton.backends.nvidia.compiler import CUDABackend


def verify_extraction():
    upstream = ast.parse((ROOT / "upstream/mean.py.txt").read_text())
    original = {n.name: n for n in upstream.body if isinstance(n, ast.FunctionDef)}
    for node in ast.parse((ROOT / "tutorial.py").read_text()).body:
        if isinstance(node, ast.FunctionDef):
            assert ast.dump(ast.Module(body=node.body, type_ignores=[])) == ast.dump(
                ast.Module(body=original[node.name].body, type_ignores=[])
            )
    original = ast.parse((ROOT / "upstream/triton_lang_extension.py.txt").read_text())
    helper = ast.parse((ROOT / "frozen_extension.py").read_text())
    before = next(
        n
        for n in original.body
        if isinstance(n, ast.FunctionDef) and n.name == "program_id"
    )
    after = next(n for n in helper.body if isinstance(n, ast.FunctionDef))
    assert ast.dump(before) == ast.dump(after)
    assert (ROOT / "frozen_heuristics.py").read_bytes() == (
        ROOT / "upstream/heuristics_config_utils.py.txt"
    ).read_bytes()
    provenance = json.loads((ROOT / "upstream/provenance.json").read_text())
    for p, expected in provenance["snapshots"].items():
        assert sha256(ROOT / p) == expected


def main():
    target = gate()
    verify_extraction()
    assert not (ROOT / "baseline.json").exists(), "Baseline already frozen"
    groups, mapping = {}, {}
    for shape in ALL_SHAPES:
        tensors = allocate(shape)
        source = launch_source(shape, tensors)
        meta, options = original_config(shape)
        parsed = normalized(vars(CUDABackend(target).parse_options(options)))
        identity = digest(source.asm["ttir"] + json.dumps(parsed, sort_keys=True))
        group = "g_" + identity[:12]
        if group not in groups:
            path = ROOT / "ir" / group / "baseline.ttir"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x") as stream:
                stream.write(source.asm["ttir"])
            runtime_indices = [
                i
                for i, (_, v) in enumerate(source.src.signature.items())
                if v != "constexpr"
            ]
            groups[group] = dict(
                ttir_path=str(path.relative_to(ROOT)),
                ttir_sha256=sha256(path),
                options=options,
                parsed_options=parsed,
                meta=meta,
                runtime_indices=runtime_indices,
                source_signature=source.src.signature,
                representative=shape,
            )
        assert (ROOT / groups[group]["ttir_path"]).read_text() == source.asm["ttir"]
        mapping[key(shape)] = dict(
            group=group,
            shape=shape,
            dtype="float16",
            layout="contiguous",
            grid=[
                shape[0],
                (
                    1
                    if shape[2] == 1
                    else triton.cdiv(
                        shape[2],
                        meta.get(
                            "TILE_K",
                            meta.get("BLOCK_SIZE_K", 1) * meta.get("VEC_SIZE", 1),
                        ),
                    )
                ),
                1,
            ],
            performance=shape in PERF_SHAPES,
            partition=(
                "validation"
                if (
                    shape[2] == 1 and shape[0] == 64 or shape[2] != 1 and shape[0] == 16
                )
                else "tuning"
            ),
            source_cubin_sha256=digest(source.asm["cubin"]),
        )
        print(key(shape), group, meta, flush=True)
    frozen = [
        "tutorial.py",
        "frozen_extension.py",
        "frozen_heuristics.py",
        "upstream/provenance.json",
        "optimization_plan.md",
        *[str(p.relative_to(ROOT)) for p in (ROOT / "upstream").glob("*.txt")],
        *[g["ttir_path"] for g in groups.values()],
    ]
    save_json(
        ROOT / "baseline.json",
        dict(
            experiment="mean_ttir_v1",
            source="FlagGems mean.py",
            groups=groups,
            mapping=mapping,
            frozen_files={p: sha256(ROOT / p) for p in frozen},
        ),
    )
    print(f"FROZEN {len(mapping)} points, {len(groups)} groups")


if __name__ == "__main__":
    main()
