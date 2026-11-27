"""Export original frozen source for the predeclared 30-point fixed-family matrix."""

import importlib.util
import sys
from pathlib import Path

import torch
import triton
from triton.backends.nvidia.compiler import CUDABackend

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import common as c  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "old_transform", ROOT.parent / "make_candidates.py"
)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
SHAPES = c.SHAPES + [
    (1, 64, 128),
    (1, 1024, 1024),
    (1, 4096, 4096),
    (2, 256, 256),
    (4, 1024, 1024),
    (8, 4096, 4096),
    (16, 256, 128),
    (16, 1024, 1024),
    (16, 11008, 4096),
    (17, 64, 128),
    (17, 272, 256),
    (31, 1008, 1024),
    (32, 64, 128),
    (32, 4096, 4096),
    (33, 1040, 1152),
    (64, 256, 512),
    (64, 1024, 2048),
    (64, 11008, 4096),
    (127, 4080, 4096),
    (128, 256, 128),
    (128, 1024, 4096),
    (256, 4096, 4096),
    (64, 2048, 8192),
]


def main():
    assert len(SHAPES) == len(set(SHAPES)) == 30
    path = ROOT / "expanded/manifest.json"
    assert not path.exists()
    env = c.environment()
    originals = {tuple(row["shape"]): row for row in c.manifest()["cases"]}
    cases = []
    for shape in SHAPES:
        config = (
            originals[shape]["config"]
            if shape in originals
            else [16 if shape[0] <= 16 else 32, 64, 64, 4, 3]
        )
        ts = c.allocate(shape, seed=42)
        source = c.source_launch(ts, config)
        folder = ROOT / "expanded/ir" / c.case_id(shape)
        c.write_once(folder / "baseline.ttir", source.asm["ttir"])
        if shape in originals:
            assert (
                source.asm["ttir"]
                == (c.ROOT / "ir" / c.case_id(shape) / "baseline.ttir").read_text()
            )
        parent = old.unsigned(old.scalar(source.asm["ttir"]))
        c.write_once(folder / "parent.ttir", parent)
        baseline = triton.compile(
            str(folder / "baseline.ttir"), options=c.options(config)
        )
        baseline._init_handles()
        assert all(source.asm[s] == baseline.asm[s] for s in ["llir", "ptx", "cubin"])
        keys = CUDABackend(source.metadata.target).parse_options({}).__dict__
        assert {k: getattr(source.metadata, k) for k in keys} == {
            k: getattr(baseline.metadata, k) for k in keys
        }
        torch.backends.cuda.matmul.allow_tf32 = False
        c.assert_output(ts, c.reference(ts))
        cases.append(
            {
                "id": c.case_id(shape),
                "shape": shape,
                "config": config,
                "config_origin": (
                    "original seven frozen autotune"
                    if shape in originals
                    else "new predeclared fixed family"
                ),
                "baseline_sha256": c.sha(folder / "baseline.ttir"),
                "parent_sha256": c.sha(folder / "parent.ttir"),
                "pipeline": {
                    "source_file_llir_ptx_cubin_identical": True,
                    "full_options_identical": True,
                },
            }
        )
        print(c.case_id(shape), config, "original pipeline PASS", flush=True)
    c.save(
        path,
        {
            "environment": env,
            "source_sha256": c.sha(ROOT.parent / "tutorial.py"),
            "matrix_plan_sha256": c.sha(ROOT / "expanded_plan.md"),
            "cases": cases,
        },
    )


if __name__ == "__main__":
    main()
