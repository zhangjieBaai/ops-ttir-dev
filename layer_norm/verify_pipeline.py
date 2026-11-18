"""Gate TTIR experiments on the tutorial's native Triton backend and launch contract."""

import argparse
import hashlib
import inspect
import json
import os
import tempfile
from pathlib import Path

import torch
import triton
from common import (
    BLOCK,
    EPS,
    OPTIONS,
    ROOT,
    allocate,
    compile_ir,
    launch,
    save_json,
    sha256,
)
from environment import verify_distribution_files
from triton._C import libtriton
from triton._C.libtriton import ir
from triton.backends.compiler import Language
from triton.backends.nvidia.compiler import CUDABackend, get_ptxas
from tutorial import _layer_norm_fwd_fused


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "Isolate an idle GPU first"
    assert not args.output.exists(), "Use a new evidence file"
    # Override/inspection hooks can replace compiler stages or their outputs.
    assert not triton.knobs.compilation.override, "Disable kernel overrides"
    assert triton.knobs.runtime.add_stages_inspection_hook is None
    environment = verify_distribution_files()
    target = triton.runtime.driver.active.get_current_target()
    assert (
        target.backend == "cuda" and target.arch == 90
    ), "Frozen experiment targets SM90"
    backend = CUDABackend(target)
    options = backend.parse_options(OPTIONS)
    assert options.ir_override is None
    stages = {}
    backend.add_stages(stages, options, Language.TRITON)
    downstream = list(stages)[list(stages).index("ttir") + 1 :]
    assert downstream == ["ttgir", "llir", "ptx", "cubin"]
    torch.manual_seed(42)
    ts = allocate(1151, 8192)
    x, y, w, b, mean, rstd = ts[:6]
    source = _layer_norm_fwd_fused[(1151,)](
        x, y, w, b, mean, rstd, x.stride(0), 8192, EPS, BLOCK_SIZE=BLOCK, **OPTIONS
    )
    assert source.asm["ttir"] == (ROOT / "ir/baseline.ttir").read_text()
    reference = [t.clone() for t in [y, mean, rstd]]
    kernels = {name: compile_ir(name) for name in ["baseline", "optimized"]}
    actual_options = [
        {key: ck.metadata._asdict()[key] for key in options.__dict__}
        for ck in [source, *kernels.values()]
    ]
    assert actual_options[0] == actual_options[1] == actual_options[2]
    assert all(ck.metadata.target == source.metadata.target for ck in kernels.values())
    assert all(ck.src.ext == "ttir" for ck in kernels.values())
    baseline = kernels["baseline"]
    assert all(
        source.asm[stage] == baseline.asm[stage] for stage in ["llir", "ptx", "cubin"]
    )
    # MLIR parse/print may rename local SSA induction variables. Compare both
    # TTGIR modules after the same parser round trip instead of comparing names.
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    normalized = []
    for text in [source.asm["ttgir"], baseline.asm["ttgir"]]:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ttgir") as f:
            f.write(text)
            f.flush()
            normalized.append(str(ir.parse_mlir_module(f.name, context)))
    assert normalized[0] == normalized[1]
    for ck in kernels.values():
        assert (
            ck.src.src
            == (
                ROOT / f"ir/{'baseline' if ck is baseline else 'optimized'}.ttir"
            ).read_text()
        )
        launch(ck, ts)
        if ck is baseline:
            for value, ref in zip([y, mean, rstd], reference):
                torch.testing.assert_close(value, ref, rtol=0, atol=0)
        torch.testing.assert_close(
            y, torch.nn.functional.layer_norm(x, (8192,), w, b, EPS), rtol=0, atol=1e-2
        )
    selected = json.loads((ROOT / "results/selection.json").read_text())["kernel"]
    assert (ROOT / "ir/optimized.ttir").read_bytes() == (
        ROOT / f"ir/{selected}.ttir"
    ).read_bytes()
    assert kernels["optimized"].asm["cubin"] == compile_ir(selected).asm["cubin"]
    result = dict(
        passed=True,
        target=str(target),
        options=actual_options[0],
        downstream_stages=downstream,
        input_stage="Exported TTIR after tutorial make_ttir; file compilation resumes at make_ttgir",
        tutorial_vs_baseline_llir_ptx_cubin_byte_identical=True,
        tutorial_vs_baseline_ttgir_parser_normalized_identical=True,
        environment=environment,
        optimized_matches_previously_selected_binary=True,
        shared_launch="common.launch; grid=(M,1,1); same arguments and stream",
        toolchain=dict(
            triton=triton.__version__,
            backend_sha256=sha256(inspect.getfile(CUDABackend)),
            libtriton_sha256=sha256(libtriton.__file__),
            ptxas_sha256=sha256(get_ptxas(90).path),
        ),
        kernels={
            name: dict(
                ttir_sha256=sha256(ROOT / f"ir/{name}.ttir"),
                stage_hashes={
                    stage: hashlib.sha256(
                        value if isinstance(value, bytes) else value.encode()
                    ).hexdigest()
                    for stage, value in ck.asm.items()
                    if stage in downstream
                },
            )
            for name, ck in kernels.items()
        },
    )
    save_json(args.output, result)
    print(
        "PASS: native tutorial backend/options; source-baseline LLIR/PTX/cubin identical; "
        "TTGIR normalized identical; optimized binary unchanged"
    )


if __name__ == "__main__":
    main()
