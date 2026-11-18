"""Audit why identical tutorial LLVM computations produced different baseline binaries."""

import argparse
import importlib.util
import re
import subprocess
from collections import Counter
from pathlib import Path

import triton
from common import OPTIONS, ROOT, save_json, sha256
from finalize import text_section_hashes
from triton._C.libtriton import llvm, nvidia
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend

OLD = Path("/root/ir-ops-dev/layer_norm")
LLC = Path("/usr/local/lib/python3.12/dist-packages/mlir/llvm_artifact/bin/llc")
PTXAS = Path(triton.__file__).parent / "backends/nvidia/bin/ptxas"


def code_before_metadata(text):
    return text[: text.index("\n!llvm.")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    old_ir = OLD / "ir_opt/baseline.llir"
    current_ir = ROOT / "artifacts/baseline/kernel.llir"
    assert code_before_metadata(old_ir.read_text()) == code_before_metadata(
        current_ir.read_text()
    )
    spec = importlib.util.spec_from_file_location(
        "legacy_compile", OLD / "ir_opt/compile_llir.py"
    )
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    paths = {
        "legacy_retained": OLD / "ir_opt/baseline.cubin",
        "old_tutorial_native": OLD / "compiled/_layer_norm_fwd_fused.cubin",
        "current_native": ROOT / "artifacts/baseline/kernel.cubin",
    }
    for label, path in [
        ("old_ir_legacy_backend", old_ir),
        ("current_ir_legacy_backend", current_ir),
    ]:
        dest = out / f"{label}.cubin"
        legacy.compile_llir(str(path), str(dest))
        paths[label] = dest
    llvm.init_targets()
    nvidia.set_short_ptr()
    backend = CUDABackend(GPUTarget("cuda", 90, 32))
    options = backend.parse_options(OPTIONS)
    metadata = {}
    ptx_text = backend.make_ptx(old_ir.read_text(), metadata, options, 90)
    (out / "old_ir_native_backend.ptx").write_text(ptx_text)
    dest = out / "old_ir_native_backend.cubin"
    dest.write_bytes(backend.make_cubin(ptx_text, metadata, options, 90))
    paths["old_ir_native_backend"] = dest
    # Isolate aggressive FP contraction within the old llc route; other commands stay unchanged.
    source = out / "llc_input.llir"
    source.write_text(old_ir.read_text().replace(" nocreateundeforpoison", ""))
    for label, flags in [
        ("llc_default", []),
        ("llc_contract_fast", ["-fp-contract=fast"]),
    ]:
        ptx = out / f"{label}.ptx"
        command = [
            str(LLC),
            "-march=nvptx64",
            "-mcpu=sm_90",
            *flags,
            "-o",
            str(ptx),
            str(source),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        ptx.write_text(ptx.read_text().replace(".target sm_90, debug", ".target sm_90"))
        cubin = out / f"{label}.cubin"
        subprocess.run(
            [str(PTXAS), "-O3", "-arch=sm_90", "-o", str(cubin), str(ptx)],
            check=True,
            capture_output=True,
            text=True,
        )
        paths[label] = cubin
    source.unlink()
    result = dict(
        executable_llir_identical=True,
        triton_version=triton.__version__,
        llc_version=subprocess.check_output([str(LLC), "--version"], text=True),
        inputs={"legacy_llir": sha256(old_ir), "current_llir": sha256(current_ir)},
        binaries={},
    )
    for label, path in paths.items():
        sass = subprocess.check_output(
            ["cuobjdump", "--dump-sass", str(path)], text=True
        )
        ops = re.findall(r"/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9.]*)", sass)
        counts = Counter(ops)
        result["binaries"][label] = dict(
            path=str(path),
            cubin_sha256=sha256(path),
            text=text_section_hashes(path),
            instructions=len(ops),
            op_counts={op: counts[op] for op in ["FADD", "FMUL", "FFMA"]},
        )
    b = result["binaries"]
    assert b["legacy_retained"]["text"] == b["old_ir_legacy_backend"]["text"]
    assert b["legacy_retained"]["text"] == b["current_ir_legacy_backend"]["text"]
    assert b["legacy_retained"]["text"] == b["llc_default"]["text"]
    assert b["current_native"]["text"] == b["old_ir_native_backend"]["text"]
    assert b["current_native"]["text"] == b["old_tutorial_native"]["text"]
    save_json(out / "audit.json", result)
    print(result)


if __name__ == "__main__":
    main()
