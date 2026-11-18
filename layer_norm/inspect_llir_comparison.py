"""Inspect retained binary occupancy and static SASS without modifying any kernel."""

import ctypes
import importlib.util
import os
import re
import subprocess
from collections import Counter

import torch
from common import ROOT, compile_ir, save_json
from compare_llir import OLD
from finalize import text_section_hashes


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    spec = importlib.util.spec_from_file_location(
        "retained_llir_bench", OLD / "bench_ir.py"
    )
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    kernels = {}
    for label, filename in [
        ("llir_baseline", "baseline.cubin"),
        ("llir_optimized", "opt_cached.cubin"),
    ]:
        launcher = old.make_kernel_from_cubin(str(OLD / filename))
        kernels[label] = next(
            c.cell_contents
            for c in launcher.__closure__
            if hasattr(c.cell_contents, "function")
        )
    kernels["ttir_baseline"] = compile_ir("baseline")
    kernels["ttir_optimized"] = compile_ir("tile_dispatch_math_v2")
    paths = {
        "llir_baseline": OLD / "baseline.cubin",
        "llir_optimized": OLD / "opt_cached.cubin",
        "ttir_baseline": ROOT / "artifacts/baseline/kernel.cubin",
        "ttir_optimized": ROOT / "artifacts/tile_dispatch_math_v2/kernel.cubin",
    }
    driver = ctypes.CDLL("libcuda.so.1")
    occupancy = driver.cuOccupancyMaxActiveBlocksPerMultiprocessor
    occupancy.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
    ]
    result = {}
    out = ROOT / "results/llir_comparison"
    for label, ck in kernels.items():
        blocks = ctypes.c_int()
        status = occupancy(
            ctypes.byref(blocks), ctypes.c_void_p(ck.function), 256, ck.metadata.shared
        )
        assert status == 0, status
        sass = subprocess.check_output(
            ["cuobjdump", "--dump-sass", str(paths[label])], text=True
        )
        (out / f"{label}.sass").write_text(sass)
        ops = re.findall(r"/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9.]*)", sass)
        counts = Counter(ops)
        result[label] = dict(
            n_regs=ck.n_regs,
            n_spills=ck.n_spills,
            dynamic_shared_bytes=ck.metadata.shared,
            cuda_driver_active_blocks_per_sm=blocks.value,
            theoretical_occupancy=blocks.value
            * 256
            / torch.cuda.get_device_properties(0).max_threads_per_multi_processor,
            static_instruction_count=len(ops),
            static_op_counts=dict(counts),
        )
    # These are disjoint uniform-N branches with early EXIT. Count each branch separately.
    # Do not confuse static barrier totals across the entire binary with executed barriers per CTA.
    sas = (out / "ttir_optimized.sass").read_text()
    instructions = [
        (int(m[1], 16), m[2])
        for m in re.finditer(r"/\*([0-9a-f]+)\*/\s+([^\n;]+);", sas)
    ]
    assert text_section_hashes(paths["ttir_optimized"])[
        ".text._layer_norm_fwd_fused"
    ] == (
        "4d31a6eeffb032d90bd3d80b4fc137145f0bf8d22c71ad2373c3cc3a06c3b248"
    ), "Branch address ranges must be re-derived for a changed binary"
    ranges = [
        (0, 0xA50, "N<=1024"),
        (0xA50, 0x1800, "1024<N<=2048"),
        (0x1800, 0x2C50, "2048<N<=4096"),
        (0x2C50, 0x4F20, "4096<N<=8192"),
        (0x4F20, 0x8CE0, "8192<N<=16384"),
        (0x8CE0, 0xBAB0, "N>16384"),
    ]
    result["ttir_optimized"]["branch_static_counts"] = {
        name: dict(
            barriers=sum(
                "BAR.SYNC" in op for addr, op in instructions if start <= addr < end
            ),
            local_loads=sum(
                "LDL." in op for addr, op in instructions if start <= addr < end
            ),
            local_stores=sum(
                op.startswith("STL ")
                for addr, op in instructions
                if start <= addr < end
            ),
        )
        for start, end, name in ranges
    }
    save_json(out / "binary_inspection.json", result)
    print(result)


if __name__ == "__main__":
    main()
