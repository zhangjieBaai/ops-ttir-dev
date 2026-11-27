"""Shared native three-way launcher, fingerprints and immutable source checks."""

import json
import struct
import sys
from pathlib import Path

import triton

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
sys.path.insert(0, str(PARENT))
from common import (  # noqa: E402
    MATRIX,
    OPTIONS,
    allocate,
    digest,
    env,
    group,
)
from common import launch as launch_ir  # noqa: E402
from common import (  # noqa: E402
    launch_source,
    save,
    snapshot,
    source_args,
)
from frozen_port import _attn_bwd_dkdv_only as port_kernel  # noqa: E402
from frozen_source import _attn_bwd_dkdv_only as baseline_kernel  # noqa: E402
from validate import cpu_autograd  # noqa: E402

__all__ = [
    "MATRIX",
    "OPTIONS",
    "allocate",
    "baseline_kernel",
    "cpu_autograd",
    "snapshot",
]

ROUTES = ("baseline", "ir", "source")
INPUT_KEYS = ("q", "k", "v", "do", "m", "delta")


def check_source():
    manifest = json.loads((ROOT / "source_manifest.json").read_text())
    assert digest(ROOT / "frozen_port.py") == manifest["source_sha256"]
    assert digest(PARENT / "frozen_source.py") == manifest["parent_source_sha256"]
    for g, sha in manifest["parent_ir"].items():
        assert digest(PARENT / "ir" / g / "r1_dv_late.ttir") == sha
    return manifest


def input_hashes(t):
    return {k: digest(t[k].detach().cpu().numpy().tobytes()) for k in INPUT_KEYS}


def metadata(ck):
    ck._init_handles()
    data = ck.asm["cubin"]
    assert data[:6] == b"\x7fELF\x02\x01", "Requires little-endian ELF64 cubin"
    shoff = struct.unpack_from("<Q", data, 40)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 58)
    headers = [
        struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize)
        for i in range(shnum)
    ]
    string_header = headers[shstrndx]
    strings = data[string_header[4] : string_header[4] + string_header[5]]
    sections = {}
    for header in headers:
        name = strings[header[0] :].split(b"\x00", 1)[0].decode()
        if name.startswith(".text"):
            sections[name] = digest(data[header[4] : header[4] + header[5]])
    result = dict(
        ck.metadata._asdict(),
        n_regs=ck.n_regs,
        n_spills=ck.n_spills,
        hashes={ext: digest(data) for ext, data in ck.asm.items()},
        elf_text_sections=sections,
    )
    from triton.backends.nvidia.compiler import CUDABackend

    target = triton.runtime.driver.active.get_current_target()
    result["parsed_options"] = CUDABackend(target).parse_options(OPTIONS).__dict__
    result["target"] = str(target)
    return result


def preserve(route, d, c, ck):
    folder = ROOT / "artifacts" / group(d, c) / route
    folder.mkdir(parents=True, exist_ok=True)
    for ext, data in ck.asm.items():
        path = folder / f"kernel.{ext}"
        if isinstance(data, bytes):
            path.write_bytes(data)
        else:
            path.write_text(data)
    meta = metadata(ck)
    save(folder / "metadata.json", meta)
    return meta


def compile_routes(t):
    env.check_environment()
    check_source()
    assert not triton.knobs.compilation.override
    assert triton.knobs.runtime.add_stages_inspection_hook is None
    d, c = t["shape"][-1], t["causal"]
    kernels, metas = {}, {}
    for route, filename in (("baseline", "baseline"), ("ir", "r1_dv_late")):
        path = PARENT / "ir" / group(d, c) / (filename + ".ttir")
        kernels[route] = triton.compile(str(path), options=OPTIONS)
        metas[route] = preserve(route, d, c, kernels[route])
        metas[route]["input_ttir_sha256"] = digest(path)
    # This object is compiled by the Python source frontend and is the timed route.
    kernels["source"] = launch_source(port_kernel, t)
    metas["source"] = preserve("source", d, c, kernels["source"])
    metas["source"]["input_source_sha256"] = digest(ROOT / "frozen_port.py")
    assert all(
        metas[r]["parsed_options"] == metas["baseline"]["parsed_options"]
        for r in ROUTES
    )
    assert all(metas[r]["target"] == metas["baseline"]["target"] for r in ROUTES)
    return kernels, metas


def launch(ck, t):
    """Same device ABI/grid; AST launchers additionally consume constexpr bindings."""
    if hasattr(ck.src, "fn"):
        b, h, n, d = t["shape"]
        args = source_args(t) + (32, 128, 128, 32, 2, d, t["causal"])
        ck[(n // 128, 1, b * h)](*args)
    else:
        launch_ir(ck, t)
