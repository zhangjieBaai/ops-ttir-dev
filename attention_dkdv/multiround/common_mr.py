"""Native fixed-contract Attention round2 compiler, snapshots and launcher."""

import json
import os
import struct
import sys
from pathlib import Path

import triton

BASE = Path(__file__).resolve().parent
ROOT = BASE / os.environ.get("ATTENTION_MR_ROUND", "r3")
PARENT = BASE.parent
sys.path.insert(0, str(PARENT))
from common import (  # noqa: E402
    OPTIONS,
    allocate,
    digest,
    env,
    group,
    launch,
    launch_source,
    save,
)
from common import snapshot as original_snapshot  # noqa: E402
from frozen_source import _attn_bwd_dkdv_only  # noqa: E402
from validate import cpu_autograd  # noqa: E402

__all__ = [
    "MATRIX",
    "ROOT",
    "OPTIONS",
    "allocate",
    "compile_ir",
    "cpu_autograd",
    "digest",
    "env",
    "group",
    "input_hashes",
    "launch",
    "launch_source",
    "save",
    "snapshot",
    "_attn_bwd_dkdv_only",
    "candidates",
]

BHNS = [
    (1, 4, 128),
    (1, 4, 256),
    (1, 4, 512),
    (1, 4, 1024),
    (1, 4, 2048),
    (2, 8, 1024),
    (2, 8, 2048),
    (4, 8, 2048),
]
MATRIX = [(b, h, n, d, c) for d in (64, 128) for c in (False, True) for b, h, n in BHNS]
INPUT_KEYS = ("q", "k", "v", "do", "m", "delta")


def snapshot():
    value = original_snapshot()
    requested = os.environ.get("ATTENTION_MR_LOCK_SM_MHZ")
    value["clocks_locked"] = requested is not None
    value["requested_sm_clock_mhz"] = int(requested) if requested else None
    value["clock_note"] = (
        "Fixed GPU4 SM clock; actual observed SM/memory clocks are in gpu snapshot"
        if requested
        else "Automatic clocks; no fixed-frequency claim"
    )
    return value


def text_sections(blob):
    assert blob[:6] == b"\x7fELF\x02\x01"
    offset = struct.unpack_from("<Q", blob, 40)[0]
    size, count, names_index = struct.unpack_from("<HHH", blob, 58)
    heads = [
        struct.unpack_from("<IIQQQQIIQQ", blob, offset + i * size) for i in range(count)
    ]
    h = heads[names_index]
    names = blob[h[4] : h[4] + h[5]]
    sections = {}
    for h in heads:
        name = names[h[0] :].split(b"\0", 1)[0].decode()
        if name.startswith(".text"):
            sections[name] = digest(blob[h[4] : h[4] + h[5]])
    assert sections
    return sections


def input_hashes(t):
    return {k: digest(t[k].cpu().numpy().tobytes()) for k in INPUT_KEYS}


def compile_ir(name, d, c):
    env.check_environment()
    assert not triton.knobs.compilation.override
    assert triton.knobs.runtime.add_stages_inspection_hook is None
    path = ROOT / "ir" / group(d, c) / (name + ".ttir")
    ck = triton.compile(str(path), options=OPTIONS)
    ck._init_handles()
    from triton.backends.nvidia.compiler import CUDABackend

    fields = CUDABackend(ck.metadata.target).parse_options({}).__dict__
    meta = dict(
        ck.metadata._asdict(),
        n_regs=ck.n_regs,
        n_spills=ck.n_spills,
        ttir_sha256=digest(path),
        hashes={k: digest(v) for k, v in ck.asm.items()},
        elf_text=text_sections(ck.asm["cubin"]),
        parsed_options={k: ck.metadata._asdict()[k] for k in fields},
    )
    artifacts = ROOT / "artifacts" / group(d, c) / name
    artifacts.mkdir(parents=True, exist_ok=True)
    for ext, data in ck.asm.items():
        (artifacts / f"kernel.{ext}").write_bytes(
            data if isinstance(data, bytes) else data.encode()
        )
    save(artifacts / "metadata.json", meta)
    return ck, meta


def candidates():
    return {
        g: list(v["candidates"])
        for g, v in json.loads((ROOT / "manifest.json").read_text()).items()
    }
