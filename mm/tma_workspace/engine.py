"""Native TTIR compilation and a scoped, shared allocator for the new workspace group."""

# flake8: noqa: E402 -- existing standalone MM helpers require this path bootstrap.

import contextlib
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import triton
from common import ROOT as MM_ROOT
from common import dump_kernel, frozen, gate, launch, sha256
from experiment_round1 import compatible, entry
from triton.runtime import _allocation

ROOT = Path(__file__).resolve().parent
FAMILY = "bm64_bn128_bk64_g8_w4_s4"
CONFIG = frozen()["families"][FAMILY]
OPTIONS = CONFIG["options"]


class NativeScratchAllocator:
    """Allocate only; honor the native driver's active device/stream and alignment."""

    def __init__(self):
        self.device = torch.cuda.current_device()
        self.calls = 0
        self.max_bytes = 0
        self.samples = []

    def __call__(self, size, alignment, stream):
        actual = torch.cuda.current_stream(self.device).cuda_stream
        assert torch.cuda.current_device() == self.device
        assert int(actual) == int(
            stream
        ), "Native launch stream differs from Torch allocator stream"
        buffer = torch.empty((size,), device=f"cuda:{self.device}", dtype=torch.uint8)
        assert buffer.data_ptr() % alignment == 0
        self.calls += 1
        self.max_bytes = max(self.max_bytes, size)
        if len(self.samples) < 8:
            self.samples.append(
                dict(
                    size=size,
                    alignment=alignment,
                    requested_stream=int(stream),
                    actual_stream=int(actual),
                    aligned=True,
                    capturing=torch.cuda.is_current_stream_capturing(),
                )
            )
        return buffer


@contextlib.contextmanager
def allocation_scope(allocator):
    previous = _allocation._allocator.get()
    triton.set_allocator(allocator)
    try:
        assert _allocation._allocator.get() is allocator
        yield
    finally:
        triton.set_allocator(previous)
        assert _allocation._allocator.get() is previous


def compile_native(name):
    gate()
    path = ROOT / "ir" / f"{name}.ttir"
    ck = triton.compile(str(path), options=OPTIONS)
    assert ck.src.ext == "ttir" and ck.src.src == path.read_text()
    ck._init_handles()
    dump_kernel(ck, f"tma_workspace/{name}")
    return ck


def compare_contract(base, candidate):
    compatible(base, candidate, CONFIG)
    assert entry(base) == entry(candidate)
    assert base.metadata.global_scratch_size == 0
    assert candidate.metadata.global_scratch_size == 384
    assert (
        base.metadata.profile_scratch_size
        == candidate.metadata.profile_scratch_size
        == 0
    )
    assert base.metadata.tensordesc_meta == candidate.metadata.tensordesc_meta


def launch_native(ck, tensors):
    launch(ck, tensors, CONFIG)


def fingerprint():
    files = [
        ROOT / "plan.md",
        ROOT / "engine.py",
        ROOT / "experiment.py",
        MM_ROOT / "common.py",
        MM_ROOT / "experiment_round1.py",
        MM_ROOT / "experiment_round2.py",
        Path(inspect.getfile(_allocation)),
    ]
    return {str(p): sha256(p) for p in files}
