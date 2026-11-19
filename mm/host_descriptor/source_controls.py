"""Native source controls for ABI isolation; the tutorial computation remains unchanged."""

# flake8: noqa: E402 -- existing standalone owners require explicit path setup.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import triton
import triton.language as tl
from tutorial import matmul_kernel as original_tutorial_mm


@triton.jit
def matched_baseline(
    ad,
    bd,
    cd,
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_bk,
    stride_cm,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GM: tl.constexpr,
):
    # This wrapper only adds a common descriptor-bearing argument envelope.
    original_tutorial_mm(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        1,
        stride_bk,
        1,
        stride_cm,
        1,
        BM,
        BN,
        BK,
        GM,
        "",
    )


@triton.jit
def native_host_control(
    ad,
    bd,
    cd,
    M,
    N,
    K,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GM: tl.constexpr,
):
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(N, BN)
    width = GM * nn
    group = pid // width
    first = group * GM
    size = min(nm - first, GM)
    local = pid % width
    pm = first + local % size
    pn = local // size
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(tl.cdiv(K, BK)):
        a = ad.load([pm * BM, k * BK])
        b = bd.load([k * BK, pn * BN])
        acc = tl.dot(a, b, acc)
    cd.store([pm * BM, pn * BN], acc.to(tl.float16))
