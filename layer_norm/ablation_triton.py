"""Public Triton implementation of the final forward TTIR algorithm, for parity experiments."""

import torch
import triton
import triton.language as tl
from common import OPTIONS, ROOT, allocate, compile_ir, launch, save_json, sha256


@triton.jit
def _body(X, Y, W, B, Mean, Rstd, stride, N, eps, C: tl.constexpr, K: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, C)
    acc = tl.full((C,), 0, tl.float32)
    if K > 0:
        for i in tl.static_range(K):
            c = i * C + cols
            x = tl.load(X + row * stride + c, c < N, other=0, cache_modifier=".ca").to(
                tl.float32
            )
            acc += x
    else:
        for i in range(tl.cdiv(N, C)):
            c = i * C + cols
            x = tl.load(X + row * stride + c, c < N, other=0, cache_modifier=".ca").to(
                tl.float32
            )
            acc += x
    inv = 1.0 / N
    mean = tl.sum(acc, 0) * inv
    acc = tl.full((C,), 0, tl.float32)
    if K > 0:
        for i in tl.static_range(K):
            c = i * C + cols
            x = tl.load(X + row * stride + c, c < N, other=0).to(tl.float32)
            v = tl.where(c < N, x - mean, 0.0)
            acc += v * v
    else:
        for i in range(tl.cdiv(N, C)):
            c = i * C + cols
            x = tl.load(X + row * stride + c, c < N, other=0).to(tl.float32)
            v = tl.where(c < N, x - mean, 0.0)
            acc += v * v
    rstd = tl.rsqrt(tl.sum(acc, 0) * inv + eps)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    if K > 0:
        for i in tl.static_range(K):
            c = i * C + cols
            x = tl.load(X + row * stride + c, c < N, other=0).to(tl.float32)
            w = tl.load(W + c, c < N, other=0).to(tl.float32)
            b = tl.load(B + c, c < N, other=0).to(tl.float32)
            y = ((x - mean) * rstd) * w + b
            tl.store(Y + row * stride + c, y, c < N, cache_modifier=".cs")
    else:
        for i in range(tl.cdiv(N, C)):
            c = i * C + cols
            x = tl.load(X + row * stride + c, c < N, other=0).to(tl.float32)
            w = tl.load(W + c, c < N, other=0).to(tl.float32)
            b = tl.load(B + c, c < N, other=0).to(tl.float32)
            y = ((x - mean) * rstd) * w + b
            tl.store(Y + row * stride + c, y, c < N, cache_modifier=".cs")


@triton.jit
def forward_source(X, Y, W, B, Mean, Rstd, stride, N, eps):
    if N <= 1024:
        _body(X, Y, W, B, Mean, Rstd, stride, N, eps, 1024, 1)
    elif N <= 2048:
        _body(X, Y, W, B, Mean, Rstd, stride, N, eps, 1024, 2)
    elif N <= 3072:
        _body(X, Y, W, B, Mean, Rstd, stride, N, eps, 1024, 3)
    elif N <= 4096:
        _body(X, Y, W, B, Mean, Rstd, stride, N, eps, 2048, 2)
    elif N <= 6144:
        _body(X, Y, W, B, Mean, Rstd, stride, N, eps, 2048, 3)
    elif N <= 8192:
        _body(X, Y, W, B, Mean, Rstd, stride, N, eps, 2048, 4)
    elif N <= 10240:
        _body(X, Y, W, B, Mean, Rstd, stride, N, eps, 2048, 5)
    elif N <= 12288:
        _body(X, Y, W, B, Mean, Rstd, stride, N, eps, 2048, 6)
    elif N <= 14336:
        _body(X, Y, W, B, Mean, Rstd, stride, N, eps, 2048, 7)
    elif N <= 16384:
        _body(X, Y, W, B, Mean, Rstd, stride, N, eps, 2048, 8)
    else:
        _body(X, Y, W, B, Mean, Rstd, stride, N, eps, 2048, 0)


def main():
    ts = allocate(1151, 8192)
    x, y, w, b, mean, rstd = ts[:6]
    ck = forward_source[(1151,)](
        x, y, w, b, mean, rstd, x.stride(0), 8192, 1e-5, **OPTIONS
    )
    path = ROOT / "ir/abl_source_formatted.ttir"
    text = ck.asm["ttir"]
    if path.exists():
        assert path.read_text() == text
    else:
        path.write_text(text)
    reference = [t.clone() for t in [y, mean, rstd]]
    loaded = compile_ir("abl_source_formatted")
    launch(loaded, ts)
    assert all(torch.equal(a, v) for a, v in zip(reference, [y, mean, rstd]))
    matches = {s: ck.asm[s] == loaded.asm[s] for s in ["llir", "ptx", "cubin"]}
    assert all(matches.values())
    save_json(
        ROOT / "results/ablation/source_formatted_parity.json",
        dict(
            passed=True,
            source_sha256=sha256(__file__),
            ttir_sha256=sha256(path),
            source_file_downstream_equal=matches,
            output_bitwise_equal=True,
            options=OPTIONS,
            n_regs=ck.n_regs,
            n_spills=ck.n_spills,
        ),
    )


if __name__ == "__main__":
    main()
