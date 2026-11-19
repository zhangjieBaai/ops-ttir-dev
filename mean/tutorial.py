import triton
import triton.language as tl
import frozen_extension as ext


@triton.jit
def mean_dim_kernel_non_inner_vec(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_K: tl.constexpr,  # number of threads per block along K
    VEC_SIZE: tl.constexpr,  # elements per thread (1 for FP32, 8 for FP16/BF16)
):
    # Determine accumulation and load behavior
    input_dtype = input_ptr.dtype.element_ty
    if tl.constexpr(input_dtype == tl.float16) or tl.constexpr(
        input_dtype == tl.bfloat16
    ):
        ACC_DTYPE = tl.float32
        # VEC_SIZE should be 4 or 8 for vectorization
    else:
        ACC_DTYPE = input_dtype
        # VEC_SIZE = 1 for FP32

    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)

    # Each thread handles VEC_SIZE consecutive elements
    k_base = pid_k * BLOCK_SIZE_K * VEC_SIZE
    k_offsets = (
        k_base
        + tl.arange(0, BLOCK_SIZE_K)[:, None] * VEC_SIZE
        + tl.arange(0, VEC_SIZE)[None, :]
    )
    # Shape: [BLOCK_SIZE_K, VEC_SIZE]
    k_mask = k_offsets < K

    # Accumulator: [BLOCK_SIZE_K, VEC_SIZE]
    acc = tl.zeros((BLOCK_SIZE_K, VEC_SIZE), dtype=ACC_DTYPE)

    base = pid_m * N * K

    for n in range(N):
        offsets = base + n * K + k_offsets
        # This will trigger vectorized load if VEC_SIZE >= 4 and aligned
        val = tl.load(input_ptr + offsets, mask=k_mask, other=0.0)
        acc += val.to(ACC_DTYPE)

    mean_val = acc / N

    # Store back
    out_offsets = pid_m * K + k_offsets
    tl.store(output_ptr + out_offsets, mean_val, mask=k_mask)


@triton.jit
def mean_dim_kernel_non_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    # accumulation dtype
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = input_ptr.dtype.element_ty

    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)

    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)[None, :]

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)[:, None]
        inp_offset = pid_m * N * K + n_offsets * K + k_offsets
        mask = (n_offsets < N) & (k_offsets < K)
        input_ptrs = input_ptr + inp_offset
        inp = tl.load(input_ptrs, mask=mask, other=0).to(cdtype)
        # sum along reduction axis (N) -> keep dims so axis 0 corresponds to TILE_K
        summed = tl.sum(inp, axis=0, keep_dims=True)
        # divide by N to get mean
        out = summed / N
        out_offset = pid_m * K + k_offsets
        output_ptrs = output_ptr + out_offset
        tl.store(output_ptrs, out, mask=k_offsets < K)
    else:
        sum_tile = tl.zeros([TILE_N, TILE_K], dtype=cdtype)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)[:, None]
            inp_offsets = pid_m * N * K + n_offsets * K + k_offsets
            mask = (n_offsets < N) & (k_offsets < K)
            inp = tl.load(input_ptr + inp_offsets, mask=mask, other=0).to(cdtype)
            sum_tile += inp
        summed = tl.sum(sum_tile, axis=0, keep_dims=True)
        out = summed / N
        out_offset = pid_m * K + k_offsets
        output_ptrs = output_ptr + out_offset
        tl.store(output_ptrs, out, mask=k_offsets < K)


@triton.jit
def mean_dim_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = input_ptr.dtype.element_ty

    pid_m = ext.program_id(0)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        inp_offset = pid_m * N + n_offsets
        input_ptrs = input_ptr + inp_offset
        mask = n_offsets < N
        inp = tl.load(input_ptrs, mask=mask, other=0).to(cdtype)
        summed = tl.sum(inp, axis=0)
        out = summed / N
        out_offset = pid_m
        output_ptrs = output_ptr + out_offset
        tl.store(output_ptrs, out)
    else:
        sum_vec = tl.zeros(
            [
                TILE_N,
            ],
            dtype=cdtype,
        )
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            inp_offsets = pid_m * N + n_offsets
            mask = n_offsets < N
            inp = tl.load(input_ptr + inp_offsets, mask=mask, other=0).to(cdtype)
            sum_vec += inp
        summed = tl.sum(sum_vec, axis=0)
        out = summed / N
        out_offset = pid_m
        output_ptrs = output_ptr + out_offset
        tl.store(output_ptrs, out)
