import triton
import triton.language as tl

@triton.jit
def program_id(
    axis: int,
) -> tl.tensor:
    return tl.program_id(axis).to(tl.int64)

