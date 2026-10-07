# Alpaccaroo - native-packed Triton kernels for the Qwen35 ROCm target.
# MIT License. See LICENSE.
"""Imported only after strict ROCm/gfx1151 capability checks pass."""

from __future__ import annotations

import triton
import triton.language as tl


BLOCK_SIZE = 1024
_DTYPE_IDS = {
    "Q8_0": 1,
    "Q3_K": 2,
    "Q4_K": 3,
    "Q5_K": 4,
    "Q6_K": 5,
    "IQ4_NL": 6,
    "IQ4_XS": 7,
    "IQ3_S": 8,
}
_BLOCK_INFO = {
    "Q8_0": (32, 34),
    "Q3_K": (256, 110),
    "Q4_K": (256, 144),
    "Q5_K": (256, 176),
    "Q6_K": (256, 210),
    "IQ4_NL": (32, 18),
    "IQ4_XS": (256, 136),
    "IQ3_S": (256, 110),
}


@triton.jit
def _packed_f16(packed, offsets, valid):
    low = tl.load(packed + offsets, mask=valid, other=0).to(tl.uint16)
    high = tl.load(packed + offsets + 1, mask=valid, other=0).to(tl.uint16)
    bits = low | (high << 8)
    return bits.to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _signed_byte(packed, offsets, valid):
    raw = tl.load(packed + offsets, mask=valid, other=0).to(tl.int32)
    return tl.where(raw >= 128, raw - 256, raw).to(tl.float32)


@triton.jit
def _packed_k_scale(packed, block_offsets, scale_index, valid, MINIMUM: tl.constexpr):
    scales = block_offsets + 4
    first = tl.load(
        packed + scales + scale_index + (4 if MINIMUM else 0),
        mask=valid,
        other=0,
    ).to(tl.int32)
    tail = tl.load(
        packed + scales + scale_index + 4,
        mask=valid,
        other=0,
    ).to(tl.int32)
    upper = tl.load(
        packed + scales + scale_index + (-4 if not MINIMUM else 0),
        mask=valid,
        other=0,
    ).to(tl.int32)
    if MINIMUM:
        later = (tail >> 4) | ((upper >> 6) << 4)
    else:
        later = (tail & 15) | ((upper >> 6) << 4)
    return tl.where(scale_index < 4, first & 63, later).to(tl.float32)


@triton.jit
def _packed_q3_scale(packed, block_offsets, scale_index, valid):
    quartet = scale_index // 4
    lane = scale_index - quartet * 4
    low_byte = tl.load(
        packed + block_offsets + 96 + (quartet & 1) * 4 + lane,
        mask=valid,
        other=0,
    ).to(tl.int32)
    high_byte = tl.load(
        packed + block_offsets + 104 + lane,
        mask=valid,
        other=0,
    ).to(tl.int32)
    low_shift = tl.where(quartet < 2, 0, 4)
    raw = ((low_byte >> low_shift) & 15) | (
        ((high_byte >> (2 * quartet)) & 3) << 4
    )
    return (raw - 32).to(tl.float32)


@triton.jit
def _packed_matvec(
    packed,
    values,
    output,
    codebook,
    ROW_BYTES: tl.constexpr,
    COLUMNS: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_ELEMENTS: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    local_offsets = tl.arange(0, TILE_SIZE)
    columns = tile * TILE_SIZE + local_offsets
    valid = columns < COLUMNS
    block = columns // BLOCK_ELEMENTS
    local = columns - block * BLOCK_ELEMENTS
    block_offsets = row * ROW_BYTES + block * BLOCK_BYTES

    if DTYPE_ID == 1:  # Q8_0
        d = _packed_f16(packed, block_offsets, valid)
        code = _signed_byte(packed, block_offsets + 2 + local, valid)
        weight = d * code

    elif DTYPE_ID == 2:  # Q3_K
        half = local // 128
        within = local - half * 128
        group = within // 32
        group_local = within - group * 32
        second = group_local // 16
        lane = group_local - second * 16
        quant = tl.load(
            packed + block_offsets + 32 + half * 32 + second * 16 + lane,
            mask=valid,
            other=0,
        ).to(tl.int32)
        low = (quant >> (2 * group)) & 3
        high_mask = 1 << (half * 4 + group)
        high_byte = tl.load(
            packed + block_offsets + second * 16 + lane,
            mask=valid,
            other=0,
        ).to(tl.int32)
        high = tl.where((high_byte & high_mask) != 0, 0, 4)
        scale_index = half * 8 + group * 2 + second
        scale = _packed_q3_scale(
            packed, block_offsets, scale_index, valid,
        )
        d = _packed_f16(packed, block_offsets + 108, valid)
        weight = d * scale * (low - high).to(tl.float32)

    elif DTYPE_ID == 3 or DTYPE_ID == 4:  # Q4_K / Q5_K
        chunk = local // 64
        within = local - chunk * 64
        second = within // 32
        lane = within - second * 32
        quant_start = 16 if DTYPE_ID == 3 else 48
        quant = tl.load(
            packed + block_offsets + quant_start + chunk * 32 + lane,
            mask=valid,
            other=0,
        ).to(tl.int32)
        code = tl.where(second == 0, quant & 15, quant >> 4)
        if DTYPE_ID == 4:
            high_byte = tl.load(
                packed + block_offsets + 16 + lane,
                mask=valid,
                other=0,
            ).to(tl.int32)
            high_mask = 1 << (2 * chunk + second)
            code += tl.where((high_byte & high_mask) != 0, 16, 0)
        scale_index = 2 * chunk + second
        scale = _packed_k_scale(
            packed, block_offsets, scale_index, valid, MINIMUM=False,
        )
        minimum = _packed_k_scale(
            packed, block_offsets, scale_index, valid, MINIMUM=True,
        )
        d = _packed_f16(packed, block_offsets, valid)
        dmin = _packed_f16(packed, block_offsets + 2, valid)
        weight = d * scale * code.to(tl.float32) - dmin * minimum

    elif DTYPE_ID == 5:  # Q6_K
        half = local // 128
        within = local - half * 128
        quarter = within // 32
        lane = within - quarter * 32
        low_byte = tl.load(
            packed + block_offsets + half * 64 + (quarter & 1) * 32 + lane,
            mask=valid,
            other=0,
        ).to(tl.int32)
        low = tl.where(quarter < 2, low_byte & 15, low_byte >> 4)
        high_byte = tl.load(
            packed + block_offsets + 128 + half * 32 + lane,
            mask=valid,
            other=0,
        ).to(tl.int32)
        code = low | (((high_byte >> (2 * quarter)) & 3) << 4)
        scale_index = half * 8 + lane // 16 + quarter * 2
        scale = _signed_byte(
            packed, block_offsets + 192 + scale_index, valid,
        )
        d = _packed_f16(packed, block_offsets + 208, valid)
        weight = d * scale * (code - 32).to(tl.float32)

    elif DTYPE_ID == 6:  # IQ4_NL
        second = local // 16
        lane = local - second * 16
        quant = tl.load(
            packed + block_offsets + 2 + lane,
            mask=valid,
            other=0,
        ).to(tl.int32)
        code = tl.where(second == 0, quant & 15, quant >> 4)
        magnitude = tl.load(codebook + code, mask=valid, other=0).to(tl.float32)
        d = _packed_f16(packed, block_offsets, valid)
        weight = d * magnitude

    elif DTYPE_ID == 7:  # IQ4_XS
        subblock = local // 32
        within = local - subblock * 32
        second = within // 16
        lane = within - second * 16
        scale_low_byte = tl.load(
            packed + block_offsets + 4 + subblock // 2,
            mask=valid,
            other=0,
        ).to(tl.int32)
        scale_low = (
            scale_low_byte >> (4 * (subblock & 1))
        ) & 15
        scales_high_low = tl.load(
            packed + block_offsets + 2,
            mask=valid,
            other=0,
        ).to(tl.int32)
        scales_high_high = tl.load(
            packed + block_offsets + 3,
            mask=valid,
            other=0,
        ).to(tl.int32)
        scales_high = scales_high_low | (scales_high_high << 8)
        scale_high = ((scales_high >> (2 * subblock)) & 3) << 4
        quant = tl.load(
            packed + block_offsets + 8 + subblock * 16 + lane,
            mask=valid,
            other=0,
        ).to(tl.int32)
        code = tl.where(second == 0, quant & 15, quant >> 4)
        magnitude = tl.load(codebook + code, mask=valid, other=0).to(tl.float32)
        d = _packed_f16(packed, block_offsets, valid)
        weight = d * ((scale_low | scale_high) - 32) * magnitude

    else:  # IQ3_S
        subblock = local // 32
        within = local - subblock * 32
        lane = within // 8
        coordinate = within - lane * 8
        second = coordinate // 4
        grid_coordinate = coordinate - second * 4
        scale_byte = tl.load(
            packed + block_offsets + 106 + subblock // 2,
            mask=valid,
            other=0,
        ).to(tl.int32)
        nibble = tl.where(
            (subblock & 1) == 0, scale_byte & 15, scale_byte >> 4,
        )
        high_bits = tl.load(
            packed + block_offsets + 66 + subblock,
            mask=valid,
            other=0,
        ).to(tl.int32)
        grid_low = tl.load(
            packed + block_offsets + 2 + subblock * 8 + 2 * lane + second,
            mask=valid,
            other=0,
        ).to(tl.int32)
        grid_high = (high_bits >> (2 * lane + second)) & 1
        grid_index = grid_low | (grid_high << 8)
        grid = tl.load(
            codebook + grid_index, mask=valid, other=0,
        ).to(tl.int32)
        magnitude = (grid >> (8 * grid_coordinate)) & 255
        signs = tl.load(
            packed + block_offsets + 74 + subblock * 4 + lane,
            mask=valid,
            other=0,
        ).to(tl.int32)
        sign = tl.where((signs & (1 << coordinate)) != 0, -1.0, 1.0)
        d = _packed_f16(packed, block_offsets, valid)
        weight = d * (1 + 2 * nibble) * magnitude.to(tl.float32) * sign

    source = tl.load(values + columns, mask=valid, other=0.0).to(tl.float32)
    total = tl.sum(weight * source, axis=0)
    tl.atomic_add(output + row, total)


def launch_packed_matvec(
    packed,
    values,
    output,
    codebook=None,
    *,
    dtype: str,
    rows: int,
    columns: int,
    packed_row_bytes: int,
) -> None:
    """Launch one tiled row-dot directly over native GGUF blocks."""

    try:
        dtype_id = _DTYPE_IDS[dtype]
        block_elements, block_bytes = _BLOCK_INFO[dtype]
    except KeyError:
        raise ValueError(f"no Triton packed matvec for {dtype}") from None
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise ValueError("packed Triton rows must be a positive integer")
    if columns <= 0 or columns % block_elements:
        raise ValueError(
            f"{dtype} Triton columns must be positive and divisible by "
            f"{block_elements}"
        )
    expected_row_bytes = columns // block_elements * block_bytes
    if packed_row_bytes != expected_row_bytes:
        raise ValueError(f"{dtype} Triton packed row geometry differs")
    grid = (rows, triton.cdiv(columns, BLOCK_SIZE))
    if codebook is None:
        codebook = packed
    _packed_matvec[grid](
        packed,
        values,
        output,
        codebook,
        ROW_BYTES=packed_row_bytes,
        COLUMNS=columns,
        DTYPE_ID=dtype_id,
        BLOCK_ELEMENTS=block_elements,
        BLOCK_BYTES=block_bytes,
        TILE_SIZE=BLOCK_SIZE,
        num_warps=4,
    )


def launch_q4_k_matvec(
    packed,
    values,
    output,
    *,
    rows: int,
    columns: int,
    packed_row_bytes: int,
) -> None:
    """Compatibility wrapper for the original Q4_K-only launch seam."""

    launch_packed_matvec(
        packed,
        values,
        output,
        dtype="Q4_K",
        rows=rows,
        columns=columns,
        packed_row_bytes=packed_row_bytes,
    )


__all__ = ["launch_packed_matvec", "launch_q4_k_matvec"]
