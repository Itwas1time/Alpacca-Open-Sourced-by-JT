# Alpaccaroo - dedicated Qwen35 hybrid CUDA primitives and transactions.
# MIT License. See LICENSE.
"""Qwen35 CUDA boundary, deliberately separate from conventional DecodeChain.

Primitive kernels can run under Numba's CUDA simulator for deterministic CI.
Production capability requires the pinned JIT and a real NVIDIA device; the
simulator is never accepted by :func:`cuda_capability`.  The transactional
chain is backend-neutral enough to fault-test checkpoint/replay without
claiming that packed projection kernels have been qualified on this host.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .gguf import GGML_BLOCK_INFO
from .memory import ModelState, StateSnapshot
from .quants import _IQ3S_GRID, _IQ4_NL_VALUES

NUMBA_PIN = "0.65.1"
NUMBA_CUDA_PIN = "0.30.4"
DEFAULT_CHECKPOINT_INTERVAL = 128
MIN_CHECKPOINT_INTERVAL = 128
MAX_CHECKPOINT_INTERVAL = 512

# No packed dtype is production-qualified until it has run on a real NVIDIA
# device.  Storage layout support lives in packed_gpu.PACKED_LAYOUT_DTYPES and
# must not be confused with this execution qualification set.
CUDA_PACKED_KERNEL_DTYPES: frozenset[str] = frozenset()

# Implemented means that the kernel has deterministic simulator parity.  It is
# deliberately separate from CUDA_PACKED_KERNEL_DTYPES, which remains empty
# until these kernels pass the real-device qualification gates.
CUDA_PACKED_IMPLEMENTED_DTYPES: frozenset[str] = frozenset({
    "F32", "IQ3_S", "IQ4_NL", "IQ4_XS", "Q3_K", "Q4_K", "Q5_K",
    "Q6_K", "Q8_0",
})
_PACKED_DTYPE_IDS = {
    "Q8_0": 1,
    "IQ4_NL": 2,
    "IQ4_XS": 3,
    "IQ3_S": 4,
    "Q3_K": 5,
    "Q4_K": 6,
    "Q5_K": 7,
    "Q6_K": 8,
}


class Qwen35CudaUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CudaCapability:
    available: bool
    reason: str
    numba_version: str | None = None
    numba_cuda_version: str | None = None
    device_name: str | None = None
    simulator: bool = False
    implemented_packed_dtypes: tuple[str, ...] = tuple(
        sorted(CUDA_PACKED_IMPLEMENTED_DTYPES)
    )
    qualified_packed_dtypes: tuple[str, ...] = ()

    def descriptor(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "numba_version": self.numba_version,
            "numba_cuda_version": self.numba_cuda_version,
            "device_name": self.device_name,
            "simulator": self.simulator,
            "implemented_packed_dtypes": list(self.implemented_packed_dtypes),
            "qualified_packed_dtypes": list(self.qualified_packed_dtypes),
        }


def cuda_capability(*, allow_simulator: bool = False) -> CudaCapability:
    mode = os.environ.get("ALPACCAROO_GPU", "").strip().lower()
    if mode in ("0", "off", "no") or os.environ.get("ALPACCAROO_PURE"):
        return CudaCapability(False, "CUDA disabled by environment")
    try:
        import numba
        import numba_cuda
        from numba import cuda
    except Exception as exc:
        return CudaCapability(False, f"pinned CUDA dependencies unavailable: {exc}")
    if numba.__version__ != NUMBA_PIN:
        return CudaCapability(
            False, f"numba {numba.__version__} != pinned {NUMBA_PIN}",
            numba_version=numba.__version__,
            numba_cuda_version=numba_cuda.__version__,
        )
    if numba_cuda.__version__ != NUMBA_CUDA_PIN:
        return CudaCapability(
            False,
            f"numba-cuda {numba_cuda.__version__} != pinned {NUMBA_CUDA_PIN}",
            numba_version=numba.__version__,
            numba_cuda_version=numba_cuda.__version__,
        )
    simulator = os.environ.get("NUMBA_ENABLE_CUDASIM", "").strip() not in (
        "", "0", "off", "no",
    )
    if simulator and not allow_simulator:
        return CudaCapability(
            False, "Numba CUDA simulator is not a production device",
            numba_version=numba.__version__,
            numba_cuda_version=numba_cuda.__version__,
            simulator=True,
        )
    try:
        if not cuda.is_available():
            return CudaCapability(
                False, "no supported NVIDIA CUDA device detected",
                numba_version=numba.__version__,
                numba_cuda_version=numba_cuda.__version__,
                simulator=simulator,
            )
        if simulator:
            name = "Numba CUDA simulator"
        else:
            device = cuda.get_current_device()
            name = device.name.decode() if isinstance(device.name, bytes) else str(device.name)
    except Exception as exc:
        return CudaCapability(
            False, f"CUDA initialization failed: {exc}",
            numba_version=numba.__version__,
            numba_cuda_version=numba_cuda.__version__,
            simulator=simulator,
        )
    return CudaCapability(
        True,
        "CUDA primitive runtime available; packed model execution remains unqualified",
        numba_version=numba.__version__,
        numba_cuda_version=numba_cuda.__version__,
        device_name=name,
        simulator=simulator,
        qualified_packed_dtypes=tuple(sorted(CUDA_PACKED_KERNEL_DTYPES)),
    )


_kernels: dict[str, Any] | None = None
cuda = None  # populated lazily; module-global so Numba's simulator can patch it


def cuda_kernels(*, allow_simulator: bool = False) -> dict[str, Any]:
    """Return compiled primitive launchers after a strict capability check."""

    global _kernels, cuda
    capability = cuda_capability(allow_simulator=allow_simulator)
    if not capability.available:
        raise Qwen35CudaUnavailable(capability.reason)
    if _kernels is not None:
        return _kernels

    import numpy as np
    from numba import cuda as cuda_module

    cuda = cuda_module

    @cuda.jit
    def _conv_step(current, weights, history, output):
        channel = cuda.grid(1)
        if channel < current.shape[0]:
            total = 0.0
            for tap in range(weights.shape[1] - 1):
                total += history[tap, channel] * weights[channel, tap]
            total += current[channel] * weights[channel, weights.shape[1] - 1]
            output[channel] = total / (1.0 + np.exp(-total))
            if history.shape[0] > 0:
                for tap in range(history.shape[0] - 1):
                    history[tap, channel] = history[tap + 1, channel]
                history[history.shape[0] - 1, channel] = current[channel]

    @cuda.jit
    def _conv_step_out(
        current, weights, source_history, target_history, output,
    ):
        """Write a complete next history without copying or mutating source."""

        channel = cuda.grid(1)
        if channel < current.shape[0]:
            total = 0.0
            for tap in range(weights.shape[1] - 1):
                total += source_history[tap, channel] * weights[channel, tap]
            total += current[channel] * weights[channel, weights.shape[1] - 1]
            output[channel] = total / (1.0 + np.exp(-total))
            if target_history.shape[0] > 0:
                for tap in range(target_history.shape[0] - 1):
                    target_history[tap, channel] = source_history[tap + 1, channel]
                target_history[target_history.shape[0] - 1, channel] = current[channel]

    @cuda.jit
    def _conv_raw_observe(current, weights, history, output):
        """Read pre-SiLU convolution output without mutating live history."""

        channel = cuda.grid(1)
        if channel < current.shape[0]:
            total = 0.0
            for tap in range(weights.shape[1] - 1):
                total += history[tap, channel] * weights[channel, tap]
            total += current[channel] * weights[channel, weights.shape[1] - 1]
            output[channel] = total

    @cuda.jit
    def _normalize_heads(source, output, epsilon):
        head = cuda.grid(1)
        if head < source.shape[0]:
            total = 0.0
            for column in range(source.shape[1]):
                total += source[head, column] * source[head, column]
            denominator = max(np.sqrt(total), epsilon)
            for column in range(source.shape[1]):
                output[head, column] = source[head, column] / denominator

    @cuda.jit
    def _beta_decay(beta_logits, alpha_logits, dt_bias, ssm_a, beta, log_decay):
        head = cuda.grid(1)
        if head < beta_logits.shape[0]:
            value = beta_logits[head]
            if value >= 0.0:
                z = np.exp(-value)
                beta[head] = 1.0 / (1.0 + z)
            else:
                z = np.exp(value)
                beta[head] = z / (1.0 + z)
            x = alpha_logits[head] + dt_bias[head]
            if x > 0.0:
                dt = x + np.log1p(np.exp(-x))
            else:
                dt = np.log1p(np.exp(x))
            log_decay[head] = dt * ssm_a[head]

    @cuda.jit
    def _gdn_step(state, queries, keys, values, beta, log_decay, output):
        # Exactly one execution owner per value-head matrix: no atomics and no
        # cross-head mutable state.  A real-device tuning pass may distribute
        # rows within a block after parity is qualified.
        head = cuda.grid(1)
        if head < state.shape[0]:
            key_head = head % keys.shape[0]
            decay = np.exp(log_decay[head])
            for row in range(state.shape[1]):
                prediction = 0.0
                for column in range(state.shape[2]):
                    decayed = state[head, row, column] * decay
                    state[head, row, column] = decayed
                    prediction += decayed * keys[key_head, column]
                correction = beta[head] * (values[head, row] - prediction)
                for column in range(state.shape[2]):
                    state[head, row, column] += correction * keys[key_head, column]
            scale = 1.0 / np.sqrt(state.shape[2])
            for row in range(state.shape[1]):
                total = 0.0
                for column in range(state.shape[2]):
                    total += state[head, row, column] * queries[key_head, column]
                output[head, row] = total * scale

    @cuda.jit
    def _gdn_step_out(
        source_state, target_state, queries, keys, values,
        beta, log_decay, output,
    ):
        """Write the complete next DeltaNet generation from immutable source."""

        head = cuda.grid(1)
        if head < source_state.shape[0]:
            key_head = head % keys.shape[0]
            decay = np.exp(log_decay[head])
            for row in range(source_state.shape[1]):
                prediction = 0.0
                for column in range(source_state.shape[2]):
                    decayed = source_state[head, row, column] * decay
                    target_state[head, row, column] = decayed
                    prediction += decayed * keys[key_head, column]
                correction = beta[head] * (values[head, row] - prediction)
                for column in range(source_state.shape[2]):
                    target_state[head, row, column] += (
                        correction * keys[key_head, column]
                    )
            scale = 1.0 / np.sqrt(source_state.shape[2])
            for row in range(source_state.shape[1]):
                total = 0.0
                for column in range(source_state.shape[2]):
                    total += (
                        target_state[head, row, column]
                        * queries[key_head, column]
                    )
                output[head, row] = total * scale

    @cuda.jit
    def _gated_rms(values, weight, gates, epsilon, output):
        head = cuda.grid(1)
        if head < values.shape[0]:
            total = 0.0
            for column in range(values.shape[1]):
                total += values[head, column] * values[head, column]
            denominator = np.sqrt(total / values.shape[1] + epsilon)
            for column in range(values.shape[1]):
                gate = gates[head, column]
                if gate >= 0.0:
                    z = np.exp(-gate)
                    activated = gate / (1.0 + z)
                else:
                    z = np.exp(gate)
                    activated = gate * z / (1.0 + z)
                output[head, column] = (
                    values[head, column] / denominator * weight[column]
                    * activated
                )

    @cuda.jit
    def _attention_decode(queries, keys, values, output):
        head = cuda.grid(1)
        if head < queries.shape[0]:
            group = queries.shape[0] // keys.shape[1]
            key_head = head // group
            scale = 1.0 / np.sqrt(queries.shape[1])
            maximum = -np.inf
            denominator = 0.0
            for position in range(keys.shape[0]):
                score = 0.0
                for column in range(queries.shape[1]):
                    score += queries[head, column] * keys[position, key_head, column]
                score *= scale
                next_maximum = max(maximum, score)
                old_scale = np.exp(maximum - next_maximum)
                new_scale = np.exp(score - next_maximum)
                for column in range(values.shape[2]):
                    output[head, column] = (
                        output[head, column] * old_scale
                        + values[position, key_head, column] * new_scale
                    )
                denominator = denominator * old_scale + new_scale
                maximum = next_maximum
            for column in range(values.shape[2]):
                output[head, column] /= denominator

    @cuda.jit(device=True)
    def _packed_u16(packed, offset):
        return int(packed[offset]) | (int(packed[offset + 1]) << 8)

    @cuda.jit(device=True)
    def _packed_u32(packed, offset):
        return (
            int(packed[offset])
            | (int(packed[offset + 1]) << 8)
            | (int(packed[offset + 2]) << 16)
            | (int(packed[offset + 3]) << 24)
        )

    @cuda.jit(device=True)
    def _packed_i8(packed, offset):
        value = int(packed[offset])
        return value - 256 if value >= 128 else value

    @cuda.jit(device=True)
    def _packed_f16(packed, offset):
        """Decode one little-endian IEEE binary16 without a widened array."""

        bits = _packed_u16(packed, offset)
        sign = -1.0 if bits & 0x8000 else 1.0
        exponent = (bits >> 10) & 0x1F
        fraction = bits & 0x03FF
        if exponent == 0:
            return sign * fraction * 5.960464477539063e-8
        if exponent == 0x1F:
            return sign * np.inf if fraction == 0 else np.nan
        return sign * (1.0 + fraction / 1024.0) * (2.0 ** (exponent - 15))

    @cuda.jit(device=True)
    def _packed_k_scale(packed, block_offset, scale_index, minimum):
        """Read one Q4_K/Q5_K six-bit scale or minimum."""

        scales = block_offset + 4
        if scale_index < 4:
            if minimum:
                return int(packed[scales + scale_index + 4]) & 63
            return int(packed[scales + scale_index]) & 63
        if minimum:
            return (
                (int(packed[scales + scale_index + 4]) >> 4)
                | ((int(packed[scales + scale_index]) >> 6) << 4)
            )
        return (
            (int(packed[scales + scale_index + 4]) & 0x0F)
            | ((int(packed[scales + scale_index - 4]) >> 6) << 4)
        )

    @cuda.jit(device=True)
    def _packed_q3_scale(packed, block_offset, scale_index):
        """Read one signed Q3_K six-bit scale in llama.cpp element order."""

        aux0 = _packed_u32(packed, block_offset + 96)
        aux1 = _packed_u32(packed, block_offset + 100)
        high = _packed_u32(packed, block_offset + 104)
        kmask1 = 0x03030303
        kmask2 = 0x0F0F0F0F
        word_index = scale_index // 4
        if word_index == 0:
            word = (aux0 & kmask2) | (((high >> 0) & kmask1) << 4)
        elif word_index == 1:
            word = (aux1 & kmask2) | (((high >> 2) & kmask1) << 4)
        elif word_index == 2:
            word = ((aux0 >> 4) & kmask2) | (((high >> 4) & kmask1) << 4)
        else:
            word = ((aux1 >> 4) & kmask2) | (((high >> 6) & kmask1) << 4)
        value = (word >> (8 * (scale_index % 4))) & 0xFF
        if value >= 128:
            value -= 256
        return value - 32

    @cuda.jit(device=True)
    def _packed_quant_value(
        packed, row_offset, column, dtype_id, iq4_values, iq3_grid,
    ):
        """Decode one logical value directly from a native packed row."""

        if dtype_id == 1:  # Q8_0
            block = column // 32
            local = column - block * 32
            offset = row_offset + block * 34
            return (
                _packed_f16(packed, offset)
                * _packed_i8(packed, offset + 2 + local)
            )

        if dtype_id == 2:  # IQ4_NL
            block = column // 32
            local = column - block * 32
            offset = row_offset + block * 18
            quant = int(packed[offset + 2 + local % 16])
            code = quant & 0x0F if local < 16 else quant >> 4
            return _packed_f16(packed, offset) * iq4_values[code]

        if dtype_id == 3:  # IQ4_XS
            block = column // 256
            local = column - block * 256
            subblock = local // 32
            within = local - subblock * 32
            offset = row_offset + block * 136
            scales_high = _packed_u16(packed, offset + 2)
            scale_byte = int(packed[offset + 4 + subblock // 2])
            low = (scale_byte >> (4 * (subblock % 2))) & 0x0F
            high = ((scales_high >> (2 * subblock)) & 0x03) << 4
            quant = int(packed[offset + 8 + subblock * 16 + within % 16])
            code = quant & 0x0F if within < 16 else quant >> 4
            return (
                _packed_f16(packed, offset) * ((low | high) - 32)
                * iq4_values[code]
            )

        if dtype_id == 4:  # IQ3_S
            block = column // 256
            local = column - block * 256
            subblock = local // 32
            within = local - subblock * 32
            lane = within // 8
            coordinate = within - lane * 8
            offset = row_offset + block * 110
            scale_byte = int(packed[offset + 106 + subblock // 2])
            if subblock % 2 == 0:
                nibble = scale_byte & 0x0F
            else:
                nibble = scale_byte >> 4
            effective = _packed_f16(packed, offset) * (1 + 2 * nibble)
            high_bits = int(packed[offset + 66 + subblock])
            quant_offset = offset + 2 + subblock * 8 + 2 * lane
            if coordinate < 4:
                grid_index = (
                    int(packed[quant_offset])
                    | (((high_bits >> (2 * lane)) & 1) << 8)
                )
                grid_coordinate = coordinate
            else:
                grid_index = (
                    int(packed[quant_offset + 1])
                    | (((high_bits >> (2 * lane + 1)) & 1) << 8)
                )
                grid_coordinate = coordinate - 4
            grid = int(iq3_grid[grid_index])
            magnitude = (grid >> (8 * grid_coordinate)) & 0xFF
            signs = int(packed[offset + 74 + subblock * 4 + lane])
            sign = -1.0 if signs & (1 << coordinate) else 1.0
            return effective * magnitude * sign

        if dtype_id == 5:  # Q3_K
            block = column // 256
            local = column - block * 256
            half = local // 128
            within = local - half * 128
            group = within // 32
            group_local = within - group * 32
            second = group_local // 16
            lane = group_local - second * 16
            offset = row_offset + block * 110
            shift = 2 * group
            quant_offset = offset + 32 + half * 32 + second * 16 + lane
            low = (int(packed[quant_offset]) >> shift) & 3
            mask = 1 << (half * 4 + group)
            high_offset = offset + second * 16 + lane
            high = 0 if int(packed[high_offset]) & mask else 4
            scale_index = half * 8 + group * 2 + second
            return (
                _packed_f16(packed, offset + 108)
                * _packed_q3_scale(packed, offset, scale_index)
                * (low - high)
            )

        if dtype_id == 6 or dtype_id == 7:  # Q4_K / Q5_K
            block = column // 256
            local = column - block * 256
            chunk = local // 64
            within = local - chunk * 64
            second = within // 32
            lane = within - second * 32
            block_bytes = 144 if dtype_id == 6 else 176
            quant_start = 16 if dtype_id == 6 else 48
            offset = row_offset + block * block_bytes
            quant = int(packed[offset + quant_start + chunk * 32 + lane])
            code = quant & 0x0F if second == 0 else quant >> 4
            if dtype_id == 7:
                high = int(packed[offset + 16 + lane])
                if high & (1 << (2 * chunk + second)):
                    code += 16
            scale_index = 2 * chunk + second
            return (
                _packed_f16(packed, offset)
                * _packed_k_scale(packed, offset, scale_index, False)
                * code
                - _packed_f16(packed, offset + 2)
                * _packed_k_scale(packed, offset, scale_index, True)
            )

        if dtype_id == 8:  # Q6_K
            block = column // 256
            local = column - block * 256
            half = local // 128
            within = local - half * 128
            quarter = within // 32
            lane = within - quarter * 32
            offset = row_offset + block * 210
            low_offset = offset + half * 64
            high = int(packed[offset + 128 + half * 32 + lane])
            if quarter == 0:
                low = int(packed[low_offset + lane]) & 0x0F
            elif quarter == 1:
                low = int(packed[low_offset + 32 + lane]) & 0x0F
            elif quarter == 2:
                low = int(packed[low_offset + lane]) >> 4
            else:
                low = int(packed[low_offset + 32 + lane]) >> 4
            code = low | (((high >> (2 * quarter)) & 3) << 4)
            scale_index = half * 8 + lane // 16 + quarter * 2
            return (
                _packed_f16(packed, offset + 208)
                * _packed_i8(packed, offset + 192 + scale_index)
                * (code - 32)
            )

        return 0.0

    @cuda.jit
    def _packed_f32_matvec(weights, values, output, columns):
        row = cuda.grid(1)
        if row < output.shape[0]:
            total = np.float32(0.0)
            row_offset = row * columns
            for column in range(columns):
                total += weights[row_offset + column] * values[column]
            output[row] = total

    @cuda.jit
    def _packed_f32_row(weights, output, row_index, columns):
        column = cuda.grid(1)
        if column < columns:
            output[column] = weights[row_index * columns + column]

    @cuda.jit
    def _packed_quant_row(
        packed, output, row_offset, dtype_id, iq4_values, iq3_grid,
    ):
        column = cuda.grid(1)
        if column < output.shape[0]:
            output[column] = _packed_quant_value(
                packed, row_offset, column, dtype_id, iq4_values, iq3_grid,
            )

    @cuda.jit
    def _packed_f32_matmul(weights, values, output, columns):
        index = cuda.grid(1)
        if index < output.size:
            batch = index // output.shape[1]
            row = index - batch * output.shape[1]
            total = np.float32(0.0)
            row_offset = row * columns
            for column in range(columns):
                total += weights[row_offset + column] * values[batch, column]
            output[batch, row] = total

    @cuda.jit
    def _packed_quant_matmul(
        packed, values, output, row_bytes, dtype_id, iq4_values, iq3_grid,
    ):
        index = cuda.grid(1)
        if index < output.size:
            batch = index // output.shape[1]
            row = index - batch * output.shape[1]
            total = np.float32(0.0)
            row_offset = row * row_bytes
            for column in range(values.shape[1]):
                total += _packed_quant_value(
                    packed, row_offset, column, dtype_id,
                    iq4_values, iq3_grid,
                ) * values[batch, column]
            output[batch, row] = total

    @cuda.jit
    def _packed_quant_matvec(
        packed, values, output, row_bytes, blocks_per_row, dtype_id,
        iq4_values, iq3_grid,
    ):
        """One-row correctness kernel over native GGUF packed blocks.

        No branch constructs a logical code or float matrix.  Each packed
        code, scale and minimum is decoded only while contributing to its row
        dot product.  The intentionally simple row ownership is a correctness
        baseline for simulator parity, not a real-device performance claim.
        """

        row = cuda.grid(1)
        if row >= output.shape[0]:
            return
        total = np.float32(0.0)
        row_offset = row * row_bytes

        if dtype_id == 1:  # Q8_0, 32 values / 34 bytes
            for block in range(blocks_per_row):
                offset = row_offset + block * 34
                scale = _packed_f16(packed, offset)
                x_offset = block * 32
                for index in range(32):
                    quant = _packed_i8(packed, offset + 2 + index)
                    total += scale * quant * values[x_offset + index]

        elif dtype_id == 2:  # IQ4_NL, 32 values / 18 bytes
            for block in range(blocks_per_row):
                offset = row_offset + block * 18
                scale = _packed_f16(packed, offset)
                x_offset = block * 32
                for index in range(16):
                    quant = int(packed[offset + 2 + index])
                    total += (
                        scale * iq4_values[quant & 0x0F]
                        * values[x_offset + index]
                    )
                    total += (
                        scale * iq4_values[quant >> 4]
                        * values[x_offset + 16 + index]
                    )

        elif dtype_id == 3:  # IQ4_XS, 256 values / 136 bytes
            for block in range(blocks_per_row):
                offset = row_offset + block * 136
                scale = _packed_f16(packed, offset)
                scales_high = _packed_u16(packed, offset + 2)
                x_offset = block * 256
                for subblock in range(8):
                    low_byte = int(packed[offset + 4 + subblock // 2])
                    low = (low_byte >> (4 * (subblock % 2))) & 0x0F
                    high = ((scales_high >> (2 * subblock)) & 0x03) << 4
                    effective = scale * ((low | high) - 32)
                    quant_offset = offset + 8 + subblock * 16
                    output_offset = x_offset + subblock * 32
                    for index in range(16):
                        quant = int(packed[quant_offset + index])
                        total += (
                            effective * iq4_values[quant & 0x0F]
                            * values[output_offset + index]
                        )
                        total += (
                            effective * iq4_values[quant >> 4]
                            * values[output_offset + 16 + index]
                        )

        elif dtype_id == 4:  # IQ3_S, 256 values / 110 bytes
            for block in range(blocks_per_row):
                offset = row_offset + block * 110
                scale = _packed_f16(packed, offset)
                x_offset = block * 256
                for subblock in range(8):
                    packed_scale = int(packed[offset + 106 + subblock // 2])
                    if subblock % 2 == 0:
                        nibble = packed_scale & 0x0F
                    else:
                        nibble = packed_scale >> 4
                    effective = scale * (1 + 2 * nibble)
                    high_bits = int(packed[offset + 66 + subblock])
                    for lane in range(4):
                        quant_offset = offset + 2 + subblock * 8 + 2 * lane
                        index0 = (
                            int(packed[quant_offset])
                            | (((high_bits >> (2 * lane)) & 1) << 8)
                        )
                        index1 = (
                            int(packed[quant_offset + 1])
                            | (((high_bits >> (2 * lane + 1)) & 1) << 8)
                        )
                        grid0 = int(iq3_grid[index0])
                        grid1 = int(iq3_grid[index1])
                        signs = int(packed[offset + 74 + subblock * 4 + lane])
                        output_offset = x_offset + subblock * 32 + lane * 8
                        for coordinate in range(4):
                            magnitude0 = (grid0 >> (8 * coordinate)) & 0xFF
                            magnitude1 = (grid1 >> (8 * coordinate)) & 0xFF
                            sign0 = -1.0 if signs & (1 << coordinate) else 1.0
                            sign1 = -1.0 if signs & (1 << (4 + coordinate)) else 1.0
                            total += (
                                effective * magnitude0 * sign0
                                * values[output_offset + coordinate]
                            )
                            total += (
                                effective * magnitude1 * sign1
                                * values[output_offset + 4 + coordinate]
                            )

        elif dtype_id == 5:  # Q3_K, 256 values / 110 bytes
            for block in range(blocks_per_row):
                offset = row_offset + block * 110
                scale = _packed_f16(packed, offset + 108)
                x_offset = block * 256
                for half in range(2):
                    quant_offset = offset + 32 + half * 32
                    for group in range(4):
                        shift = 2 * group
                        mask = 1 << (half * 4 + group)
                        scale0 = scale * _packed_q3_scale(
                            packed, offset, half * 8 + group * 2,
                        )
                        scale1 = scale * _packed_q3_scale(
                            packed, offset, half * 8 + group * 2 + 1,
                        )
                        output_offset = x_offset + half * 128 + group * 32
                        for lane in range(16):
                            low0 = (
                                int(packed[quant_offset + lane]) >> shift
                            ) & 3
                            low1 = (
                                int(packed[quant_offset + 16 + lane]) >> shift
                            ) & 3
                            high0 = 0 if int(packed[offset + lane]) & mask else 4
                            high1 = (
                                0 if int(packed[offset + 16 + lane]) & mask else 4
                            )
                            total += (
                                scale0 * (low0 - high0)
                                * values[output_offset + lane]
                            )
                            total += (
                                scale1 * (low1 - high1)
                                * values[output_offset + 16 + lane]
                            )

        elif dtype_id == 6:  # Q4_K, 256 values / 144 bytes
            for block in range(blocks_per_row):
                offset = row_offset + block * 144
                scale = _packed_f16(packed, offset)
                minimum = _packed_f16(packed, offset + 2)
                x_offset = block * 256
                for chunk in range(4):
                    scale_index0 = 2 * chunk
                    scale_index1 = scale_index0 + 1
                    d0 = scale * _packed_k_scale(
                        packed, offset, scale_index0, False,
                    )
                    d1 = scale * _packed_k_scale(
                        packed, offset, scale_index1, False,
                    )
                    m0 = minimum * _packed_k_scale(
                        packed, offset, scale_index0, True,
                    )
                    m1 = minimum * _packed_k_scale(
                        packed, offset, scale_index1, True,
                    )
                    quant_offset = offset + 16 + chunk * 32
                    output_offset = x_offset + chunk * 64
                    for lane in range(32):
                        quant = int(packed[quant_offset + lane])
                        total += (
                            (d0 * (quant & 0x0F) - m0)
                            * values[output_offset + lane]
                        )
                        total += (
                            (d1 * (quant >> 4) - m1)
                            * values[output_offset + 32 + lane]
                        )

        elif dtype_id == 7:  # Q5_K, 256 values / 176 bytes
            for block in range(blocks_per_row):
                offset = row_offset + block * 176
                scale = _packed_f16(packed, offset)
                minimum = _packed_f16(packed, offset + 2)
                x_offset = block * 256
                for chunk in range(4):
                    scale_index0 = 2 * chunk
                    scale_index1 = scale_index0 + 1
                    d0 = scale * _packed_k_scale(
                        packed, offset, scale_index0, False,
                    )
                    d1 = scale * _packed_k_scale(
                        packed, offset, scale_index1, False,
                    )
                    m0 = minimum * _packed_k_scale(
                        packed, offset, scale_index0, True,
                    )
                    m1 = minimum * _packed_k_scale(
                        packed, offset, scale_index1, True,
                    )
                    quant_offset = offset + 48 + chunk * 32
                    output_offset = x_offset + chunk * 64
                    high_mask0 = 1 << (2 * chunk)
                    high_mask1 = 1 << (2 * chunk + 1)
                    for lane in range(32):
                        quant = int(packed[quant_offset + lane])
                        high = int(packed[offset + 16 + lane])
                        code0 = (quant & 0x0F) + (
                            16 if high & high_mask0 else 0
                        )
                        code1 = (quant >> 4) + (
                            16 if high & high_mask1 else 0
                        )
                        total += (
                            (d0 * code0 - m0) * values[output_offset + lane]
                        )
                        total += (
                            (d1 * code1 - m1)
                            * values[output_offset + 32 + lane]
                        )

        elif dtype_id == 8:  # Q6_K, 256 values / 210 bytes
            for block in range(blocks_per_row):
                offset = row_offset + block * 210
                scale = _packed_f16(packed, offset + 208)
                x_offset = block * 256
                for half in range(2):
                    low_offset = offset + half * 64
                    high_offset = offset + 128 + half * 32
                    scale_offset = offset + 192 + half * 8
                    output_offset = x_offset + half * 128
                    for lane in range(32):
                        scale_pair = lane // 16
                        low0 = int(packed[low_offset + lane])
                        low1 = int(packed[low_offset + 32 + lane])
                        high = int(packed[high_offset + lane])
                        code0 = (low0 & 0x0F) | ((high & 3) << 4)
                        code1 = (low1 & 0x0F) | (((high >> 2) & 3) << 4)
                        code2 = (low0 >> 4) | (((high >> 4) & 3) << 4)
                        code3 = (low1 >> 4) | (((high >> 6) & 3) << 4)
                        sc0 = _packed_i8(
                            packed, scale_offset + scale_pair,
                        )
                        sc1 = _packed_i8(
                            packed, scale_offset + scale_pair + 2,
                        )
                        sc2 = _packed_i8(
                            packed, scale_offset + scale_pair + 4,
                        )
                        sc3 = _packed_i8(
                            packed, scale_offset + scale_pair + 6,
                        )
                        total += (
                            scale * sc0 * (code0 - 32)
                            * values[output_offset + lane]
                        )
                        total += (
                            scale * sc1 * (code1 - 32)
                            * values[output_offset + 32 + lane]
                        )
                        total += (
                            scale * sc2 * (code2 - 32)
                            * values[output_offset + 64 + lane]
                        )
                        total += (
                            scale * sc3 * (code3 - 32)
                            * values[output_offset + 96 + lane]
                        )

        output[row] = total

    @cuda.jit
    def _state_copy(source, target):
        index = cuda.grid(1)
        if index < source.size:
            target[index] = source[index]

    @cuda.jit
    def _fill_zero(target):
        index = cuda.grid(1)
        if index < target.size:
            target[index] = 0.0

    @cuda.jit
    def _split_query_gate(interleaved, queries, gates):
        index = cuda.grid(1)
        width = queries.shape[1]
        if index < queries.size:
            head = index // width
            column = index - head * width
            stride = 2 * width
            queries[head, column] = interleaved[head * stride + column]
            gates[head, column] = interleaved[
                head * stride + width + column
            ]

    @cuda.jit
    def _rms_norm(source, weight, epsilon, output):
        vector = cuda.grid(1)
        if vector < source.shape[0]:
            total = 0.0
            for column in range(source.shape[1]):
                value = source[vector, column]
                total += value * value
            denominator = np.sqrt(total / source.shape[1] + epsilon)
            for column in range(source.shape[1]):
                output[vector, column] = (
                    source[vector, column] / denominator * weight[column]
                )

    @cuda.jit
    def _residual_add(left, right, output):
        index = cuda.grid(1)
        if index < left.size:
            output[index] = left[index] + right[index]

    @cuda.jit
    def _swiglu(gate, up, output):
        index = cuda.grid(1)
        if index < gate.size:
            value = gate[index]
            if value >= 0.0:
                z = np.exp(-value)
                activated = value / (1.0 + z)
            else:
                z = np.exp(value)
                activated = value * z / (1.0 + z)
            output[index] = activated * up[index]

    @cuda.jit
    def _sigmoid_gate(values, gates, output):
        index = cuda.grid(1)
        if index < values.size:
            gate = gates[index]
            if gate >= 0.0:
                z = np.exp(-gate)
                scale = 1.0 / (1.0 + z)
            else:
                z = np.exp(gate)
                scale = z / (1.0 + z)
            output[index] = values[index] * scale

    @cuda.jit
    def _imrope(
        source, position_lanes, sections, frequency_base, frequency_scale,
        output,
    ):
        head = cuda.grid(1)
        if head < source.shape[0]:
            rotary_pairs = (
                sections[0] + sections[1] + sections[2] + sections[3]
            )
            rotary_dimensions = 2 * rotary_pairs
            for column in range(source.shape[1]):
                output[head, column] = source[head, column]
            for pair in range(rotary_pairs):
                sector = pair % rotary_pairs
                if sector % 3 == 1 and sector < 3 * sections[1]:
                    lane = 1
                elif sector % 3 == 2 and sector < 3 * sections[2]:
                    lane = 2
                elif sector % 3 == 0 and sector < 3 * sections[0]:
                    lane = 0
                else:
                    lane = 3
                theta = (
                    position_lanes[lane] * frequency_scale
                    * frequency_base ** (-2.0 * pair / rotary_dimensions)
                )
                cosine = np.cos(theta)
                sine = np.sin(theta)
                other = pair + rotary_pairs
                first = source[head, pair]
                second = source[head, other]
                output[head, pair] = first * cosine - second * sine
                output[head, other] = first * sine + second * cosine

    @cuda.jit
    def _kv_append(keys, values, key_cache, value_cache, position):
        index = cuda.grid(1)
        width = keys.shape[1]
        if index < keys.size:
            head = index // width
            column = index - head * width
            key_cache[position, head, column] = keys[head, column]
            value_cache[position, head, column] = values[head, column]

    _kernels = {
        "cuda": cuda,
        "np": np,
        "capability": capability,
        "conv_step": _conv_step,
        "conv_step_out": _conv_step_out,
        "conv_raw_observe": _conv_raw_observe,
        "normalize_heads": _normalize_heads,
        "beta_decay": _beta_decay,
        "gdn_step": _gdn_step,
        "gdn_step_out": _gdn_step_out,
        "gated_rms": _gated_rms,
        "attention_decode": _attention_decode,
        "packed_f32_matvec": _packed_f32_matvec,
        "packed_quant_matvec": _packed_quant_matvec,
        "packed_f32_row": _packed_f32_row,
        "packed_quant_row": _packed_quant_row,
        "packed_f32_matmul": _packed_f32_matmul,
        "packed_quant_matmul": _packed_quant_matmul,
        "packed_iq4_values": None,
        "packed_iq3_grid": None,
        "state_copy": _state_copy,
        "fill_zero": _fill_zero,
        "split_query_gate": _split_query_gate,
        "rms_norm": _rms_norm,
        "residual_add": _residual_add,
        "swiglu": _swiglu,
        "sigmoid_gate": _sigmoid_gate,
        "imrope": _imrope,
        "kv_append": _kv_append,
    }
    return _kernels


def primitive_roundtrip(
    name: str,
    *arrays,
    scalars: tuple[float, ...] = (),
    allow_simulator: bool = False,
):
    """Small host launcher for primitive parity tests, never model execution."""

    kernels = cuda_kernels(allow_simulator=allow_simulator)
    cuda = kernels["cuda"]
    np = kernels["np"]
    host = [np.ascontiguousarray(item, dtype=np.float32) for item in arrays]
    device = [cuda.to_device(item) for item in host]
    threads = 64
    if name == "conv_step":
        output = cuda.device_array(host[0].shape, dtype=np.float32)
        kernels[name][(host[0].size + threads - 1) // threads, threads](
            device[0], device[1], device[2], output,
        )
        return output.copy_to_host(), device[2].copy_to_host()
    if name == "normalize_heads":
        output = cuda.device_array(host[0].shape, dtype=np.float32)
        kernels[name][(host[0].shape[0] + threads - 1) // threads, threads](
            device[0], output, float(scalars[0]),
        )
        return output.copy_to_host()
    if name == "beta_decay":
        beta = cuda.device_array(host[0].shape, dtype=np.float32)
        decay = cuda.device_array(host[0].shape, dtype=np.float32)
        kernels[name][1, threads](*device, beta, decay)
        return beta.copy_to_host(), decay.copy_to_host()
    if name == "gdn_step":
        output = cuda.device_array(host[3].shape, dtype=np.float32)
        kernels[name][(host[0].shape[0] + threads - 1) // threads, threads](
            *device, output,
        )
        return output.copy_to_host(), device[0].copy_to_host()
    if name == "gated_rms":
        output = cuda.device_array(host[0].shape, dtype=np.float32)
        kernels[name][1, threads](*device, float(scalars[0]), output)
        return output.copy_to_host()
    if name == "attention_decode":
        output = cuda.to_device(np.zeros(
            (host[0].shape[0], host[2].shape[2]), dtype=np.float32,
        ))
        kernels[name][1, threads](*device, output)
        return output.copy_to_host()
    if name == "state_copy":
        shape = host[0].shape
        flat = np.ascontiguousarray(host[0].reshape(-1))
        source = cuda.to_device(flat)
        target = cuda.device_array(flat.shape, dtype=np.float32)
        kernels[name][(flat.size + threads - 1) // threads, threads](
            source, target,
        )
        return target.copy_to_host().reshape(shape)
    if name == "rms_norm":
        source = host[0]
        if source.ndim == 1:
            source = source.reshape(1, -1)
        if source.ndim != 2 or host[1].shape != (source.shape[1],):
            raise ValueError("RMSNorm expects vectors and one matching weight")
        output = cuda.device_array(source.shape, dtype=np.float32)
        kernels[name][
            (source.shape[0] + threads - 1) // threads, threads
        ](
            cuda.to_device(source), device[1], float(scalars[0]), output,
        )
        return output.copy_to_host().reshape(host[0].shape)
    if name in ("residual_add", "swiglu", "sigmoid_gate"):
        if host[0].shape != host[1].shape:
            raise ValueError(f"{name} inputs must have identical shapes")
        shape = host[0].shape
        left = np.ascontiguousarray(host[0].reshape(-1))
        right = np.ascontiguousarray(host[1].reshape(-1))
        output = cuda.device_array(left.shape, dtype=np.float32)
        kernels[name][(left.size + threads - 1) // threads, threads](
            cuda.to_device(left), cuda.to_device(right), output,
        )
        return output.copy_to_host().reshape(shape)
    if name == "imrope":
        source = host[0]
        if source.ndim != 2:
            raise ValueError("IMROPE source must have shape [heads, width]")
        position_lanes = np.ascontiguousarray(arrays[1], dtype=np.int32)
        sections = np.ascontiguousarray(arrays[2], dtype=np.int32)
        if position_lanes.shape != (4,) or sections.shape != (4,):
            raise ValueError("IMROPE requires four position lanes and sections")
        if source.shape[1] < 2 * int(sections.sum()):
            raise ValueError("IMROPE source is shorter than its rotary dimensions")
        output = cuda.device_array(source.shape, dtype=np.float32)
        kernels[name][
            (source.shape[0] + threads - 1) // threads, threads
        ](
            device[0], cuda.to_device(position_lanes), cuda.to_device(sections),
            float(scalars[0]), float(scalars[1]), output,
        )
        return output.copy_to_host()
    if name == "kv_append":
        position = int(scalars[0])
        if host[0].ndim != 2 or host[0].shape != host[1].shape:
            raise ValueError("current K/V must have identical [heads, width] shapes")
        if (
            host[2].shape != host[3].shape
            or host[2].shape[1:] != host[0].shape
        ):
            raise ValueError("K/V caches must match current K/V geometry")
        if position < 0 or position >= host[2].shape[0]:
            raise ValueError("K/V append position is outside the cache")
        kernels[name][(host[0].size + threads - 1) // threads, threads](
            *device, position,
        )
        return device[2].copy_to_host(), device[3].copy_to_host()
    raise ValueError(f"unknown CUDA primitive {name!r}")


class PackedCudaMatrix:
    """One immutable GGUF matrix kept in its native packed device layout.

    Construction uploads exactly ``packed_bytes`` for the weights.  Quantized
    codes are decoded inside the row-dot kernel and are never widened into a
    full code or float matrix.  This class is low-level kernel plumbing; its
    implemented dtype inventory is simulator-tested, not real-device
    qualified.
    """

    __slots__ = (
        "name", "dtype", "rows", "columns", "packed_bytes",
        "device_packed_bytes", "simulator", "_kernels", "_device_weights",
        "_closed",
    )

    def __init__(
        self,
        packed: bytes | bytearray | memoryview,
        dtype: str,
        *,
        rows: int,
        columns: int,
        name: str = "packed-matrix",
        allow_simulator: bool = False,
    ) -> None:
        if dtype not in CUDA_PACKED_IMPLEMENTED_DTYPES:
            supported = ", ".join(sorted(CUDA_PACKED_IMPLEMENTED_DTYPES))
            raise ValueError(
                f"packed CUDA matvec does not implement {dtype}; supported: {supported}"
            )
        if rows <= 0 or columns <= 0:
            raise ValueError("packed CUDA matrix dimensions must be positive")
        block_elements, block_bytes = GGML_BLOCK_INFO[dtype]
        if columns % block_elements:
            raise ValueError(
                f"{dtype} matrix columns {columns} are not divisible by "
                f"block size {block_elements}"
            )
        expected = rows * (columns // block_elements) * block_bytes
        if len(packed) != expected:
            raise ValueError(
                f"{name} has {len(packed)} packed bytes, expected {expected} "
                f"for {rows}x{columns} {dtype}"
            )

        kernels = cuda_kernels(allow_simulator=allow_simulator)
        cuda_module = kernels["cuda"]
        np = kernels["np"]
        if dtype == "F32":
            host = np.frombuffer(packed, dtype="<f4", count=rows * columns)
        else:
            host = np.frombuffer(packed, dtype=np.uint8, count=expected)
        # to_device performs the packed upload synchronously.  No host ndarray
        # or mmap-export is retained by the matrix after construction.
        device_weights = cuda_module.to_device(host)
        del host

        self.name = name
        self.dtype = dtype
        self.rows = rows
        self.columns = columns
        self.packed_bytes = expected
        self.device_packed_bytes = int(device_weights.nbytes)
        self.simulator = bool(kernels["capability"].simulator)
        self._kernels = kernels
        self._device_weights = device_weights
        self._closed = False

    @classmethod
    def from_source(
        cls,
        source: Any,
        tensor_name: str,
        *,
        allow_simulator: bool = False,
    ) -> "PackedCudaMatrix":
        """Upload one matrix from ``PackedTensorSource`` without decoding it."""

        source._ensure_open()
        try:
            spec = source.specs[tensor_name]
        except KeyError:
            raise KeyError(f"GGUF has no packed tensor named {tensor_name!r}") from None
        if len(spec.shape) != 2:
            raise ValueError(f"tensor {tensor_name} has shape {spec.shape}, expected matrix")
        payload = source.packed_bytes(tensor_name)
        return cls(
            payload,
            spec.dtype,
            rows=spec.rows,
            columns=spec.columns,
            name=tensor_name,
            allow_simulator=allow_simulator,
        )

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed or self._device_weights is None:
            raise RuntimeError(f"packed CUDA matrix {self.name} is closed")

    def _quant_tables(self):
        kernels = self._kernels
        cuda_module = kernels["cuda"]
        np = kernels["np"]
        if kernels["packed_iq4_values"] is None:
            kernels["packed_iq4_values"] = cuda_module.to_device(
                np.asarray(_IQ4_NL_VALUES, dtype=np.float32)
            )
        if kernels["packed_iq3_grid"] is None:
            kernels["packed_iq3_grid"] = cuda_module.to_device(
                np.asarray(_IQ3S_GRID, dtype=np.uint32)
            )
        return kernels["packed_iq4_values"], kernels["packed_iq3_grid"]

    def _quant_geometry(self):
        block_elements, block_bytes = GGML_BLOCK_INFO[self.dtype]
        blocks_per_row = self.columns // block_elements
        return blocks_per_row, blocks_per_row * block_bytes

    def _device_array(
        self,
        value: Any,
        shape: tuple[int, ...],
        what: str,
    ) -> None:
        """Validate the small device-array contract without a host copy.

        Numba's simulator arrays do not expose ``__cuda_array_interface__``;
        both simulator and real device arrays do expose ``copy_to_host``.
        The method deliberately accepts strided device views.
        """

        np = self._kernels["np"]
        if not hasattr(value, "copy_to_host"):
            raise TypeError(f"{what} must be a CUDA device array")
        if tuple(getattr(value, "shape", ())) != shape:
            raise ValueError(
                f"{what} has shape {getattr(value, 'shape', None)}, "
                f"expected {shape}"
            )
        if getattr(value, "dtype", None) != np.dtype(np.float32):
            raise TypeError(f"{what} must use float32")

    def f32_device_view(self):
        """Return the native F32 matrix as ``[rows, columns]`` on device.

        This is used for the depthwise convolution tensor, which is already
        dense F32 in GGUF.  Quantized tensors are never widened through this
        API.
        """

        self._ensure_open()
        if self.dtype != "F32":
            raise TypeError(f"{self.name} is {self.dtype}, not native F32")
        return self._device_weights.reshape(self.rows, self.columns)

    def matvec_device(self, values_device: Any, output_device: Any | None = None):
        """Launch a direct packed matvec with device input and output."""

        self._ensure_open()
        self._device_array(
            values_device, (self.columns,), f"{self.name} matvec input",
        )
        kernels = self._kernels
        cuda_module = kernels["cuda"]
        np = kernels["np"]
        if output_device is None:
            output_device = cuda_module.device_array(self.rows, dtype=np.float32)
        else:
            self._device_array(
                output_device, (self.rows,), f"{self.name} matvec output",
            )
        threads = 64
        blocks = (self.rows + threads - 1) // threads
        if self.dtype == "F32":
            kernels["packed_f32_matvec"][blocks, threads](
                self._device_weights, values_device, output_device, self.columns,
            )
        else:
            iq4_values, iq3_grid = self._quant_tables()
            blocks_per_row, row_bytes = self._quant_geometry()
            kernels["packed_quant_matvec"][blocks, threads](
                self._device_weights,
                values_device,
                output_device,
                row_bytes,
                blocks_per_row,
                _PACKED_DTYPE_IDS[self.dtype],
                iq4_values,
                iq3_grid,
            )
        return output_device

    def matvec(self, values: Sequence[float]):
        """Return an owned F32 host vector after a direct packed row-dot."""

        self._ensure_open()
        if len(values) != self.columns:
            raise ValueError(
                f"{self.name} expects {self.columns} matvec values, got {len(values)}"
            )
        kernels = self._kernels
        cuda_module = kernels["cuda"]
        np = kernels["np"]
        host_values = np.ascontiguousarray(values, dtype=np.float32)
        device_values = cuda_module.to_device(host_values)
        return self.matvec_device(device_values).copy_to_host()

    def embedding_row_device(
        self, row_index: int, output_device: Any | None = None,
    ):
        """Decode exactly one native packed row to a device F32 vector."""

        self._ensure_open()
        if row_index < 0:
            row_index += self.rows
        if row_index < 0 or row_index >= self.rows:
            raise IndexError(f"row {row_index} is out of range for {self.name}")
        kernels = self._kernels
        cuda_module = kernels["cuda"]
        np = kernels["np"]
        if output_device is None:
            output_device = cuda_module.device_array(
                self.columns, dtype=np.float32,
            )
        else:
            self._device_array(
                output_device,
                (self.columns,),
                f"{self.name} embedding-row output",
            )
        threads = 64
        blocks = (self.columns + threads - 1) // threads
        if self.dtype == "F32":
            kernels["packed_f32_row"][blocks, threads](
                self._device_weights, output_device, row_index, self.columns,
            )
        else:
            iq4_values, iq3_grid = self._quant_tables()
            _, row_bytes = self._quant_geometry()
            kernels["packed_quant_row"][blocks, threads](
                self._device_weights,
                output_device,
                row_index * row_bytes,
                _PACKED_DTYPE_IDS[self.dtype],
                iq4_values,
                iq3_grid,
            )
        return output_device

    def embedding_row(self, row_index: int):
        """Decode exactly one native packed row to an owned F32 vector."""
        return self.embedding_row_device(row_index).copy_to_host()

    def matmul_device(self, values_device: Any, output_device: Any | None = None):
        """Launch a direct packed small-batch projection on device arrays."""

        self._ensure_open()
        shape = tuple(getattr(values_device, "shape", ()))
        if len(shape) != 2:
            raise ValueError(f"{self.name} packed matmul expects a rank-2 input")
        batch = shape[0]
        if batch <= 0:
            raise ValueError(f"{self.name} packed matmul batch cannot be empty")
        self._device_array(
            values_device, (batch, self.columns), f"{self.name} matmul input",
        )
        kernels = self._kernels
        cuda_module = kernels["cuda"]
        np = kernels["np"]
        output_shape = (batch, self.rows)
        if output_device is None:
            output_device = cuda_module.device_array(
                output_shape, dtype=np.float32,
            )
        else:
            self._device_array(
                output_device, output_shape, f"{self.name} matmul output",
            )
        threads = 64
        blocks = (output_device.size + threads - 1) // threads
        if self.dtype == "F32":
            kernels["packed_f32_matmul"][blocks, threads](
                self._device_weights, values_device, output_device, self.columns,
            )
        else:
            iq4_values, iq3_grid = self._quant_tables()
            _, row_bytes = self._quant_geometry()
            kernels["packed_quant_matmul"][blocks, threads](
                self._device_weights,
                values_device,
                output_device,
                row_bytes,
                _PACKED_DTYPE_IDS[self.dtype],
                iq4_values,
                iq3_grid,
            )
        return output_device

    def matmul(self, values: Sequence[Sequence[float]]):
        """Direct packed small-batch projection, returning ``[batch, rows]``."""

        self._ensure_open()
        kernels = self._kernels
        cuda_module = kernels["cuda"]
        np = kernels["np"]
        host_values = np.asarray(values, dtype=np.float32)
        if host_values.ndim != 2:
            raise ValueError(f"{self.name} packed matmul expects a rank-2 input")
        if host_values.shape[0] <= 0:
            raise ValueError(f"{self.name} packed matmul batch cannot be empty")
        if host_values.shape[1] != self.columns:
            raise ValueError(
                f"{self.name} expects {self.columns} matmul columns, got "
                f"{host_values.shape[1]}"
            )
        host_values = np.ascontiguousarray(host_values)
        device_values = cuda_module.to_device(host_values)
        return self.matmul_device(device_values).copy_to_host()

    def descriptor(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": [self.columns, self.rows],
            "packed_bytes": self.packed_bytes,
            "device_packed_bytes": self.device_packed_bytes,
            "native_packed": True,
            "simulator": self.simulator,
            "real_device_qualified": self.dtype in CUDA_PACKED_KERNEL_DTYPES,
        }

    def close(self) -> None:
        self._device_weights = None
        self._closed = True

    def __enter__(self) -> "PackedCudaMatrix":
        self._ensure_open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def packed_matvec_roundtrip(
    packed: bytes | bytearray | memoryview,
    dtype: str,
    rows: int,
    columns: int,
    values: Sequence[float],
    *,
    allow_simulator: bool = False,
):
    """One-shot direct packed matvec helper for focused kernel tests."""

    with PackedCudaMatrix(
        packed,
        dtype,
        rows=rows,
        columns=columns,
        allow_simulator=allow_simulator,
    ) as matrix:
        return matrix.matvec(values)


class Qwen35CudaChain:
    """Transactional device-state owner with checkpoint/replay recovery.

    ``executor`` advances a supplied :class:`ModelState`; tests use the CPU
    semantic executor as a stand-in for device-resident math.  Production
    construction is intentionally unavailable until packed projections are
    qualified, preventing a primitive-only simulator from becoming a claimed
    model backend.
    """

    def __init__(
        self,
        model: Any,
        executor: Callable[[int, ModelState], Any],
        *,
        checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
        backend_label: str = "test-device",
    ) -> None:
        if not MIN_CHECKPOINT_INTERVAL <= checkpoint_interval <= MAX_CHECKPOINT_INTERVAL:
            raise ValueError(
                f"checkpoint interval must be {MIN_CHECKPOINT_INTERVAL}.."
                f"{MAX_CHECKPOINT_INTERVAL}"
            )
        self.model = model
        self.executor = executor
        self.checkpoint_interval = checkpoint_interval
        self.backend_label = backend_label
        self.device_state = model.branch()
        self.host_checkpoint = model.snapshot()
        self.recorded_suffix: list[int] = []
        self.host_checkpoint_generation = self.host_checkpoint.generation
        self.device_generation = self.host_checkpoint.generation
        self.dirty_transaction = False
        self.parked = False
        self.fallback_reason = ""
        self.replay_count = 0
        self.checkpoint_count = 0
        self.host_to_device_bytes = 0
        self.device_to_host_bytes = 0
        self.routine_state_host_to_device_bytes = 0
        self.routine_state_device_to_host_bytes = 0
        self.restore_state_host_to_device_bytes = 0
        self.checkpoint_state_device_to_host_bytes = 0
        self.launches = 0
        self.synchronizations = 0

    @classmethod
    def create_production(cls, model: Any, **kwargs):
        capability = cuda_capability()
        if not capability.available:
            raise Qwen35CudaUnavailable(capability.reason)
        if model is None:
            raise Qwen35CudaUnavailable(
                "Qwen35 CUDA qualification requires a validated loaded model"
            )
        from .qwen35_cuda_graph import Qwen35CudaDeviceGraph

        return Qwen35CudaDeviceGraph(model, **kwargs)

    def forward(self, token_id: int):
        if self.parked:
            return self.model.forward(token_id)
        self.dirty_transaction = True
        self.host_to_device_bytes += 8  # token id and logical position only
        self.launches += 1
        try:
            logits = self.executor(token_id, self.device_state)
        except Exception as exc:
            return self._recover_and_continue(token_id, exc)
        self.dirty_transaction = False
        self.device_generation = self.device_state.generation
        self.recorded_suffix.append(token_id)
        # Logits are the routine boundary result; recurrent/KV state stays on
        # the device between checkpoints.
        self.device_to_host_bytes += len(logits) * 4
        self.synchronizations += 1
        if len(self.recorded_suffix) >= self.checkpoint_interval:
            self.checkpoint()
        return logits

    def checkpoint(self) -> StateSnapshot:
        if self.parked:
            # After recovery the CPU model is authoritative.  device_state is
            # deliberately the discarded generation that may have been
            # dirtied by the failed launch; never publish it over host state.
            snapshot = self.model.snapshot()
            self.host_checkpoint = snapshot
            self.host_checkpoint_generation = snapshot.generation
            self.recorded_suffix = []
            return snapshot
        snapshot = self.device_state.snapshot()
        self.model.restore(snapshot)
        self.host_checkpoint = snapshot
        self.host_checkpoint_generation = snapshot.generation
        self.recorded_suffix = []
        self.checkpoint_count += 1
        state_bytes = snapshot.memory_bytes()
        self.device_to_host_bytes += state_bytes
        self.checkpoint_state_device_to_host_bytes += state_bytes
        self.synchronizations += 1
        return snapshot

    def snapshot(self) -> StateSnapshot:
        return self.checkpoint()

    def restore(self, snapshot: StateSnapshot) -> None:
        self.model.restore(snapshot)
        self.device_state = self.model.branch()
        self.host_checkpoint = snapshot
        self.recorded_suffix = []
        self.host_checkpoint_generation = snapshot.generation
        self.device_generation = self.device_state.generation
        self.dirty_transaction = False
        if not self.parked:
            state_bytes = snapshot.memory_bytes()
            self.host_to_device_bytes += state_bytes
            self.restore_state_host_to_device_bytes += state_bytes
            self.synchronizations += 1

    def _recover_and_continue(self, failed_token: int, error: Exception):
        suffix = tuple(self.recorded_suffix)
        self.model.restore(self.host_checkpoint)
        for token in suffix:
            self.model.forward(token)
            self.replay_count += 1
        self.device_state = self.model.branch()
        self.recorded_suffix = []
        self.dirty_transaction = False
        self.parked = True
        self.fallback_reason = f"{type(error).__name__}: {error}"
        return self.model.forward(failed_token)

    def transfer_limits(self) -> dict[str, Any]:
        return {
            "no_recurrent_state_copy_per_token": (
                self.routine_state_host_to_device_bytes == 0
                and self.routine_state_device_to_host_bytes == 0
            ),
            "host_to_device_bytes": self.host_to_device_bytes,
            "device_to_host_bytes": self.device_to_host_bytes,
            "routine_state_host_to_device_bytes": (
                self.routine_state_host_to_device_bytes
            ),
            "routine_state_device_to_host_bytes": (
                self.routine_state_device_to_host_bytes
            ),
            "restore_state_host_to_device_bytes": (
                self.restore_state_host_to_device_bytes
            ),
            "checkpoint_state_device_to_host_bytes": (
                self.checkpoint_state_device_to_host_bytes
            ),
            "launches": self.launches,
            "synchronizations": self.synchronizations,
            "checkpoint_count": self.checkpoint_count,
        }

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.backend_label,
            "parked": self.parked,
            "fallback_reason": self.fallback_reason,
            "checkpoint_interval": self.checkpoint_interval,
            "host_checkpoint_generation": self.host_checkpoint_generation,
            "device_generation": self.device_generation,
            "dirty_transaction": self.dirty_transaction,
            "recorded_suffix_tokens": len(self.recorded_suffix),
            "replay_count": self.replay_count,
            **self.transfer_limits(),
        }


__all__ = [
    "CUDA_PACKED_IMPLEMENTED_DTYPES", "CUDA_PACKED_KERNEL_DTYPES",
    "CudaCapability", "PackedCudaMatrix", "Qwen35CudaChain",
    "Qwen35CudaUnavailable", "cuda_capability", "cuda_kernels",
    "packed_matvec_roundtrip", "primitive_roundtrip",
]
