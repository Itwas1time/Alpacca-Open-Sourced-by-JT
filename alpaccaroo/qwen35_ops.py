# Alpaccaroo - dependency-free Qwen3.5/Qwen3.8 semantic primitives.
# MIT License. See LICENSE.
"""Small, explicit scalar contracts for the Qwen35 hybrid architecture.

This module intentionally uses only the standard library.  It owns the
architecture-specific Gated DeltaNet and IMROPE rules; assembling a recurrent
layer or a complete Qwen35 model belongs in later runtime modules.

The IMROPE layout follows llama.cpp b10507 (commit 95c409c1): position lanes
are selected in an interleaved T/H/W pattern, while each rotated coordinate
pair uses the NeoX half-split layout ``(i, i + n_dims // 2)``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .tensor import (
    l2_normalize as _shared_l2_normalize,
    stable_sigmoid,
    stable_softplus,
)


def silu(value: float) -> float:
    """Return the scalar SiLU activation ``value * sigmoid(value)``."""
    if value == -math.inf:
        return -0.0
    return value * stable_sigmoid(value)


def _positive_finite_epsilon(epsilon: float) -> float:
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(f"epsilon must be positive and finite, got {epsilon!r}")
    return epsilon


def l2_normalize(values: Sequence[float], epsilon: float) -> list[float]:
    """Dependency-neutral list facade over the shared tensor primitive."""

    return [float(value) for value in _shared_l2_normalize(values, epsilon)]


def rms_norm(
    values: Sequence[float],
    weight: Sequence[float],
    epsilon: float,
) -> list[float]:
    """Apply learned RMSNorm using ``sqrt(mean(x*x) + epsilon)``."""
    epsilon = _positive_finite_epsilon(epsilon)
    vector = [float(value) for value in values]
    learned_weight = [float(value) for value in weight]
    if not vector:
        raise ValueError("cannot RMS-normalize an empty vector")
    if len(learned_weight) != len(vector):
        raise ValueError(
            "RMSNorm weight length must equal input length: "
            f"{len(learned_weight)} != {len(vector)}"
        )
    if not all(math.isfinite(value) for value in vector + learned_weight):
        raise ValueError("RMSNorm input and weight must contain only finite values")

    # sqrt(mean(x*x) + epsilon), evaluated relative to a common scale.
    epsilon_root = math.sqrt(epsilon)
    den_scale = max(epsilon_root, max(abs(value) for value in vector))
    scaled_mean_square = math.fsum(
        (value / den_scale) * (value / den_scale) for value in vector
    ) / len(vector)
    scaled_epsilon = (epsilon_root / den_scale) ** 2
    scaled_denominator = math.sqrt(scaled_mean_square + scaled_epsilon)
    return [
        ((value / den_scale) / scaled_denominator) * learned
        for value, learned in zip(vector, learned_weight)
    ]


def gated_rms_norm(
    values: Sequence[float],
    weight: Sequence[float],
    gate: Sequence[float],
    epsilon: float,
) -> list[float]:
    """Apply learned RMSNorm followed by the Qwen recurrent SiLU gate."""

    gated = [float(value) for value in gate]
    normalized = rms_norm(values, weight, epsilon)
    if len(gated) != len(normalized):
        raise ValueError(
            "gated RMSNorm gate length must equal input length: "
            f"{len(gated)} != {len(normalized)}"
        )
    if not all(math.isfinite(value) for value in gated):
        raise ValueError("gated RMSNorm gate must contain only finite values")
    return [
        value * silu(gate_value)
        for value, gate_value in zip(normalized, gated)
    ]


def imrope_text_position(position: int) -> tuple[int, int, int, int]:
    """Return Qwen text position lanes ``[p, p, p, 0]``."""
    if isinstance(position, bool) or not isinstance(position, int):
        raise TypeError("text position must be an integer")
    if position < 0:
        raise ValueError(f"text position must be non-negative, got {position}")
    return (position, position, position, 0)


def imrope_text_positions(
    start: int,
    count: int,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Return four lane-major position arrays for consecutive text tokens."""
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("position count must be an integer")
    if count < 0:
        raise ValueError(f"position count must be non-negative, got {count}")
    first = imrope_text_position(start)[0]
    text = list(range(first, first + count))
    return (list(text), list(text), list(text), [0] * count)


def _imrope_sections(sections: Sequence[int]) -> tuple[int, int, int, int]:
    if len(sections) != 4:
        raise ValueError(f"IMROPE requires four sections, got {len(sections)}")
    result = tuple(sections)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in result):
        raise TypeError("IMROPE sections must be integers")
    if any(value < 0 for value in result):
        raise ValueError(f"IMROPE sections cannot be negative: {result}")
    if sum(result) <= 0:
        raise ValueError("at least one IMROPE section must be nonzero")
    return result


def imrope_lane_for_pair(pair_index: int, sections: Sequence[int]) -> int:
    """Return the T/H/W/E position-lane index for one rotary pair.

    This is the pinned ``GGML_ROPE_TYPE_IMROPE`` schedule.  Lane selection is
    interleaved, not four contiguous section ranges.
    """
    if isinstance(pair_index, bool) or not isinstance(pair_index, int):
        raise TypeError("rotary pair index must be an integer")
    if pair_index < 0:
        raise ValueError(f"rotary pair index must be non-negative, got {pair_index}")
    section = _imrope_sections(sections)
    sector = pair_index % sum(section)
    if sector % 3 == 1 and sector < 3 * section[1]:
        return 1
    if sector % 3 == 2 and sector < 3 * section[2]:
        return 2
    if sector % 3 == 0 and sector < 3 * section[0]:
        return 0
    return 3


def imrope_rotate(
    values: Sequence[float],
    position_lanes: Sequence[int],
    sections: Sequence[int],
    *,
    frequency_base: float,
    frequency_scale: float = 1.0,
) -> list[float]:
    """Rotate one Q/K head with pinned Qwen IMROPE semantics.

    ``2 * sum(sections)`` leading coordinates are rotated.  Coordinate
    ``pair`` is paired with ``pair + n_dims // 2`` (NeoX half-split); any
    trailing head coordinates are copied unchanged.
    """
    vector = [float(value) for value in values]
    lane_positions = tuple(position_lanes)
    section = _imrope_sections(sections)
    if len(lane_positions) != 4:
        raise ValueError(
            f"IMROPE requires four position lanes, got {len(lane_positions)}"
        )
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in lane_positions):
        raise TypeError("IMROPE position lanes must be integers")
    frequency_base = float(frequency_base)
    frequency_scale = float(frequency_scale)
    if not math.isfinite(frequency_base) or frequency_base <= 0.0:
        raise ValueError("IMROPE frequency_base must be positive and finite")
    if not math.isfinite(frequency_scale) or frequency_scale <= 0.0:
        raise ValueError("IMROPE frequency_scale must be positive and finite")

    rotary_pairs = sum(section)
    rotary_dimensions = 2 * rotary_pairs
    if len(vector) < rotary_dimensions:
        raise ValueError(
            "IMROPE input is shorter than 2 * sum(sections): "
            f"{len(vector)} < {rotary_dimensions}"
        )
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("IMROPE input must contain only finite values")

    result = list(vector)
    for pair in range(rotary_pairs):
        lane = imrope_lane_for_pair(pair, section)
        theta = (
            lane_positions[lane]
            * frequency_scale
            * frequency_base ** (-2.0 * pair / rotary_dimensions)
        )
        cosine = math.cos(theta)
        sine = math.sin(theta)
        other = pair + rotary_pairs
        first_value = vector[pair]
        second_value = vector[other]
        result[pair] = first_value * cosine - second_value * sine
        result[other] = first_value * sine + second_value * cosine
    return result


def depthwise_causal_conv_step(
    current: Sequence[float],
    weights: Sequence[Sequence[float]],
    history: Sequence[Sequence[float]],
) -> tuple[list[float], list[list[float]]]:
    """Apply one raw depthwise causal-convolution step.

    ``weights`` has GGUF logical shape ``[kernel, channels]``. ``history`` is
    oldest-to-newest with shape ``[kernel - 1, channels]``.  The returned
    history is a new value and the inputs are never mutated.  SiLU is a
    separate operation, matching the model's decomposed order.
    """
    sample = [float(value) for value in current]
    if not sample:
        raise ValueError("causal convolution requires at least one channel")
    kernel = [[float(value) for value in row] for row in weights]
    previous = [[float(value) for value in row] for row in history]
    if not kernel:
        raise ValueError("causal convolution kernel cannot be empty")
    channels = len(sample)
    if any(len(row) != channels for row in kernel):
        raise ValueError("every convolution weight row must match channel count")
    if len(previous) != len(kernel) - 1:
        raise ValueError(
            "convolution history length must be kernel length - 1: "
            f"{len(previous)} != {len(kernel) - 1}"
        )
    if any(len(row) != channels for row in previous):
        raise ValueError("every convolution history row must match channel count")

    window = previous + [sample]
    output = [
        math.fsum(window[tap][channel] * kernel[tap][channel]
                  for tap in range(len(kernel)))
        for channel in range(channels)
    ]
    new_history = window[1:]
    return output, new_history


def depthwise_causal_conv_scan(
    samples: Sequence[Sequence[float]],
    weights: Sequence[Sequence[float]],
    history: Sequence[Sequence[float]],
) -> tuple[list[list[float]], list[list[float]]]:
    """Sequentially apply the one-token convolution contract to a chunk."""

    current_history = [[float(value) for value in row] for row in history]
    output: list[list[float]] = []
    for sample in samples:
        convolved, current_history = depthwise_causal_conv_step(
            sample, weights, current_history,
        )
        output.append(convolved)
    return output, current_history


def key_head_for_value_head(value_head_index: int, key_head_count: int) -> int:
    """Map a value head to the pinned post-converter interleaved key head."""
    if isinstance(value_head_index, bool) or not isinstance(value_head_index, int):
        raise TypeError("value head index must be an integer")
    if isinstance(key_head_count, bool) or not isinstance(key_head_count, int):
        raise TypeError("key head count must be an integer")
    if value_head_index < 0:
        raise ValueError("value head index must be non-negative")
    if key_head_count <= 0:
        raise ValueError("key head count must be positive")
    return value_head_index % key_head_count


def repeat_key_heads_interleaved(
    key_heads: Sequence[Sequence[float]],
    value_head_count: int,
) -> list[list[float]]:
    """Expand key heads for value heads using ``value_index % key_count``."""
    heads = [[float(value) for value in head] for head in key_heads]
    if not heads:
        raise ValueError("at least one key head is required")
    if isinstance(value_head_count, bool) or not isinstance(value_head_count, int):
        raise TypeError("value head count must be an integer")
    if value_head_count <= 0 or value_head_count % len(heads) != 0:
        raise ValueError("value head count must be a positive multiple of key heads")
    width = len(heads[0])
    if not width or any(len(head) != width for head in heads):
        raise ValueError("key heads must be nonempty vectors of equal width")
    return [
        list(heads[key_head_for_value_head(index, len(heads))])
        for index in range(value_head_count)
    ]


def gated_delta_net_step(
    state: Sequence[Sequence[float]],
    query: Sequence[float],
    key: Sequence[float],
    value: Sequence[float],
    *,
    beta: float,
    log_decay: float,
) -> tuple[list[float], list[list[float]]]:
    """Perform one Gated DeltaNet update for one value head.

    State is explicitly row-major ``[value_dim, key_dim]``.  This function
    returns ``(output, new_state)`` without mutating the prior state.  The
    output uses the plan's ``1 / sqrt(key_dim)`` scale.
    """
    matrix = [[float(item) for item in row] for row in state]
    q = [float(item) for item in query]
    k = [float(item) for item in key]
    v = [float(item) for item in value]
    if not matrix or not matrix[0]:
        raise ValueError("GDN state must have nonzero value and key dimensions")
    value_dimension = len(matrix)
    key_dimension = len(matrix[0])
    if any(len(row) != key_dimension for row in matrix):
        raise ValueError("GDN state rows must all have the same key dimension")
    if len(q) != key_dimension or len(k) != key_dimension:
        raise ValueError("GDN query/key lengths must equal the state key dimension")
    if len(v) != value_dimension:
        raise ValueError("GDN value length must equal the state value dimension")
    beta = float(beta)
    log_decay = float(log_decay)
    if not math.isfinite(beta) or beta < 0.0 or beta > 1.0:
        raise ValueError(f"GDN beta must be finite and in [0, 1], got {beta!r}")
    if not math.isfinite(log_decay) or log_decay > 0.0:
        raise ValueError(
            "GDN log_decay must be finite and non-positive for converted ssm_a"
        )

    decay = math.exp(log_decay)
    decayed = [[item * decay for item in row] for row in matrix]
    prediction = [
        math.fsum(row[column] * k[column] for column in range(key_dimension))
        for row in decayed
    ]
    delta = [
        beta * (v[row] - prediction[row])
        for row in range(value_dimension)
    ]
    new_state = [
        [
            decayed[row][column] + delta[row] * k[column]
            for column in range(key_dimension)
        ]
        for row in range(value_dimension)
    ]
    scale = 1.0 / math.sqrt(key_dimension)
    output = [
        math.fsum(row[column] * q[column] for column in range(key_dimension))
        * scale
        for row in new_state
    ]
    return output, new_state


def gated_delta_net_scan(
    state: Sequence[Sequence[float]],
    queries: Sequence[Sequence[float]],
    keys: Sequence[Sequence[float]],
    values: Sequence[Sequence[float]],
    betas: Sequence[float],
    log_decays: Sequence[float],
) -> tuple[list[list[float]], list[list[float]]]:
    """Causally scan a token chunk through one value head's GDN state."""

    count = len(queries)
    if not (
        len(keys) == count
        and len(values) == count
        and len(betas) == count
        and len(log_decays) == count
    ):
        raise ValueError("GDN scan token-axis lengths must agree")
    current = [[float(item) for item in row] for row in state]
    outputs: list[list[float]] = []
    for index in range(count):
        output, current = gated_delta_net_step(
            current,
            queries[index],
            keys[index],
            values[index],
            beta=betas[index],
            log_decay=log_decays[index],
        )
        outputs.append(output)
    return outputs, current


def causal_gqa_step(
    query_heads: Sequence[Sequence[float]],
    key_cache: Sequence[Sequence[Sequence[float]]],
    value_cache: Sequence[Sequence[Sequence[float]]],
    *,
    scale: float | None = None,
) -> list[list[float]]:
    """Pure causal GQA for one query token and an already-appended cache.

    Full-attention query heads use ordinary contiguous grouping: head ``h``
    maps to KV head ``h // (query_heads / kv_heads)``.  This intentionally
    differs from recurrent DeltaNet's converter-tiled modulo mapping.
    """

    queries = [[float(value) for value in head] for head in query_heads]
    keys = [
        [[float(value) for value in head] for head in token]
        for token in key_cache
    ]
    values = [
        [[float(value) for value in head] for head in token]
        for token in value_cache
    ]
    if not queries or not keys or len(keys) != len(values):
        raise ValueError("GQA requires queries and equal nonempty K/V caches")
    kv_heads = len(keys[0])
    if kv_heads <= 0 or len(queries) % kv_heads:
        raise ValueError("query heads must be a multiple of KV heads")
    if any(len(token) != kv_heads for token in keys + values):
        raise ValueError("every GQA cache token must have the same KV head count")
    key_dimension = len(queries[0])
    if key_dimension <= 0 or any(len(head) != key_dimension for head in queries):
        raise ValueError("GQA query heads must be nonempty and equally sized")
    if any(len(head) != key_dimension for token in keys for head in token):
        raise ValueError("GQA key heads must match the query head dimension")
    value_dimension = len(values[0][0])
    if value_dimension <= 0 or any(
        len(head) != value_dimension for token in values for head in token
    ):
        raise ValueError("GQA value heads must be nonempty and equally sized")
    if scale is None:
        scale = 1.0 / math.sqrt(key_dimension)
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("GQA scale must be positive and finite")

    group = len(queries) // kv_heads
    result: list[list[float]] = []
    for query_index, query in enumerate(queries):
        kv_index = query_index // group
        scores = [
            math.fsum(a * b for a, b in zip(query, token[kv_index])) * scale
            for token in keys
        ]
        maximum = max(scores)
        exponentials = [math.exp(score - maximum) for score in scores]
        denominator = math.fsum(exponentials)
        probabilities = [value / denominator for value in exponentials]
        result.append([
            math.fsum(
                probabilities[token_index]
                * values[token_index][kv_index][column]
                for token_index in range(len(values))
            )
            for column in range(value_dimension)
        ])
    return result


__all__ = [
    "causal_gqa_step",
    "depthwise_causal_conv_scan",
    "depthwise_causal_conv_step",
    "gated_delta_net_scan",
    "gated_delta_net_step",
    "gated_rms_norm",
    "imrope_lane_for_pair",
    "imrope_rotate",
    "imrope_text_position",
    "imrope_text_positions",
    "key_head_for_value_head",
    "l2_normalize",
    "repeat_key_heads_interleaved",
    "rms_norm",
    "silu",
    "stable_sigmoid",
    "stable_softplus",
]
