# Alpaccaroo - optional pinned-Numba kernels for Qwen35 hybrid state.
# MIT License. See LICENSE.
"""Architecture-specific CPU kernels under the scalar Qwen35 contracts.

The module is inert unless the repository's pinned ``numba==0.65.1`` is
available.  Every compiled function uses ``njit`` (never object mode), and
the public wrappers retain a NumPy implementation as the independently
testable fallback.  Token scans stay sequential; independent heads are the
only parallel state owners.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any

NUMBA_PIN = "0.65.1"

_state: dict[str, Any] | None = None


def _init() -> dict[str, Any]:
    global _state
    if _state is not None:
        return _state
    _state = {}
    mode = os.environ.get("ALPACCAROO_QWEN35_NUMBA", "").strip().lower()
    if mode in ("0", "off", "no") or os.environ.get("ALPACCAROO_PURE"):
        return _state
    try:
        import numpy as np
        import numba
        from numba import njit, prange
    except Exception:
        return _state
    if numba.__version__ != NUMBA_PIN and mode != "force":
        print(
            f"alpaccaroo: numba {numba.__version__} != pinned {NUMBA_PIN}; "
            "Qwen35 kernels disabled (ALPACCAROO_QWEN35_NUMBA=force to override)",
            file=sys.stderr,
        )
        return _state

    threads = os.environ.get("ALPACCAROO_THREADS", "").strip()
    if threads:
        try:
            numba.set_num_threads(max(1, int(threads)))
        except (ValueError, RuntimeError):
            pass

    @njit(cache=True)
    def _conv_step(current, weights, history):
        channels = current.shape[0]
        kernel = weights.shape[1]
        output = np.empty(channels, np.float32)
        for channel in range(channels):
            total = np.float32(0.0)
            for tap in range(kernel - 1):
                total += history[tap, channel] * weights[channel, tap]
            total += current[channel] * weights[channel, kernel - 1]
            output[channel] = total / (np.float32(1.0) + np.exp(-total))
        if kernel > 1:
            for tap in range(kernel - 2):
                history[tap, :] = history[tap + 1, :]
            history[kernel - 2, :] = current
        return output

    @njit(cache=True)
    def _normalize_qk(queries, keys, epsilon):
        q_out = np.empty_like(queries)
        k_out = np.empty_like(keys)
        for head in range(queries.shape[0]):
            total = np.float32(0.0)
            for column in range(queries.shape[1]):
                total += queries[head, column] * queries[head, column]
            denominator = max(np.sqrt(total), epsilon)
            for column in range(queries.shape[1]):
                q_out[head, column] = queries[head, column] / denominator
        for head in range(keys.shape[0]):
            total = np.float32(0.0)
            for column in range(keys.shape[1]):
                total += keys[head, column] * keys[head, column]
            denominator = max(np.sqrt(total), epsilon)
            for column in range(keys.shape[1]):
                k_out[head, column] = keys[head, column] / denominator
        return q_out, k_out

    @njit(cache=True)
    def _beta_decay(beta_logits, alpha_logits, dt_bias, ssm_a):
        count = beta_logits.shape[0]
        beta = np.empty(count, np.float32)
        log_decay = np.empty(count, np.float32)
        for head in range(count):
            b = beta_logits[head]
            if b >= 0.0:
                z = np.exp(-b)
                beta[head] = np.float32(1.0) / (np.float32(1.0) + z)
            else:
                z = np.exp(b)
                beta[head] = z / (np.float32(1.0) + z)
            x = alpha_logits[head] + dt_bias[head]
            if x > 0.0:
                dt = x + np.log1p(np.exp(-x))
            else:
                dt = np.log1p(np.exp(x))
            log_decay[head] = dt * ssm_a[head]
        return beta, log_decay

    @njit(parallel=True, cache=True)
    def _gdn_step(state, queries, keys, values, beta, log_decay):
        value_heads, value_dim, key_dim = state.shape
        key_heads = keys.shape[0]
        output = np.empty((value_heads, value_dim), np.float32)
        scale = np.float32(1.0 / np.sqrt(key_dim))
        for head in prange(value_heads):
            key_head = head % key_heads
            decay = np.exp(log_decay[head])
            # Decay and prediction are fused into the first state traversal.
            for row in range(value_dim):
                prediction = np.float32(0.0)
                for column in range(key_dim):
                    decayed = state[head, row, column] * decay
                    state[head, row, column] = decayed
                    prediction += decayed * keys[key_head, column]
                correction = beta[head] * (values[head, row] - prediction)
                for column in range(key_dim):
                    state[head, row, column] += (
                        correction * keys[key_head, column]
                    )
            for row in range(value_dim):
                total = np.float32(0.0)
                for column in range(key_dim):
                    total += (
                        state[head, row, column] * queries[key_head, column]
                    )
                output[head, row] = total * scale
        return output

    @njit(parallel=True, cache=True)
    def _gated_rms(values, weight, gates, epsilon):
        output = np.empty_like(values)
        for head in prange(values.shape[0]):
            total = np.float32(0.0)
            for column in range(values.shape[1]):
                total += values[head, column] * values[head, column]
            denominator = np.sqrt(total / values.shape[1] + epsilon)
            for column in range(values.shape[1]):
                gate = gates[head, column]
                silu_gate = gate / (np.float32(1.0) + np.exp(-gate))
                output[head, column] = (
                    values[head, column] / denominator
                    * weight[column] * silu_gate
                )
        return output

    @njit(parallel=True, cache=True)
    def _attention_decode(queries, keys, values):
        query_heads, key_dim = queries.shape
        positions, kv_heads, value_dim = values.shape
        group = query_heads // kv_heads
        scale = np.float32(1.0 / np.sqrt(key_dim))
        output = np.zeros((query_heads, value_dim), np.float32)
        for query_head in prange(query_heads):
            kv_head = query_head // group
            maximum = np.float32(-np.inf)
            denominator = np.float32(0.0)
            for position in range(positions):
                score = np.float32(0.0)
                for column in range(key_dim):
                    score += (
                        queries[query_head, column]
                        * keys[position, kv_head, column]
                    )
                score *= scale
                next_maximum = max(maximum, score)
                old_scale = np.exp(maximum - next_maximum)
                new_scale = np.exp(score - next_maximum)
                for column in range(value_dim):
                    output[query_head, column] = (
                        output[query_head, column] * old_scale
                        + values[position, kv_head, column] * new_scale
                    )
                denominator = denominator * old_scale + new_scale
                maximum = next_maximum
            for column in range(value_dim):
                output[query_head, column] /= denominator
        return output

    @njit(cache=True)
    def _attention_prefill(queries, keys, values, prefix_length):
        tokens, query_heads, key_dim = queries.shape
        kv_heads = keys.shape[1]
        value_dim = values.shape[2]
        group = query_heads // kv_heads
        scale = np.float32(1.0 / np.sqrt(key_dim))
        output = np.zeros((tokens, query_heads, value_dim), np.float32)
        for token in range(tokens):
            limit = prefix_length + token + 1
            for query_head in range(query_heads):
                kv_head = query_head // group
                maximum = np.float32(-np.inf)
                denominator = np.float32(0.0)
                for position in range(limit):
                    score = np.float32(0.0)
                    for column in range(key_dim):
                        score += (
                            queries[token, query_head, column]
                            * keys[position, kv_head, column]
                        )
                    score *= scale
                    next_maximum = max(maximum, score)
                    old_scale = np.exp(maximum - next_maximum)
                    new_scale = np.exp(score - next_maximum)
                    for column in range(value_dim):
                        output[token, query_head, column] = (
                            output[token, query_head, column] * old_scale
                            + values[position, kv_head, column] * new_scale
                        )
                    denominator = denominator * old_scale + new_scale
                    maximum = next_maximum
                for column in range(value_dim):
                    output[token, query_head, column] /= denominator
        return output

    @njit(cache=True)
    def _state_copy_restore(source, target):
        target[:] = source

    _state.update({
        "np": np,
        "numba": numba,
        "conv_step": _conv_step,
        "normalize_qk": _normalize_qk,
        "beta_decay": _beta_decay,
        "gdn_step": _gdn_step,
        "gated_rms": _gated_rms,
        "attention_decode": _attention_decode,
        "attention_prefill": _attention_prefill,
        "state_copy_restore": _state_copy_restore,
    })
    return _state


def available() -> bool:
    return bool(_init())


def backend_name() -> str:
    state = _init()
    return f"numba=={state['numba'].__version__}" if state else "numpy"


def _np_f32(values):
    import numpy as np

    return np.ascontiguousarray(values, dtype=np.float32)


def qwen35_conv_step(current, weights, history):
    current = _np_f32(current)
    weights = _np_f32(weights)
    history = _np_f32(history)
    state = _init()
    if state:
        return state["conv_step"](current, weights, history), history
    window = __import__("numpy").concatenate((history, current[None, :]), axis=0)
    output = (window.T * weights).sum(axis=1, dtype="float32")
    output = output / (1.0 + __import__("numpy").exp(-output))
    if history.shape[0]:
        history[:-1] = history[1:]
        history[-1] = current
    return _np_f32(output), history


def qwen35_normalize_qk(queries, keys, epsilon: float):
    queries = _np_f32(queries)
    keys = _np_f32(keys)
    state = _init()
    if state:
        return state["normalize_qk"](queries, keys, float(epsilon))
    import numpy as np

    qd = np.maximum(np.sqrt(np.sum(queries * queries, axis=1)), epsilon)
    kd = np.maximum(np.sqrt(np.sum(keys * keys, axis=1)), epsilon)
    return queries / qd[:, None], keys / kd[:, None]


def qwen35_beta_decay(beta_logits, alpha_logits, dt_bias, ssm_a):
    args = tuple(_np_f32(item) for item in (
        beta_logits, alpha_logits, dt_bias, ssm_a,
    ))
    state = _init()
    if state:
        return state["beta_decay"](*args)
    import numpy as np

    beta_logits, alpha_logits, dt_bias, ssm_a = args
    beta = np.where(
        beta_logits >= 0,
        1.0 / (1.0 + np.exp(-beta_logits)),
        np.exp(beta_logits) / (1.0 + np.exp(beta_logits)),
    )
    x = alpha_logits + dt_bias
    dt = np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))
    return _np_f32(beta), _np_f32(dt * ssm_a)


def qwen35_gdn_step(state_matrix, queries, keys, values, beta, log_decay):
    state_matrix = _np_f32(state_matrix)
    args = tuple(_np_f32(item) for item in (
        queries, keys, values, beta, log_decay,
    ))
    compiled = _init()
    if compiled:
        return compiled["gdn_step"](state_matrix, *args)
    import numpy as np

    queries, keys, values, beta, log_decay = args
    output = np.empty_like(values)
    for head in range(state_matrix.shape[0]):
        key_head = head % keys.shape[0]
        state_matrix[head] *= np.exp(log_decay[head])
        prediction = state_matrix[head] @ keys[key_head]
        correction = beta[head] * (values[head] - prediction)
        state_matrix[head] += correction[:, None] * keys[key_head][None, :]
        output[head] = (
            state_matrix[head] @ queries[key_head]
            / math.sqrt(state_matrix.shape[2])
        )
    return output


def qwen35_gated_rms(values, weight, gates, epsilon: float):
    values, weight, gates = (_np_f32(item) for item in (values, weight, gates))
    state = _init()
    if state:
        return state["gated_rms"](values, weight, gates, float(epsilon))
    import numpy as np

    denominator = np.sqrt(np.mean(values * values, axis=1) + epsilon)
    return _np_f32(
        values / denominator[:, None] * weight[None, :]
        * (gates / (1.0 + np.exp(-gates)))
    )


def qwen35_recurrent_layer(
    state_matrix, queries, keys, values, beta_logits, alpha_logits,
    dt_bias, ssm_a, norm_weight, gates, epsilon: float,
):
    queries, keys = qwen35_normalize_qk(queries, keys, epsilon)
    beta, log_decay = qwen35_beta_decay(
        beta_logits, alpha_logits, dt_bias, ssm_a,
    )
    output = qwen35_gdn_step(
        state_matrix, queries, keys, values, beta, log_decay,
    )
    return qwen35_gated_rms(output, norm_weight, gates, epsilon), queries, keys


def qwen35_full_attention_decode(queries, keys, values):
    args = tuple(_np_f32(item) for item in (queries, keys, values))
    state = _init()
    if state:
        return state["attention_decode"](*args)
    # The same online recurrence as the JIT function keeps this fallback from
    # materializing a complete score matrix.
    import numpy as np

    queries, keys, values = args
    output = np.zeros((queries.shape[0], values.shape[2]), np.float32)
    group = queries.shape[0] // keys.shape[1]
    scale = 1.0 / math.sqrt(queries.shape[1])
    for query_head in range(queries.shape[0]):
        kv_head = query_head // group
        maximum = -math.inf
        denominator = 0.0
        for position in range(keys.shape[0]):
            score = float(queries[query_head] @ keys[position, kv_head]) * scale
            next_maximum = max(maximum, score)
            old_scale = math.exp(maximum - next_maximum)
            new_scale = math.exp(score - next_maximum)
            output[query_head] = (
                output[query_head] * old_scale
                + values[position, kv_head] * new_scale
            )
            denominator = denominator * old_scale + new_scale
            maximum = next_maximum
        output[query_head] /= denominator
    return output


def qwen35_full_attention_prefill(queries, keys, values, prefix_length: int = 0):
    args = tuple(_np_f32(item) for item in (queries, keys, values))
    state = _init()
    if state:
        return state["attention_prefill"](*args, int(prefix_length))
    import numpy as np

    queries, keys, values = args
    result = np.empty(
        (queries.shape[0], queries.shape[1], values.shape[2]), np.float32,
    )
    for token in range(queries.shape[0]):
        result[token] = qwen35_full_attention_decode(
            queries[token], keys[:prefix_length + token + 1],
            values[:prefix_length + token + 1],
        )
    return result


def qwen35_state_copy_restore(source, target):
    source = _np_f32(source)
    if getattr(target, "dtype", None) is None or target.dtype.name != "float32":
        raise ValueError("state restore target must be a float32 NumPy array")
    if source.shape != target.shape:
        raise ValueError(f"state restore shapes differ: {source.shape} != {target.shape}")
    state = _init()
    if state:
        state["state_copy_restore"](source, target)
    else:
        target[:] = source
    return target


def diagnostics() -> dict[str, Any]:
    state = _init()
    if not state:
        return {"available": False, "backend": "numpy", "numba_pin": NUMBA_PIN}
    names = (
        "conv_step", "normalize_qk", "beta_decay", "gdn_step", "gated_rms",
        "attention_decode", "attention_prefill", "state_copy_restore",
    )
    return {
        "available": True,
        "backend": backend_name(),
        "numba_pin": NUMBA_PIN,
        "threads": int(state["numba"].get_num_threads()),
        "nopython_signatures": {
            name: [str(signature) for signature in state[name].nopython_signatures]
            for name in names
        },
    }


__all__ = [
    "NUMBA_PIN", "available", "backend_name", "diagnostics",
    "qwen35_beta_decay", "qwen35_conv_step", "qwen35_full_attention_decode",
    "qwen35_full_attention_prefill", "qwen35_gated_rms", "qwen35_gdn_step",
    "qwen35_normalize_qk", "qwen35_recurrent_layer",
    "qwen35_state_copy_restore",
]
