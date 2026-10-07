# Alpaccaroo - NumPy projection backend for the Qwen35 hybrid runtime.
# MIT License. See LICENSE.
"""NumPy/optional-Numba execution under the frozen Qwen35 semantics.

Projection views never retain mmap exports. Recurrent convolution, Q/K
normalization, DeltaNet state traversal, gated RMSNorm, and online attention
use contiguous float32 arrays and the optional pinned kernels; state chronology
and transaction publication remain owned by the shared reference runtime.
"""

from __future__ import annotations

from array import array
from collections.abc import Sequence

import numpy as np

from .memory import FullAttentionState, RecurrentState
from . import qwen35_kernels as K
from .qwen35_runtime import (
    Qwen35Model,
    Qwen35RecurrentWeights,
    _trace,
)
from .weights import MatrixView


class Qwen35NumpyModel(Qwen35Model):
    backend_name = "numpy-f32-mmap"
    numerical_mode = "numpy-f32"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._numpy_recurrent: dict[int, dict[str, np.ndarray]] = {}
        for layer_index, weights in enumerate(self.layers):
            if not isinstance(weights, Qwen35RecurrentWeights):
                continue
            conv = np.stack([
                weights.conv.embedding_row_numpy(channel)
                for channel in range(self.config.conv_width)
            ]).astype(np.float32, copy=False)
            self._numpy_recurrent[layer_index] = {
                "conv": np.ascontiguousarray(conv),
                "dt_bias": np.asarray(tuple(weights.dt_bias), dtype=np.float32),
                "ssm_a": np.asarray(tuple(weights.ssm_a), dtype=np.float32),
                "norm": np.asarray(tuple(weights.norm), dtype=np.float32),
            }
        self.kernel_backend = K.backend_name()
        if K.available():
            self.backend_name = (
                ("numba-packed-stream-cpu" if self.streaming_weights
                 else "numba-packed-cpu") if self.packed_execution
                else "numba-f32-mmap"
            )
            self.weight_storage["backend"] = self.backend_name

    def _matvec(self, matrix: MatrixView, values: Sequence[float]):
        return matrix.matvec_numpy(values)

    def _embedding(self, token_id: int):
        return self.tok_embd.embedding_row_numpy(token_id)

    def _recurrent(
        self,
        layer_index: int,
        weights: Qwen35RecurrentWeights,
        normalized: Sequence[float],
        state: RecurrentState,
        trace,
    ) -> array:
        cfg = self.config
        mixed = np.asarray(self._matvec(weights.qkv, normalized), dtype=np.float32)
        gate = np.asarray(self._matvec(weights.gate, normalized), dtype=np.float32)
        alpha0 = np.asarray(self._matvec(weights.alpha, normalized), dtype=np.float32)
        beta0 = np.asarray(self._matvec(weights.beta, normalized), dtype=np.float32)
        _trace(trace, f"layer.{layer_index}.linear_attn_qkv_mixed", mixed)

        auxiliary = self._numpy_recurrent[layer_index]
        history = np.frombuffer(state.conv_history, dtype=np.float32).reshape(
            cfg.conv_kernel - 1, cfg.conv_width,
        )
        if trace is not None:
            window = np.concatenate((history, mixed[None, :]), axis=0)
            raw = (window.T * auxiliary["conv"]).sum(axis=1, dtype="float32")
            _trace(trace, f"layer.{layer_index}.conv_output_raw", raw)
        activated, _ = K.qwen35_conv_step(
            mixed, auxiliary["conv"], history,
        )
        _trace(trace, f"layer.{layer_index}.conv_output_silu", activated)

        qk_width = cfg.group_count * cfg.state_size
        queries = activated[:qk_width].reshape(cfg.group_count, cfg.state_size)
        keys = activated[qk_width:2 * qk_width].reshape(
            cfg.group_count, cfg.state_size,
        )
        values = activated[2 * qk_width:].reshape(
            cfg.time_step_rank, cfg.state_size,
        )
        gates = gate.reshape(cfg.time_step_rank, cfg.state_size)
        delta = np.frombuffer(state.delta_matrix, dtype=np.float32).reshape(
            cfg.time_step_rank, cfg.state_size, cfg.state_size,
        )
        queries, keys = K.qwen35_normalize_qk(
            queries, keys, cfg.rms_epsilon,
        )
        _trace(trace, f"layer.{layer_index}.Qcur_normed", queries.reshape(-1))
        _trace(trace, f"layer.{layer_index}.Kcur_normed", keys.reshape(-1))
        _trace(
            trace, f"layer.{layer_index}.state_before",
            delta.reshape(-1).copy(),
        )
        beta, log_decay = K.qwen35_beta_decay(
            beta0, alpha0, auxiliary["dt_bias"], auxiliary["ssm_a"],
        )
        state_output = K.qwen35_gdn_step(
            delta, queries, keys, values, beta, log_decay,
        )
        _trace(
            trace, f"layer.{layer_index}.new_state", delta.reshape(-1).copy(),
        )
        gated = K.qwen35_gated_rms(
            state_output, auxiliary["norm"], gates, cfg.rms_epsilon,
        )
        state.logical_position += 1
        joined = np.ascontiguousarray(gated.reshape(-1), dtype=np.float32)
        _trace(trace, f"layer.{layer_index}.attn_gated", joined)
        result = self._matvec(weights.output, joined)
        _trace(trace, f"layer.{layer_index}.attn_output", result)
        return array("f", result)

    def _attention_read(
        self,
        query_heads: Sequence[Sequence[float]],
        state: FullAttentionState,
    ) -> list[array]:
        cfg = self.config
        queries = np.asarray(query_heads, dtype=np.float32)
        keys = np.frombuffer(state.key, dtype=np.float32).reshape(
            state.logical_length, cfg.kv_heads, cfg.key_length,
        )
        values = np.frombuffer(state.value, dtype=np.float32).reshape(
            state.logical_length, cfg.kv_heads, cfg.value_length,
        )
        attended = K.qwen35_full_attention_decode(queries, keys, values)
        return [array("f", row) for row in attended]


__all__ = ["Qwen35NumpyModel"]
