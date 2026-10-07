# Alpaccaroo - dependency-free Qwen3.5/Qwen3.8 hybrid reference runtime.
# MIT License. See LICENSE.
"""Source-faithful scalar execution for validated dense Qwen35 GGUFs.

This is deliberately the slow correctness backend.  It keeps the GGUF mmap
as the weight authority, materializes no full matrix, and routes every token
through the same one-token semantic path.  NumPy, Numba and CUDA backends may
optimize this contract only after matching its focused fixtures.
"""

from __future__ import annotations

import math
import os
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .gguf import GGUFFile
from .memory import (
    FullAttentionState,
    ModelState,
    PrefixSnapshotStore,
    RecurrentState,
    StateIdentity,
    StateSnapshot,
)
from .qwen35 import Qwen35Config, inspect_qwen35, qwen35_memory_specs
from .qwen35_ops import (
    depthwise_causal_conv_step,
    gated_delta_net_step,
    imrope_rotate,
    imrope_text_position,
    l2_normalize,
    rms_norm,
    silu,
    stable_sigmoid,
    stable_softplus,
)
from .tokenizer import Tokenizer
from .weights import (
    DENSE_EXECUTION_DTYPES,
    MatrixView,
    PackedWeightStore,
    TensorView,
    WeightStore,
)


TraceCallback = Callable[[str, Sequence[float]], None]


def _f32(values: Sequence[float]) -> array:
    """Freeze a semantic boundary to the model's float32 execution dtype."""

    return array("f", values)


def _add(left: Sequence[float], right: Sequence[float]) -> array:
    if len(left) != len(right):
        raise ValueError(f"residual lengths differ: {len(left)} != {len(right)}")
    return array("f", (float(a) + float(b) for a, b in zip(left, right)))


def _mul(left: Sequence[float], right: Sequence[float]) -> array:
    if len(left) != len(right):
        raise ValueError(f"vector lengths differ: {len(left)} != {len(right)}")
    return array("f", (float(a) * float(b) for a, b in zip(left, right)))


def _trace(callback: TraceCallback | None, name: str, values: Sequence[float]) -> None:
    if callback is not None:
        callback(name, values)


def _heads(values: Sequence[float], count: int, width: int, what: str) -> list[array]:
    if len(values) != count * width:
        raise ValueError(
            f"{what} has {len(values)} values, expected {count} * {width}"
        )
    return [
        _f32(values[index * width:(index + 1) * width])
        for index in range(count)
    ]


def _flatten(heads: Sequence[Sequence[float]]) -> array:
    result = array("f")
    for head in heads:
        result.extend(head)
    return result


def _per_head_rms_norm(
    heads: Sequence[Sequence[float]],
    weight: Sequence[float],
    epsilon: float,
) -> list[array]:
    return [_f32(rms_norm(head, weight, epsilon)) for head in heads]


def _causal_gqa_one(
    query_heads: Sequence[Sequence[float]],
    state: FullAttentionState,
) -> list[array]:
    """One-token causal GQA with ordinary contiguous query/KV grouping."""

    spec = state.spec
    query_count = len(query_heads)
    if query_count <= 0 or query_count % spec.kv_heads:
        raise ValueError("query head count must be a positive multiple of KV heads")
    if spec.key_dim != spec.value_dim:
        raise ValueError("the scalar Qwen35 GQA path requires equal K/V dimensions")
    if any(len(head) != spec.key_dim for head in query_heads):
        raise ValueError("every query head must match the attention key dimension")
    if state.logical_length <= 0:
        raise ValueError("attention cache must contain the current token before read")

    query_group = query_count // spec.kv_heads
    scale = 1.0 / math.sqrt(spec.key_dim)
    key_row = spec.kv_heads * spec.key_dim
    value_row = spec.kv_heads * spec.value_dim
    output: list[array] = []
    for query_index, query in enumerate(query_heads):
        kv_index = query_index // query_group
        scores: list[float] = []
        for position in range(state.logical_length):
            key_start = position * key_row + kv_index * spec.key_dim
            score = math.fsum(
                float(query[column]) * float(state.key[key_start + column])
                for column in range(spec.key_dim)
            ) * scale
            scores.append(score)
        maximum = max(scores)
        weights = [math.exp(score - maximum) for score in scores]
        denominator = math.fsum(weights)
        probabilities = [weight / denominator for weight in weights]
        head = array("f", [0.0]) * spec.value_dim
        for column in range(spec.value_dim):
            head[column] = math.fsum(
                probabilities[position]
                * float(state.value[
                    position * value_row + kv_index * spec.value_dim + column
                ])
                for position in range(state.logical_length)
            )
        output.append(head)
    return output


@dataclass(frozen=True, slots=True)
class Qwen35SharedLayerWeights:
    attn_norm: TensorView
    post_attention_norm: TensorView
    ffn_gate: MatrixView
    ffn_up: MatrixView
    ffn_down: MatrixView


@dataclass(frozen=True, slots=True)
class Qwen35RecurrentWeights:
    shared: Qwen35SharedLayerWeights
    qkv: MatrixView
    gate: MatrixView
    conv: MatrixView
    dt_bias: TensorView
    ssm_a: TensorView
    alpha: MatrixView
    beta: MatrixView
    norm: TensorView
    output: MatrixView


@dataclass(frozen=True, slots=True)
class Qwen35AttentionWeights:
    shared: Qwen35SharedLayerWeights
    query_gate: MatrixView
    key: MatrixView
    value: MatrixView
    output: MatrixView
    query_norm: TensorView
    key_norm: TensorView


Qwen35LayerWeights = Qwen35RecurrentWeights | Qwen35AttentionWeights


def _config_from_manifest(manifest: dict[str, Any]) -> Qwen35Config:
    model = manifest["model"]
    names = Qwen35Config.__dataclass_fields__
    return Qwen35Config(**{name: model[name] for name in names})


def _execution_manifest(
    manifest: dict[str, Any],
    config: Qwen35Config,
    context: int,
) -> dict[str, Any]:
    """Return a non-aliasing F32/one-sequence plan for the loaded context."""

    current = manifest["memory_plan"]
    if (
        int(current["requested_context"]) == context
        and int(current["sequences"]) == 1
        and str(current["kv_dtype"]).lower() == "f32"
    ):
        return manifest
    recurrent_layers = int(manifest["model"]["recurrent_layers"])
    attention_layers = int(manifest["model"]["full_attention_layers"])
    recurrent = (
        recurrent_layers * config.time_step_rank * config.state_size
        * config.state_size * 4
    )
    convolution = (
        recurrent_layers * (config.conv_kernel - 1) * config.conv_width * 4
    )
    attention = (
        attention_layers * context * config.kv_heads
        * (config.key_length + config.value_length) * 4
    )
    activation = (
        config.vocabulary_size * 4
        + max(
            config.embedding_length, config.inner_size, config.conv_width,
        ) * 4
    )
    memory = dict(current)
    memory.update({
        "requested_context": context,
        "sequences": 1,
        "kv_dtype": "f32",
        "recurrent_matrix_bytes": recurrent,
        "convolution_history_bytes": convolution,
        "attention_kv_bytes": attention,
        "logits_and_activation_bytes": activation,
    })
    memory["planned_cuda_bytes"] = sum(int(memory[name]) for name in (
        "packed_weight_bytes",
        "recurrent_matrix_bytes",
        "convolution_history_bytes",
        "attention_kv_bytes",
        "logits_and_activation_bytes",
        "cuda_reserve_bytes",
    ))
    budget = memory.get("configured_vram_budget_bytes")
    memory["allocation_fits_configured_vram_budget"] = (
        None if budget is None
        else memory["planned_cuda_bytes"] <= int(budget)
    )
    result = dict(manifest)
    result["memory_plan"] = memory
    return result


def _load_layers(
    weights: WeightStore,
    schedule: Sequence[str],
) -> tuple[Qwen35LayerWeights, ...]:
    layers: list[Qwen35LayerWeights] = []
    for index, kind in enumerate(schedule):
        if kind == "mtp-disabled":
            continue
        prefix = f"blk.{index}."
        shared = Qwen35SharedLayerWeights(
            attn_norm=weights.vector(prefix + "attn_norm.weight"),
            post_attention_norm=weights.vector(
                prefix + "post_attention_norm.weight"
            ),
            ffn_gate=weights.matrix(prefix + "ffn_gate.weight"),
            ffn_up=weights.matrix(prefix + "ffn_up.weight"),
            ffn_down=weights.matrix(prefix + "ffn_down.weight"),
        )
        if kind == "recurrent":
            layers.append(Qwen35RecurrentWeights(
                shared=shared,
                qkv=weights.matrix(prefix + "attn_qkv.weight"),
                gate=weights.matrix(prefix + "attn_gate.weight"),
                conv=weights.matrix(prefix + "ssm_conv1d.weight"),
                dt_bias=weights.vector(prefix + "ssm_dt.bias"),
                ssm_a=weights.vector(prefix + "ssm_a"),
                alpha=weights.matrix(prefix + "ssm_alpha.weight"),
                beta=weights.matrix(prefix + "ssm_beta.weight"),
                norm=weights.vector(prefix + "ssm_norm.weight"),
                output=weights.matrix(prefix + "ssm_out.weight"),
            ))
        elif kind == "attention":
            layers.append(Qwen35AttentionWeights(
                shared=shared,
                query_gate=weights.matrix(prefix + "attn_q.weight"),
                key=weights.matrix(prefix + "attn_k.weight"),
                value=weights.matrix(prefix + "attn_v.weight"),
                output=weights.matrix(prefix + "attn_output.weight"),
                query_norm=weights.vector(prefix + "attn_q_norm.weight"),
                key_norm=weights.vector(prefix + "attn_k_norm.weight"),
            ))
        else:
            raise ValueError(f"cannot execute Qwen35 layer {index} kind {kind!r}")
    return tuple(layers)


class Qwen35Model:
    """Dense, dependency-free Qwen35 reference model with heterogeneous state."""

    backend_name = "pure-python-f32-mmap"
    numerical_mode = "pure-f32"

    def __init__(
        self,
        path: str | Path,
        manifest: dict[str, Any],
        *,
        n_ctx: int = 0,
        progress: bool = True,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.config = _config_from_manifest(manifest)
        requested = n_ctx or min(self.config.context_length, 4096)
        if requested <= 0 or requested > self.config.context_length:
            raise ValueError(
                f"Qwen35 context must be in 1..{self.config.context_length}, "
                f"got {requested}"
            )
        self.n_ctx = requested
        self.validate_finite = os.environ.get(
            "ALPACCAROO_QWEN35_VALIDATE_FINITE", ""
        ).strip().lower() in ("1", "on", "yes", "true")
        self.hp = self.config  # compatibility for CLI/server introspection
        self.full_schedule = tuple(item["kind"] for item in manifest["layers"])
        self.schedule = self.full_schedule[:self.config.main_layer_count]
        manifest = _execution_manifest(manifest, self.config, requested)
        self.manifest = manifest

        # Tokenizer metadata is small compared with weights and is detached
        # from this short-lived header mapping before it closes.
        with GGUFFile.open(self.path, prefetch=False) as gguf:
            self.tok = Tokenizer.from_gguf(gguf.metadata)
            self.metadata = {
                key: value for key, value in gguf.metadata.items()
                if not isinstance(value, list) or len(value) < 64
            }

        execution_dtypes = set(manifest["dtype_census"])
        self.packed_execution = bool(execution_dtypes - DENSE_EXECUTION_DTYPES)
        self.streaming_weights = (
            os.environ.get("ALPACCAROO_QWEN35_STREAM_WEIGHTS", "")
            .strip().lower() in ("1", "on", "yes", "true")
        )
        if self.packed_execution:
            if self.backend_name.startswith("numpy"):
                self.backend_name = (
                    "numpy-packed-stream-cpu" if self.streaming_weights
                    else "numpy-packed-cpu"
                )
                self.numerical_mode = (
                    "numpy-packed-stream-f32-state" if self.streaming_weights
                    else "numpy-packed-f32-state"
                )
            else:
                self.backend_name = "pure-packed-cpu"
                self.numerical_mode = "pure-packed-f32-state"
        store_type = PackedWeightStore if self.packed_execution else WeightStore
        self.weights = store_type.open(self.path, prefetch=False)
        try:
            self.tok_embd = self.weights.matrix("token_embd.weight")
            self.out_norm = self.weights.vector("output_norm.weight")
            self.output = (
                self.weights.matrix("output.weight")
                if "output.weight" in self.weights else self.tok_embd
            )
            self.layers = _load_layers(self.weights, self.schedule)
        except Exception:
            self.weights.close()
            raise

        source_hash = manifest["source"].get("sha256")
        if not source_hash:
            # The validating backend may intentionally skip a multi-gigabyte
            # hash.  Dense reference fixtures are small, so the executable
            # load obtains the exact identity once here.
            from .weights import stream_sha256

            source_hash = stream_sha256(self.path)
            manifest["source"]["sha256"] = source_hash
        positioning = (
            f"imrope-sections={','.join(map(str, self.config.rope_dimension_sections))};"
            f"base={self.config.rope_freq_base};dims={self.config.rope_dimension_count}"
        )
        self.state_identity = StateIdentity(
            architecture="qwen35",
            model_fingerprint=source_hash,
            weights_identity=f"sha256:{source_hash}",
            positioning=positioning,
            numerical_mode=self.numerical_mode,
        )
        specs = qwen35_memory_specs(self.config, list(self.full_schedule))
        self.state = ModelState(self.state_identity, specs)
        self._prefix_store = PrefixSnapshotStore(0)
        self.last_prefill_forwarded = 0
        self.load_seconds = 0.0
        self.architecture_backend = None
        from .qwen35_cuda import cuda_capability
        from .qwen35_rocm import rocm_capability

        self.cuda_status = cuda_capability().descriptor()
        self.rocm_status = rocm_capability().descriptor()
        self.backend_selection = {
            "requested": "direct",
            "selected": self.backend_name,
            "fallback": False,
            "reason": "Qwen35 runtime loaded directly",
        }
        self.weight_storage = {
            "dense": sum(
                census["tensors"] for dtype, census in manifest["dtype_census"].items()
                if dtype in DENSE_EXECUTION_DTYPES
            ),
            "dense_bytes": sum(
                census["packed_bytes"] for dtype, census in manifest["dtype_census"].items()
                if dtype in DENSE_EXECUTION_DTYPES
            ),
            "quantized": {
                dtype: census["tensors"]
                for dtype, census in manifest["dtype_census"].items()
                if dtype not in DENSE_EXECUTION_DTYPES
            },
            "packed_bytes": sum(
                self.weights.packed_bytes(name) for name in self.weights.tensor_names
            ),
            "backend": self.backend_name,
            "streaming_weights": self.streaming_weights,
        }

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        n_ctx: int = 0,
        progress: bool = True,
        manifest: dict[str, Any] | None = None,
    ) -> "Qwen35Model":
        if manifest is None:
            manifest = inspect_qwen35(path, context=n_ctx, calculate_hash=True)
        return cls(path, manifest, n_ctx=n_ctx, progress=progress)

    @property
    def cached_ids(self) -> list[int]:
        return self.state.token_ids

    @property
    def n_past(self) -> int:
        return self.state.position

    def new_state(self) -> ModelState:
        return ModelState(self.state_identity, self.state.descriptors)

    def _matvec(self, matrix: MatrixView, values: Sequence[float]):
        return matrix.matvec(values)

    def _embedding(self, token_id: int):
        return _f32(self.tok_embd.embedding_row(token_id))

    def _attention_read(
        self,
        query_heads: Sequence[Sequence[float]],
        state: FullAttentionState,
    ) -> list[array]:
        return _causal_gqa_one(query_heads, state)

    def _require_finite(
        self,
        values: Sequence[float],
        *,
        state: ModelState,
        token_position: int,
        layer: int | str,
        stage: str,
    ) -> None:
        if not self.validate_finite:
            return
        for index, value in enumerate(values):
            if not math.isfinite(float(value)):
                raise FloatingPointError(
                    "non-finite Qwen35 value: architecture=qwen35 "
                    f"backend={self.backend_name} layer={layer} "
                    f"token={token_position} stage={stage} "
                    f"generation={state.generation} index={index} value={value}"
                )

    def _recurrent(
        self,
        layer_index: int,
        weights: Qwen35RecurrentWeights,
        normalized: Sequence[float],
        state: RecurrentState,
        trace: TraceCallback | None,
    ) -> array:
        cfg = self.config
        mixed = self._matvec(weights.qkv, normalized)
        gate = self._matvec(weights.gate, normalized)
        alpha0 = self._matvec(weights.alpha, normalized)
        beta0 = self._matvec(weights.beta, normalized)
        _trace(trace, f"layer.{layer_index}.linear_attn_qkv_mixed", mixed)

        # GGUF bytes are channel-major: MatrixView rows are channels and each
        # row contains chronological taps.  The primitive API is tap-major.
        conv_weights = [
            [float(weights.conv[channel, tap]) for channel in range(cfg.conv_width)]
            for tap in range(cfg.conv_kernel)
        ]
        history = [
            list(state.conv_history[
                tap * cfg.conv_width:(tap + 1) * cfg.conv_width
            ])
            for tap in range(cfg.conv_kernel - 1)
        ]
        convolved, new_history = depthwise_causal_conv_step(
            mixed, conv_weights, history,
        )
        _trace(trace, f"layer.{layer_index}.conv_output_raw", convolved)
        activated = _f32([silu(value) for value in convolved])
        _trace(trace, f"layer.{layer_index}.conv_output_silu", activated)

        qk_width = cfg.group_count * cfg.state_size
        queries = _heads(
            activated[:qk_width], cfg.group_count, cfg.state_size, "recurrent Q"
        )
        keys = _heads(
            activated[qk_width:2 * qk_width],
            cfg.group_count, cfg.state_size, "recurrent K",
        )
        values = _heads(
            activated[2 * qk_width:],
            cfg.time_step_rank, cfg.state_size, "recurrent V",
        )
        queries = [
            _f32(l2_normalize(head, cfg.rms_epsilon)) for head in queries
        ]
        keys = [_f32(l2_normalize(head, cfg.rms_epsilon)) for head in keys]
        _trace(trace, f"layer.{layer_index}.Qcur_normed", _flatten(queries))
        _trace(trace, f"layer.{layer_index}.Kcur_normed", _flatten(keys))

        head_matrix = cfg.state_size * cfg.state_size
        outputs: list[array] = []
        next_delta = array("f")
        _trace(trace, f"layer.{layer_index}.state_before", state.delta_matrix)
        for value_head in range(cfg.time_step_rank):
            key_head = value_head % cfg.group_count
            start = value_head * head_matrix
            prior = [
                list(state.delta_matrix[
                    start + row * cfg.state_size:
                    start + (row + 1) * cfg.state_size
                ])
                for row in range(cfg.state_size)
            ]
            beta = stable_sigmoid(float(beta0[value_head]))
            dt = stable_softplus(
                float(alpha0[value_head]) + float(weights.dt_bias[value_head])
            )
            log_decay = dt * float(weights.ssm_a[value_head])
            head_output, updated = gated_delta_net_step(
                prior,
                queries[key_head],
                keys[key_head],
                values[value_head],
                beta=beta,
                log_decay=log_decay,
            )
            updated_f32 = [_f32(row) for row in updated]
            for row in updated_f32:
                next_delta.extend(row)
            # State output is consumed at float32 before the learned norm.
            normalized_head = _f32(rms_norm(
                _f32(head_output), weights.norm, cfg.rms_epsilon,
            ))
            gate_start = value_head * cfg.state_size
            gated = array("f", (
                float(normalized_head[column])
                * silu(float(gate[gate_start + column]))
                for column in range(cfg.state_size)
            ))
            outputs.append(gated)

        flat_history = array("f")
        for row in new_history:
            flat_history.extend(row)
        state.advance(conv_history=flat_history, delta_matrix=next_delta)
        _trace(trace, f"layer.{layer_index}.new_state", next_delta)
        joined = _flatten(outputs)
        _trace(trace, f"layer.{layer_index}.attn_gated", joined)
        result = self._matvec(weights.output, joined)
        _trace(trace, f"layer.{layer_index}.attn_output", result)
        return result

    def _attention(
        self,
        layer_index: int,
        weights: Qwen35AttentionWeights,
        normalized: Sequence[float],
        state: FullAttentionState,
        position: int,
        trace: TraceCallback | None,
    ) -> array:
        cfg = self.config
        query_gate = self._matvec(weights.query_gate, normalized)
        key_flat = self._matvec(weights.key, normalized)
        value_flat = self._matvec(weights.value, normalized)
        query_heads: list[array] = []
        gate_heads: list[array] = []
        stride = 2 * cfg.key_length
        for head in range(cfg.query_heads):
            start = head * stride
            query_heads.append(_f32(query_gate[start:start + cfg.key_length]))
            gate_heads.append(_f32(
                query_gate[start + cfg.key_length:start + stride]
            ))
        key_heads = _heads(
            key_flat, cfg.kv_heads, cfg.key_length, "attention K"
        )
        value_heads = _heads(
            value_flat, cfg.kv_heads, cfg.value_length, "attention V"
        )
        query_heads = _per_head_rms_norm(
            query_heads, weights.query_norm, cfg.rms_epsilon,
        )
        key_heads = _per_head_rms_norm(
            key_heads, weights.key_norm, cfg.rms_epsilon,
        )
        lanes = imrope_text_position(position)
        query_heads = [
            _f32(imrope_rotate(
                head, lanes, cfg.rope_dimension_sections,
                frequency_base=cfg.rope_freq_base,
            ))
            for head in query_heads
        ]
        key_heads = [
            _f32(imrope_rotate(
                head, lanes, cfg.rope_dimension_sections,
                frequency_base=cfg.rope_freq_base,
            ))
            for head in key_heads
        ]
        rotated_keys = _flatten(key_heads)
        raw_values = _flatten(value_heads)
        state.append(rotated_keys, raw_values)
        _trace(trace, f"layer.{layer_index}.Qcur_normed", _flatten(query_heads))
        _trace(trace, f"layer.{layer_index}.Kcur_normed", rotated_keys)
        attended = self._attention_read(query_heads, state)
        _trace(trace, f"layer.{layer_index}.attn_pregate", _flatten(attended))
        gated = [
            array("f", (
                float(attended[head][column])
                * stable_sigmoid(float(gate_heads[head][column]))
                for column in range(cfg.value_length)
            ))
            for head in range(cfg.query_heads)
        ]
        joined = _flatten(gated)
        _trace(trace, f"layer.{layer_index}.attn_gated", joined)
        result = self._matvec(weights.output, joined)
        _trace(trace, f"layer.{layer_index}.attn_output", result)
        return result

    def _ffn(
        self,
        layer_index: int,
        weights: Qwen35SharedLayerWeights,
        values: Sequence[float],
        trace: TraceCallback | None,
    ) -> array:
        normalized = _f32(rms_norm(
            values, weights.post_attention_norm, self.config.rms_epsilon,
        ))
        gate = self._matvec(weights.ffn_gate, normalized)
        up = self._matvec(weights.ffn_up, normalized)
        activated = array("f", (
            silu(float(gate[index])) * float(up[index])
            for index in range(len(gate))
        ))
        output = self._matvec(weights.ffn_down, activated)
        _trace(trace, f"layer.{layer_index}.ffn_out", output)
        return _add(values, output)

    def _advance(
        self,
        state: ModelState,
        token_id: int,
        *,
        trace: TraceCallback | None = None,
    ) -> array:
        self.weights._ensure_open()
        if token_id < 0 or token_id >= self.config.vocabulary_size:
            raise ValueError(
                f"token ID {token_id} is outside vocabulary size "
                f"{self.config.vocabulary_size}"
            )
        if state.identity != self.state_identity:
            raise ValueError("Qwen35 state does not belong to this model")
        if state.position >= self.n_ctx:
            raise RuntimeError(f"context window full ({self.n_ctx} tokens)")
        position = state.position
        values = self._embedding(token_id)
        self._require_finite(
            values, state=state, token_position=position,
            layer="model", stage="embedding",
        )
        for layer_index, weights in enumerate(self.layers):
            residual = values
            normalized = _f32(rms_norm(
                values, weights.shared.attn_norm, self.config.rms_epsilon,
            ))
            _trace(trace, f"layer.{layer_index}.attn_norm", normalized)
            self._require_finite(
                normalized, state=state, token_position=position,
                layer=layer_index, stage="attention_norm",
            )
            layer_state = state.layer(layer_index)
            if isinstance(weights, Qwen35RecurrentWeights):
                if not isinstance(layer_state, RecurrentState):
                    raise TypeError(f"layer {layer_index} lacks recurrent state")
                attention = self._recurrent(
                    layer_index, weights, normalized, layer_state, trace,
                )
            else:
                if not isinstance(layer_state, FullAttentionState):
                    raise TypeError(f"layer {layer_index} lacks attention state")
                attention = self._attention(
                    layer_index, weights, normalized, layer_state, position, trace,
                )
            self._require_finite(
                attention, state=state, token_position=position,
                layer=layer_index, stage="attention_output",
            )
            values = _add(residual, attention)
            self._require_finite(
                values, state=state, token_position=position,
                layer=layer_index, stage="attention_residual",
            )
            values = self._ffn(layer_index, weights.shared, values, trace)
            self._require_finite(
                values, state=state, token_position=position,
                layer=layer_index, stage="ffn_residual",
            )
        normalized = _f32(rms_norm(
            values, self.out_norm, self.config.rms_epsilon,
        ))
        logits = self._matvec(self.output, normalized)
        _trace(trace, "model.result_output", logits)
        self._require_finite(
            logits, state=state, token_position=position,
            layer="model", stage="logits",
        )
        state.record_token(token_id)
        return logits

    def forward_token(
        self,
        token_id: int,
        state: ModelState,
        *,
        trace: TraceCallback | None = None,
    ) -> array:
        """Advance *state* atomically by one token and return its logits."""

        prepared = state.clone()
        logits = self._advance(prepared, token_id, trace=trace)
        prepared._validate_live()
        state._publish(prepared)
        return logits

    def forward(self, token_id: int) -> array:
        return self.forward_token(token_id, self.state)

    def prefill_state(
        self,
        token_ids: Sequence[int],
        state: ModelState,
        *,
        chunk_size: int = 1,
        trace: TraceCallback | None = None,
    ) -> array | None:
        if chunk_size <= 0:
            raise ValueError("prefill chunk_size must be positive")
        tokens = list(token_ids)
        if len(tokens) > self.n_ctx:
            raise RuntimeError(f"context window full ({self.n_ctx} tokens)")
        if not tokens:
            return None
        prefix = 0
        while (
            prefix < len(tokens) and prefix < state.position
            and tokens[prefix] == state.token_ids[prefix]
        ):
            prefix += 1
        # A complete cached prompt has no retained last-token logits.  Replay
        # the final token (and recurrent prefix as required) to preserve the
        # public prefill contract exactly.
        if prefix == len(tokens):
            prefix = max(0, prefix - 1)
        if prefix != state.position:
            state.truncate(prefix, replay=lambda staged, token: self._advance(
                staged, token, trace=None,
            ))
        logits: array | None = None
        for token in tokens[prefix:]:
            logits = self.forward_token(token, state, trace=trace)
        return logits

    def prefill(self, token_ids: Sequence[int]) -> array | None:
        tokens = list(token_ids)
        prefix = 0
        while (
            prefix < len(tokens) and prefix < self.state.position
            and tokens[prefix] == self.state.token_ids[prefix]
        ):
            prefix += 1
        if prefix == len(tokens) and tokens:
            prefix -= 1
        self.last_prefill_forwarded = len(tokens) - prefix
        logits = self.prefill_state(token_ids, self.state)
        return logits

    def reset(self, *, clear_prefixes: bool = False) -> None:
        self.state.reset()
        if clear_prefixes:
            self._prefix_store.clear()

    def snapshot(self, state: ModelState | None = None) -> StateSnapshot:
        return (self.state if state is None else state).snapshot()

    snapshot_state = snapshot

    def restore(
        self, snapshot: StateSnapshot, state: ModelState | None = None,
    ) -> None:
        (self.state if state is None else state).restore(snapshot)

    restore_state = restore

    def branch(self, state: ModelState | None = None) -> ModelState:
        return (self.state if state is None else state).clone()

    def truncate(self, position: int, state: ModelState | None = None) -> None:
        target = self.state if state is None else state
        target.truncate(
            position,
            replay=lambda staged, token: self._advance(staged, token, trace=None),
        )

    def configure_prefix_cache(self, max_bytes: int, max_slots: int | None = None) -> None:
        self._prefix_store = PrefixSnapshotStore(max_bytes, max_slots=max_slots)

    def clear_prefix_cache(self) -> None:
        self._prefix_store.clear()

    def save_prefix(self, state: ModelState | None = None) -> bool:
        return self._prefix_store.save(self.state if state is None else state)

    def restore_prefix(
        self, token_ids: Sequence[int], state: ModelState | None = None,
    ) -> bool:
        return self._prefix_store.restore(
            self.state if state is None else state, token_ids,
        )

    def prefix_cache_stats(self) -> dict[str, int]:
        return {
            "slots": len(self._prefix_store),
            "bytes": self._prefix_store.current_bytes,
            "hits": self._prefix_store.hits,
            "misses": self._prefix_store.misses,
            "evictions": self._prefix_store.evictions,
        }

    def describe_data(self) -> dict[str, Any]:
        backend_selection = dict(self.backend_selection)
        backend_selection["selected"] = self.backend_name
        attention_layers = sum(kind == "attention" for kind in self.schedule)
        recurrent_layers = sum(kind == "recurrent" for kind in self.schedule)
        recurrent_matrix_bytes = int(
            self.manifest["memory_plan"]["recurrent_matrix_bytes"]
        )
        convolution_history_bytes = int(
            self.manifest["memory_plan"]["convolution_history_bytes"]
        )
        attention_kv_capacity_bytes = (
            attention_layers * self.n_ctx * self.config.kv_heads
            * (self.config.key_length + self.config.value_length) * 4
        )
        return {
            "architecture": "qwen35",
            "backend": self.backend_name,
            "backend_selection": backend_selection,
            "executable": True,
            "layers": self.config.main_layer_count,
            "recurrent_layers": recurrent_layers,
            "full_attention_layers": attention_layers,
            "context": self.n_ctx,
            "trained_context": self.config.context_length,
            "state": self.state.describe(),
            "memory": {
                "packed_weight_bytes": int(self.weight_storage["packed_bytes"]),
                "recurrent_matrix_bytes": recurrent_matrix_bytes,
                "convolution_history_bytes": convolution_history_bytes,
                "attention_kv_capacity_bytes": attention_kv_capacity_bytes,
                "attention_kv_dtype": "f32",
                "live_state_bytes": self.state.memory_bytes(),
                "actual_cuda_vram_bytes": 0,
                "actual_rocm_shared_bytes": 0,
            },
            "placement": {
                "backend": "cpu",
                "weights": (
                    "host-native-packed" if self.packed_execution
                    else "host-f32-mmap"
                ),
                "mutable_state": "host-f32",
                "attention_kv": "host-f32",
            },
            "checkpoint_policy": {
                "authority": "host",
                "automatic_interval_tokens": None,
                "explicit_snapshots": "complete-deep-copy",
                "truncate": "complete-checkpoint-or-reset-and-replay",
            },
            "rocm": self.rocm_status,
            "cuda": self.cuda_status,
            "unsupported_features": self.manifest["unsupported_features"],
        }

    def state_description(self) -> dict[str, Any]:
        return self.state.describe()

    def describe(self) -> str:
        recurrent = sum(kind == "recurrent" for kind in self.schedule)
        attention = sum(kind == "attention" for kind in self.schedule)
        state_bytes = self.state.memory_bytes()
        memory = self.describe_data()["memory"]
        packed_bytes = int(self.weight_storage.get("packed_bytes", 0) or 0)
        rocm = self.rocm_status
        rocm_label = (
            f"{rocm.get('device_name')} ({rocm.get('gcn_arch_name')})"
            if rocm.get("available") else "unavailable"
        )
        rocm_budget = int(rocm.get("memory_budget_bytes") or 0)
        cuda = self.cuda_status
        cuda_label = (
            cuda.get("device_name") if cuda.get("available")
            else "unavailable"
        )
        return (
            f"qwen35 | {self.config.main_layer_count} layers "
            f"({recurrent} recurrent/{attention} attention) | "
            f"embd {self.config.embedding_length} | "
            f"heads {self.config.query_heads}/{self.config.kv_heads} | "
            f"ff {self.config.feed_forward_length} | "
            f"vocab {self.config.vocabulary_size} | ctx {self.n_ctx} of "
            f"{self.config.context_length} | backend {self.backend_name} | "
            f"state {state_bytes / 1048576:.1f} MiB | "
            f"K/V capacity {memory['attention_kv_capacity_bytes'] / 1048576:.1f} MiB F32 | "
            f"packed weights {packed_bytes / (1024 ** 3):.2f} GiB | "
            f"placement CPU (actual VRAM 0 B) | "
            f"ROCm {rocm_label} (HIP {rocm.get('hip_version') or 'unavailable'}, "
            f"budget {rocm_budget / (1024 ** 3):.2f} GiB) | "
            f"CUDA {cuda_label} | "
            f"MTP {'present-disabled' if self.config.nextn_predict_layers else 'absent'}"
        )

    def close(self) -> None:
        if self.weights.closed:
            return
        self.weights.close()
        # Release owned expanded/packed execution representations promptly;
        # a closed model is intentionally unusable and _advance checks that
        # invariant before touching these attributes.
        self.layers = ()
        self.tok_embd = None
        self.out_norm = None
        self.output = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "Qwen35AttentionWeights",
    "Qwen35LayerWeights",
    "Qwen35Model",
    "Qwen35RecurrentWeights",
    "Qwen35SharedLayerWeights",
]
