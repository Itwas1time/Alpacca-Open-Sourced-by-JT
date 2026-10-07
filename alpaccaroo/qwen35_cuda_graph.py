# Alpaccaroo - simulator-qualified device-resident Qwen35 token graph.
# MIT License. See LICENSE.
"""Complete Qwen35 hybrid CUDA execution graph.

This module composes native-packed projections with the CUDA semantic
primitives while keeping activations, recurrent state and attention caches on
the device. The simulator is the recorded qualification environment; an
explicit CUDA backend request may run the same graph on a real device to
produce the still-required hardware evidence. Automatic selection stays
closed until that evidence passes.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

from .memory import FullAttentionState, ModelState, RecurrentState, StateSnapshot
from .packed_gpu import (
    PackedTensorSource,
    Qwen35PlacementPlan,
    build_qwen35_placement_plan,
    verify_actual_allocations,
)
from .qwen35_cuda import (
    CUDA_PACKED_IMPLEMENTED_DTYPES,
    PackedCudaMatrix,
    Qwen35CudaUnavailable,
    cuda_capability,
    cuda_kernels,
)
from .qwen35_runtime import (
    Qwen35AttentionWeights,
    Qwen35Model,
    Qwen35RecurrentWeights,
    TraceCallback,
)


THREADS = 64
DEFAULT_CUDA_BUDGET_BYTES = 24 * 1024**3


class Qwen35CudaGraphFault(RuntimeError):
    """Deterministic stage fault used to prove device-state rollback."""


_graph_kernels: dict[str, Any] | None = None
cuda = None  # module-global so Numba's simulator can patch kernel globals


def _cuda_graph_kernels(base: dict[str, Any]) -> dict[str, Any]:
    global _graph_kernels, cuda
    if _graph_kernels is not None:
        return _graph_kernels
    cuda = base["cuda"]

    @cuda.jit
    def _set_text_position(lanes, position):
        index = cuda.grid(1)
        if index < 4:
            lanes[index] = position if index < 3 else 0

    _graph_kernels = {
        "set_text_position": _set_text_position,
    }
    return _graph_kernels


@dataclass(slots=True)
class _SharedLayer:
    attention_norm: Any
    post_attention_norm: Any
    ffn_gate: PackedCudaMatrix
    ffn_up: PackedCudaMatrix
    ffn_down: PackedCudaMatrix


@dataclass(slots=True)
class _RecurrentLayer:
    layer_index: int
    shared: _SharedLayer
    qkv: PackedCudaMatrix
    gate: PackedCudaMatrix
    alpha: PackedCudaMatrix
    beta: PackedCudaMatrix
    output: PackedCudaMatrix
    conv_weights: Any
    dt_bias: Any
    ssm_a: Any
    norm: Any
    active_conv: Any
    staging_conv: Any
    active_delta: Any
    staging_delta: Any

    def swap_state(self) -> None:
        self.active_conv, self.staging_conv = self.staging_conv, self.active_conv
        self.active_delta, self.staging_delta = (
            self.staging_delta, self.active_delta,
        )


@dataclass(slots=True)
class _AttentionLayer:
    layer_index: int
    shared: _SharedLayer
    query_gate: PackedCudaMatrix
    key: PackedCudaMatrix
    value: PackedCudaMatrix
    output: PackedCudaMatrix
    query_norm: Any
    key_norm: Any
    active_key: Any
    staging_key: Any
    active_value: Any
    staging_value: Any

    def swap_state(self) -> None:
        self.active_key, self.staging_key = self.staging_key, self.active_key
        self.active_value, self.staging_value = (
            self.staging_value, self.active_value,
        )


_DeviceLayer = _RecurrentLayer | _AttentionLayer


class Qwen35CudaSimulatorGraph:
    """Persistent, double-buffered, device-resident one-token graph.

    ``forward`` atomically publishes device state: active recurrent/KV buffers
    are copied device-to-device into staging, the token mutates only staging,
    and buffer roles swap after logits successfully reach the host.  A raised
    kernel or injected stage fault leaves the active state unchanged.
    """

    backend_name = "qwen35-cuda-simulator-device-graph"
    production_qualified = False
    qualification = "CUDA simulator only; no real-device qualification"

    def __init__(
        self,
        model: Qwen35Model,
        *,
        owns_model: bool = False,
        cuda_budget_bytes: int = DEFAULT_CUDA_BUDGET_BYTES,
        placement_plan: Qwen35PlacementPlan | None = None,
    ) -> None:
        capability = cuda_capability(allow_simulator=True)
        if not capability.available:
            raise Qwen35CudaUnavailable(capability.reason)
        self.simulator = capability.simulator
        self.backend_name = (
            "qwen35-cuda-simulator-device-graph" if capability.simulator
            else "qwen35-cuda-qualification-device-graph"
        )
        self.qualification = (
            "CUDA simulator semantic qualification only"
            if capability.simulator else
            "real-device qualification candidate; production evidence pending"
        )
        if model.n_ctx <= 0:
            raise ValueError("Qwen35 device graph requires a positive context")
        expected_plan = self.plan_before_allocation(
            model, cuda_budget_bytes=cuda_budget_bytes,
        )
        if (
            placement_plan is not None
            and placement_plan.descriptor() != expected_plan.descriptor()
        ):
            raise Qwen35CudaUnavailable(
                "serialized CUDA placement plan differs from the current "
                "model, context, qualification set, or byte budget"
            )
        self.placement_plan = placement_plan or expected_plan
        # Materialize a stable, reviewable representation before the first
        # device allocation. Callers may persist this exact JSON alongside a
        # run; the constructor rejects any supplied plan that differs.
        self.serialized_placement_plan = json.dumps(
            self.placement_plan.descriptor(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if not self.placement_plan.all_weights_device_resident:
            rejected = [
                f"{group.name}: {group.reason}"
                for group in self.placement_plan.groups
                if group.placement != "cuda"
            ]
            raise Qwen35CudaUnavailable(
                "complete device graph requires an all-CUDA placement plan; "
                + "; ".join(rejected[:4])
            )
        self.model = model
        self.tok = model.tok
        self.tokenizer = model.tok
        self.config = model.config
        self.n_ctx = model.n_ctx
        self._owns_model = owns_model
        self._closed = False
        self._dirty = False
        self._transaction_start_position: int | None = None
        self._transaction_tokens: list[int] = []
        self._transaction_expected_tokens = 0
        self._transaction_in_place = False
        self._attention_staging_prefix = 0
        self._kernels = cuda_kernels(allow_simulator=True)
        self._graph = _cuda_graph_kernels(self._kernels)
        self._cuda = self._kernels["cuda"]
        self._np = self._kernels["np"]
        self._source = PackedTensorSource.open(model.path)
        self._matrices: dict[str, PackedCudaMatrix] = {}
        self._actual_weight_allocations: dict[str, int] = {}
        self._actual_auxiliary_allocations: dict[str, int] = {}
        self.allocation_report: dict[str, Any] = {}
        self._last_complete_host_checkpoint: StateSnapshot | None = None
        self._last_pinned_checkpoint_buffers: tuple[Any, ...] = ()
        self.checkpoint_interval = 128
        self.layers: list[_DeviceLayer] = []
        self.position = 0
        self.token_ids: list[int] = []
        self.generation = 0
        self.last_fault = ""
        self._fault: tuple[str, int] | None = None
        self._counters: dict[str, int] = {
            "tokens": 0,
            "launches": 0,
            "synchronizations": 0,
            "setup_host_to_device_bytes": 0,
            "routine_token_control_host_to_device_bytes": 0,
            "routine_logits_device_to_host_bytes": 0,
            "routine_state_host_to_device_bytes": 0,
            "routine_state_device_to_host_bytes": 0,
            "routine_intermediate_host_to_device_bytes": 0,
            "routine_intermediate_device_to_host_bytes": 0,
            "routine_recurrent_device_to_device_bytes": 0,
            "checkpoint_host_to_device_bytes": 0,
            "checkpoint_device_to_host_bytes": 0,
            "trace_device_to_host_bytes": 0,
        }
        try:
            self._load_device_graph()
            self._reconcile_device_allocations()
            if model.n_past:
                # Construction from an already-used model must not silently
                # reset its logical history.  Treat the initial import as an
                # explicit checkpoint restore so its transfer cost remains
                # visible and both K/V ping-pong slots are canonical.
                self.restore(model.snapshot())
        except Exception:
            self.close()
            raise

    @classmethod
    def plan_before_allocation(
        cls,
        model: Qwen35Model,
        *,
        cuda_budget_bytes: int = DEFAULT_CUDA_BUDGET_BYTES,
    ) -> Qwen35PlacementPlan:
        """Build the deterministic plan callers can serialize before loading."""

        return build_qwen35_placement_plan(
            model.manifest,
            budget_bytes=cuda_budget_bytes,
            qualified_dtypes=CUDA_PACKED_IMPLEMENTED_DTYPES,
            state_copies=2,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        n_ctx: int = 0,
        progress: bool = False,
        manifest: dict[str, Any] | None = None,
        cuda_budget_bytes: int = DEFAULT_CUDA_BUDGET_BYTES,
    ) -> "Qwen35CudaSimulatorGraph":
        model = Qwen35Model.load(
            path, n_ctx=n_ctx, progress=progress, manifest=manifest,
        )
        try:
            return cls(
                model,
                owns_model=True,
                cuda_budget_bytes=cuda_budget_bytes,
            )
        except Exception:
            model.close()
            raise

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Qwen35 CUDA simulator graph is closed")

    def _empty(self, shape: int | tuple[int, ...]):
        return self._cuda.device_array(shape, dtype=self._np.float32)

    def _launch(self, kernel: Any, work_items: int, *args: Any) -> None:
        if work_items <= 0:
            return
        blocks = (work_items + THREADS - 1) // THREADS
        kernel[blocks, THREADS](*args)
        self._counters["launches"] += 1

    def _zero(self, values: Any) -> None:
        self._launch(
            self._kernels["fill_zero"], values.size, values.reshape(values.size),
        )

    def _copy(self, source: Any, target: Any) -> None:
        if tuple(source.shape) != tuple(target.shape):
            raise ValueError(
                f"device state copy shape differs: {source.shape} != {target.shape}"
            )
        self._launch(
            self._kernels["state_copy"],
            source.size,
            source.reshape(source.size),
            target.reshape(target.size),
        )

    def _record_weight_allocation(self, name: str, byte_count: int) -> None:
        previous = self._actual_weight_allocations.setdefault(name, int(byte_count))
        if previous != int(byte_count):
            raise ValueError(
                f"device tensor {name} allocation changed: "
                f"{previous} != {byte_count}"
            )

    def _upload_vector(self, values: Sequence[float]):
        host = self._np.asarray(tuple(values), dtype=self._np.float32)
        device = self._cuda.to_device(host)
        self._counters["setup_host_to_device_bytes"] += int(host.nbytes)
        name = getattr(values, "name", None)
        if not isinstance(name, str) or not name:
            raise ValueError("device weight vector lacks a GGUF tensor name")
        self._record_weight_allocation(name, int(device.nbytes))
        return device

    def _matrix(self, view: Any) -> PackedCudaMatrix:
        matrix = self._matrices.get(view.name)
        if matrix is None:
            matrix = PackedCudaMatrix.from_source(
                self._source, view.name, allow_simulator=True,
            )
            self._matrices[view.name] = matrix
            self._record_weight_allocation(
                view.name, int(matrix.device_packed_bytes),
            )
            self._counters["setup_host_to_device_bytes"] += int(
                matrix.device_packed_bytes
            )
        if (
            matrix.dtype != view.dtype
            or matrix.columns != view.columns
            or matrix.rows != view.rows
        ):
            raise ValueError(f"device matrix metadata mismatch for {view.name}")
        return matrix

    def _shared(self, weights: Any) -> _SharedLayer:
        return _SharedLayer(
            attention_norm=self._upload_vector(weights.attn_norm),
            post_attention_norm=self._upload_vector(
                weights.post_attention_norm,
            ),
            ffn_gate=self._matrix(weights.ffn_gate),
            ffn_up=self._matrix(weights.ffn_up),
            ffn_down=self._matrix(weights.ffn_down),
        )

    def _fresh_zero(self, shape: tuple[int, ...]):
        values = self._empty(shape)
        self._zero(values)
        return values

    def _load_device_graph(self) -> None:
        cfg = self.config
        self.embedding = self._matrix(self.model.tok_embd)
        self.output = self._matrix(self.model.output)
        self.output_norm = self._upload_vector(self.model.out_norm)
        self.position_lanes = self._empty((4,))
        self._actual_auxiliary_allocations["position_lanes"] = int(
            self.position_lanes.nbytes
        )
        self.rope_sections = self._cuda.to_device(self._np.asarray(
            cfg.rope_dimension_sections, dtype=self._np.int32,
        ))
        self._actual_auxiliary_allocations["rope_sections"] = int(
            self.rope_sections.nbytes
        )
        self._counters["setup_host_to_device_bytes"] += int(
            self.rope_sections.nbytes
        )

        for layer_index, weights in enumerate(self.model.layers):
            shared = self._shared(weights.shared)
            if isinstance(weights, Qwen35RecurrentWeights):
                conv_matrix = self._matrix(weights.conv)
                # The tiny fixture's depthwise filter is native dense F32 and
                # already has GGUF rows [channel, chronological tap].
                conv_weights = conv_matrix.f32_device_view()
                conv_shape = (cfg.conv_kernel - 1, cfg.conv_width)
                delta_shape = (
                    cfg.time_step_rank, cfg.state_size, cfg.state_size,
                )
                self.layers.append(_RecurrentLayer(
                    layer_index=layer_index,
                    shared=shared,
                    qkv=self._matrix(weights.qkv),
                    gate=self._matrix(weights.gate),
                    alpha=self._matrix(weights.alpha),
                    beta=self._matrix(weights.beta),
                    output=self._matrix(weights.output),
                    conv_weights=conv_weights,
                    dt_bias=self._upload_vector(weights.dt_bias),
                    ssm_a=self._upload_vector(weights.ssm_a),
                    norm=self._upload_vector(weights.norm),
                    active_conv=self._fresh_zero(conv_shape),
                    staging_conv=self._empty(conv_shape),
                    active_delta=self._fresh_zero(delta_shape),
                    staging_delta=self._empty(delta_shape),
                ))
            elif isinstance(weights, Qwen35AttentionWeights):
                key_shape = (self.n_ctx, cfg.kv_heads, cfg.key_length)
                value_shape = (self.n_ctx, cfg.kv_heads, cfg.value_length)
                self.layers.append(_AttentionLayer(
                    layer_index=layer_index,
                    shared=shared,
                    query_gate=self._matrix(weights.query_gate),
                    key=self._matrix(weights.key),
                    value=self._matrix(weights.value),
                    output=self._matrix(weights.output),
                    query_norm=self._upload_vector(weights.query_norm),
                    key_norm=self._upload_vector(weights.key_norm),
                    active_key=self._fresh_zero(key_shape),
                    staging_key=self._empty(key_shape),
                    active_value=self._fresh_zero(value_shape),
                    staging_value=self._empty(value_shape),
                ))
            else:  # pragma: no cover - runtime loader already proves this
                raise TypeError(f"unsupported Qwen35 layer weights {type(weights)!r}")

    def _reconcile_device_allocations(self) -> None:
        planned_names = {
            tensor_name
            for group in self.placement_plan.groups
            if group.placement == "cuda"
            for tensor_name in group.tensor_names
        }
        actual_names = set(self._actual_weight_allocations)
        if actual_names != planned_names:
            missing = sorted(planned_names - actual_names)
            extra = sorted(actual_names - planned_names)
            raise ValueError(
                "device tensor allocations differ from placement plan; "
                f"missing={missing}, extra={extra}"
            )
        state = self._state_bytes()
        actual_state = state["allocated_ping_pong_state_bytes"]
        if actual_state != self.placement_plan.allocated_state_bytes:
            raise MemoryError(
                "device state allocation differs from placement plan: "
                f"{actual_state} != {self.placement_plan.allocated_state_bytes}"
            )
        group_allocations = {
            group.name: sum(
                self._actual_weight_allocations[name]
                for name in group.tensor_names
            )
            for group in self.placement_plan.groups
            if group.placement == "cuda"
        }
        auxiliary = sum(self._actual_auxiliary_allocations.values())
        report = verify_actual_allocations(
            self.placement_plan,
            group_allocations,
            documented_overhead_bytes=auxiliary,
        )
        report.update({
            "actual_tensor_bytes": dict(sorted(
                self._actual_weight_allocations.items()
            )),
            "actual_group_bytes": group_allocations,
            "actual_auxiliary_allocations": dict(sorted(
                self._actual_auxiliary_allocations.items()
            )),
            "actual_allocated_state_bytes": actual_state,
            "planned_activation_workspace_bytes": (
                self.placement_plan.activation_workspace_bytes
            ),
        })
        self.allocation_report = report

    def _rms(self, values: Any, weight: Any) -> Any:
        output = self._empty(tuple(values.shape))
        source = values.reshape(1, values.size)
        target = output.reshape(1, output.size)
        self._launch(
            self._kernels["rms_norm"], 1,
            source, weight, float(self.config.rms_epsilon), target,
        )
        return output

    def _residual(self, left: Any, right: Any) -> Any:
        output = self._empty(tuple(left.shape))
        self._launch(
            self._kernels["residual_add"], left.size,
            left.reshape(left.size), right.reshape(right.size),
            output.reshape(output.size),
        )
        return output

    def _matvec(self, matrix: PackedCudaMatrix, values: Any) -> Any:
        output = matrix.matvec_device(values.reshape(values.size))
        self._counters["launches"] += 1
        return output

    def _boundary(
        self,
        name: str,
        values: Any,
        trace: TraceCallback | None,
    ) -> None:
        if trace is not None:
            host = values.copy_to_host().reshape(-1)
            self._counters["trace_device_to_host_bytes"] += int(host.nbytes)
            self._counters["synchronizations"] += 1
            trace(name, host)
        if self._fault is None:
            return
        fault_name, remaining = self._fault
        if name != fault_name:
            return
        remaining -= 1
        if remaining:
            self._fault = (fault_name, remaining)
            return
        self._fault = None
        raise Qwen35CudaGraphFault(f"injected device graph fault at {name}")

    def _stage_state(self) -> None:
        for layer in self.layers:
            if isinstance(layer, _RecurrentLayer):
                if self._transaction_in_place:
                    self._copy(layer.active_conv, layer.staging_conv)
                    self._copy(layer.active_delta, layer.staging_delta)
                    self._counters[
                        "routine_recurrent_device_to_device_bytes"
                    ] += int(
                        layer.active_conv.nbytes + layer.active_delta.nbytes
                    )
            else:
                # A multi-token commit leaves the old active buffer behind by
                # exactly that chunk. Repair only the missing suffix, never
                # the complete allocated K/V cache.
                if self._attention_staging_prefix < self.position:
                    rows = slice(self._attention_staging_prefix, self.position)
                    self._copy(layer.active_key[rows], layer.staging_key[rows])
                    self._copy(layer.active_value[rows], layer.staging_value[rows])
        self._attention_staging_prefix = self.position

    def _publish_state(self) -> None:
        for layer in self.layers:
            layer.swap_state()

    def begin_step_or_chunk(self, *, expected_tokens: int = 1) -> int:
        """Open one device transaction and stage every mutable state buffer."""

        self._ensure_open()
        if self._dirty:
            raise RuntimeError("Qwen35 CUDA graph transaction is already active")
        if isinstance(expected_tokens, bool) or not isinstance(expected_tokens, int):
            raise TypeError("expected_tokens must be an integer")
        if expected_tokens <= 0:
            raise ValueError("expected_tokens must be positive")
        self._dirty = True
        self._transaction_start_position = self.position
        self._transaction_tokens = []
        self._transaction_expected_tokens = expected_tokens
        self._transaction_in_place = expected_tokens > 1
        self.last_fault = ""
        try:
            self._stage_state()
        except Exception as exc:
            self.last_fault = f"{type(exc).__name__}: {exc}"
            self._dirty = False
            self._transaction_start_position = None
            self._transaction_expected_tokens = 0
            self._transaction_in_place = False
            self._attention_staging_prefix = 0
            raise
        return self.generation

    def commit_step_or_chunk(self, token_ids: Sequence[int]) -> None:
        """Atomically publish a successfully completed device transaction."""

        if not self._dirty:
            raise RuntimeError("Qwen35 CUDA graph has no active transaction")
        committed = [int(token_id) for token_id in token_ids]
        if not committed:
            raise ValueError("a CUDA transaction must commit at least one token")
        if committed != self._transaction_tokens:
            raise RuntimeError(
                "CUDA transaction commit does not match staged token work: "
                f"committed={committed}, staged={self._transaction_tokens}"
            )
        self._publish_state()
        assert self._transaction_start_position is not None
        self._attention_staging_prefix = self._transaction_start_position
        self.token_ids.extend(committed)
        self.generation += len(committed)
        self._counters["tokens"] += len(committed)
        self._dirty = False
        self._transaction_start_position = None
        self._transaction_tokens = []
        self._transaction_expected_tokens = 0
        self._transaction_in_place = False

    def abort_step_or_chunk(self, error: BaseException | str | None = None) -> None:
        """Discard staging state; the active buffers remain authoritative."""

        if not self._dirty:
            raise RuntimeError("Qwen35 CUDA graph has no active transaction")
        if error is not None:
            self.last_fault = (
                str(error) if isinstance(error, str)
                else f"{type(error).__name__}: {error}"
            )
        assert self._transaction_start_position is not None
        self.position = self._transaction_start_position
        self._attention_staging_prefix = self.position
        self._dirty = False
        self._transaction_start_position = None
        self._transaction_tokens = []
        self._transaction_expected_tokens = 0
        self._transaction_in_place = False

    def advance_step_or_chunk(
        self,
        token_id: int,
        *,
        trace: TraceCallback | None = None,
        download_logits: bool = False,
    ) -> array | None:
        """Stage one token inside the current transaction without publishing."""

        if not self._dirty:
            raise RuntimeError("begin_step_or_chunk must precede staged token work")
        if len(self._transaction_tokens) >= self._transaction_expected_tokens:
            raise RuntimeError(
                "CUDA transaction staged more tokens than declared at begin"
            )
        token_id = int(token_id)
        if token_id < 0 or token_id >= self.config.vocabulary_size:
            raise ValueError(
                f"token ID {token_id} is outside vocabulary size "
                f"{self.config.vocabulary_size}"
            )
        if self.position >= self.n_ctx:
            raise RuntimeError(f"context window full ({self.n_ctx} tokens)")
        self._counters["routine_token_control_host_to_device_bytes"] += 8
        logits_device = self._advance_device(token_id, trace)
        logits_host = None
        if download_logits:
            logits_host = logits_device.copy_to_host()
            self._counters["routine_logits_device_to_host_bytes"] += int(
                logits_host.nbytes
            )
            self._counters["synchronizations"] += 1
        self._transaction_tokens.append(token_id)
        self.position += 1
        return None if logits_host is None else array("f", logits_host)

    def _recurrent(
        self,
        layer: _RecurrentLayer,
        normalized: Any,
        trace: TraceCallback | None,
    ) -> Any:
        cfg = self.config
        prefix = f"layer.{layer.layer_index}"
        mixed = self._matvec(layer.qkv, normalized)
        gate = self._matvec(layer.gate, normalized)
        alpha0 = self._matvec(layer.alpha, normalized)
        beta0 = self._matvec(layer.beta, normalized)
        self._boundary(prefix + ".linear_attn_qkv_mixed", mixed, trace)

        conv_source = (
            layer.staging_conv if self._transaction_in_place
            else layer.active_conv
        )
        if trace is not None:
            raw = self._empty((cfg.conv_width,))
            self._launch(
                self._kernels["conv_raw_observe"], cfg.conv_width,
                mixed, layer.conv_weights, conv_source, raw,
            )
            self._boundary(prefix + ".conv_output_raw", raw, trace)
        activated = self._empty((cfg.conv_width,))
        if self._transaction_in_place:
            self._launch(
                self._kernels["conv_step"], cfg.conv_width,
                mixed, layer.conv_weights, layer.staging_conv, activated,
            )
        else:
            self._launch(
                self._kernels["conv_step_out"], cfg.conv_width,
                mixed, layer.conv_weights,
                layer.active_conv, layer.staging_conv, activated,
            )
        self._boundary(prefix + ".conv_output_silu", activated, trace)

        qk_width = cfg.group_count * cfg.state_size
        queries_source = activated[:qk_width].reshape(
            cfg.group_count, cfg.state_size,
        )
        keys_source = activated[qk_width:2 * qk_width].reshape(
            cfg.group_count, cfg.state_size,
        )
        values = activated[2 * qk_width:].reshape(
            cfg.time_step_rank, cfg.state_size,
        )
        queries = self._empty((cfg.group_count, cfg.state_size))
        keys = self._empty((cfg.group_count, cfg.state_size))
        self._launch(
            self._kernels["normalize_heads"], cfg.group_count,
            queries_source, queries, float(cfg.rms_epsilon),
        )
        self._launch(
            self._kernels["normalize_heads"], cfg.group_count,
            keys_source, keys, float(cfg.rms_epsilon),
        )
        self._boundary(prefix + ".Qcur_normed", queries, trace)
        self._boundary(prefix + ".Kcur_normed", keys, trace)

        beta = self._empty((cfg.time_step_rank,))
        log_decay = self._empty((cfg.time_step_rank,))
        self._launch(
            self._kernels["beta_decay"], cfg.time_step_rank,
            beta0, alpha0, layer.dt_bias, layer.ssm_a, beta, log_decay,
        )
        state_output = self._empty((cfg.time_step_rank, cfg.state_size))
        state_source = (
            layer.staging_delta if self._transaction_in_place
            else layer.active_delta
        )
        self._boundary(prefix + ".state_before", state_source, trace)
        if self._transaction_in_place:
            self._launch(
                self._kernels["gdn_step"], cfg.time_step_rank,
                layer.staging_delta, queries, keys, values,
                beta, log_decay, state_output,
            )
        else:
            self._launch(
                self._kernels["gdn_step_out"], cfg.time_step_rank,
                layer.active_delta, layer.staging_delta,
                queries, keys, values, beta, log_decay, state_output,
            )
        self._boundary(prefix + ".new_state", layer.staging_delta, trace)
        gated = self._empty((cfg.time_step_rank, cfg.state_size))
        self._launch(
            self._kernels["gated_rms"], cfg.time_step_rank,
            state_output, layer.norm,
            gate.reshape(cfg.time_step_rank, cfg.state_size),
            float(cfg.rms_epsilon), gated,
        )
        self._boundary(prefix + ".attn_gated", gated, trace)
        output = self._matvec(layer.output, gated.reshape(gated.size))
        self._boundary(prefix + ".attn_output", output, trace)
        return output

    def _attention(
        self,
        layer: _AttentionLayer,
        normalized: Any,
        trace: TraceCallback | None,
    ) -> Any:
        cfg = self.config
        prefix = f"layer.{layer.layer_index}"
        query_gate = self._matvec(layer.query_gate, normalized)
        key_flat = self._matvec(layer.key, normalized)
        value_flat = self._matvec(layer.value, normalized)

        queries_source = self._empty((cfg.query_heads, cfg.key_length))
        gates = self._empty((cfg.query_heads, cfg.key_length))
        self._launch(
            self._kernels["split_query_gate"], queries_source.size,
            query_gate, queries_source, gates,
        )
        key_source = key_flat.reshape(cfg.kv_heads, cfg.key_length)
        value_heads = value_flat.reshape(cfg.kv_heads, cfg.value_length)
        queries_norm = self._empty(tuple(queries_source.shape))
        keys_norm = self._empty(tuple(key_source.shape))
        self._launch(
            self._kernels["rms_norm"], cfg.query_heads,
            queries_source, layer.query_norm, float(cfg.rms_epsilon), queries_norm,
        )
        self._launch(
            self._kernels["rms_norm"], cfg.kv_heads,
            key_source, layer.key_norm, float(cfg.rms_epsilon), keys_norm,
        )
        queries = self._empty(tuple(queries_norm.shape))
        keys = self._empty(tuple(keys_norm.shape))
        self._launch(
            self._kernels["imrope"], cfg.query_heads,
            queries_norm, self.position_lanes, self.rope_sections,
            float(cfg.rope_freq_base), 1.0, queries,
        )
        self._launch(
            self._kernels["imrope"], cfg.kv_heads,
            keys_norm, self.position_lanes, self.rope_sections,
            float(cfg.rope_freq_base), 1.0, keys,
        )
        self._launch(
            self._kernels["kv_append"], keys.size,
            keys, value_heads, layer.staging_key, layer.staging_value,
            self.position,
        )
        self._boundary(prefix + ".Qcur_normed", queries, trace)
        self._boundary(prefix + ".Kcur_normed", keys, trace)

        attended = self._empty((cfg.query_heads, cfg.value_length))
        self._zero(attended)
        logical = self.position + 1
        self._launch(
            self._kernels["attention_decode"], cfg.query_heads,
            queries,
            layer.staging_key[:logical],
            layer.staging_value[:logical],
            attended,
        )
        self._boundary(prefix + ".attn_pregate", attended, trace)
        gated = self._empty(tuple(attended.shape))
        self._launch(
            self._kernels["sigmoid_gate"], attended.size,
            attended.reshape(attended.size), gates.reshape(gates.size),
            gated.reshape(gated.size),
        )
        self._boundary(prefix + ".attn_gated", gated, trace)
        output = self._matvec(layer.output, gated.reshape(gated.size))
        self._boundary(prefix + ".attn_output", output, trace)
        return output

    def _ffn(
        self,
        layer_index: int,
        shared: _SharedLayer,
        values: Any,
        trace: TraceCallback | None,
    ) -> Any:
        normalized = self._rms(values, shared.post_attention_norm)
        gate = self._matvec(shared.ffn_gate, normalized)
        up = self._matvec(shared.ffn_up, normalized)
        activated = self._empty(tuple(gate.shape))
        self._launch(
            self._kernels["swiglu"], gate.size,
            gate, up, activated,
        )
        output = self._matvec(shared.ffn_down, activated)
        self._boundary(f"layer.{layer_index}.ffn_out", output, trace)
        return self._residual(values, output)

    def _advance_device(
        self, token_id: int, trace: TraceCallback | None,
    ) -> Any:
        values = self.embedding.embedding_row_device(token_id)
        self._counters["launches"] += 1
        self._launch(
            self._graph["set_text_position"], 4,
            self.position_lanes, self.position,
        )
        for layer in self.layers:
            residual = values
            normalized = self._rms(values, layer.shared.attention_norm)
            self._boundary(
                f"layer.{layer.layer_index}.attn_norm", normalized, trace,
            )
            if isinstance(layer, _RecurrentLayer):
                attention = self._recurrent(layer, normalized, trace)
            else:
                attention = self._attention(layer, normalized, trace)
            values = self._residual(residual, attention)
            values = self._ffn(layer.layer_index, layer.shared, values, trace)
        normalized = self._rms(values, self.output_norm)
        logits = self._matvec(self.output, normalized)
        self._boundary("model.result_output", logits, trace)
        return logits

    def _forward_one(
        self,
        token_id: int,
        *,
        trace: TraceCallback | None = None,
        download_logits: bool,
    ) -> array | None:
        """Advance one atomic token, optionally retaining logits on device."""

        self._ensure_open()
        if self._dirty:
            raise RuntimeError("Qwen35 CUDA graph does not support reentrant forward")
        if token_id < 0 or token_id >= self.config.vocabulary_size:
            raise ValueError(
                f"token ID {token_id} is outside vocabulary size "
                f"{self.config.vocabulary_size}"
            )
        if self.position >= self.n_ctx:
            raise RuntimeError(f"context window full ({self.n_ctx} tokens)")
        try:
            self.begin_step_or_chunk(expected_tokens=1)
            logits_host = self.advance_step_or_chunk(
                token_id, trace=trace, download_logits=download_logits,
            )
        except Exception as exc:
            if self._dirty:
                self.abort_step_or_chunk(exc)
            raise
        else:
            self.commit_step_or_chunk((token_id,))
            return None if logits_host is None else array("f", logits_host)

    def forward(
        self,
        token_id: int,
        *,
        trace: TraceCallback | None = None,
    ) -> array:
        """Advance one token atomically and return the sole routine D2H value."""

        logits = self._forward_one(
            token_id, trace=trace, download_logits=True,
        )
        assert logits is not None
        return logits

    forward_token = forward

    @property
    def n_past(self) -> int:
        return self.position

    def prefill(
        self,
        token_ids: Sequence[int],
        *,
        chunk_size: int = 1,
        trace: TraceCallback | None = None,
    ) -> array | None:
        """Advance a prompt while downloading only its final logits.

        ``chunk_size`` locks the public prefill contract and logical chunk
        boundaries.  The simulator graph remains deliberately decomposed;
        packed batched projection tuning is a later real-device concern.
        """

        self._ensure_open()
        if chunk_size <= 0:
            raise ValueError("prefill chunk_size must be positive")
        tokens = list(token_ids)
        if self.position + len(tokens) > self.n_ctx:
            raise RuntimeError(f"context window full ({self.n_ctx} tokens)")
        if not tokens:
            return None
        result: array | None = None
        final_index = len(tokens) - 1
        for start in range(0, len(tokens), chunk_size):
            stop = min(start + chunk_size, len(tokens))
            chunk = tokens[start:stop]
            try:
                self.begin_step_or_chunk(expected_tokens=len(chunk))
                for index, token_id in enumerate(chunk, start=start):
                    result = self.advance_step_or_chunk(
                        token_id,
                        trace=trace,
                        download_logits=index == final_index,
                    )
                self.commit_step_or_chunk(chunk)
            except BaseException as exc:
                if self._dirty:
                    self.abort_step_or_chunk(exc)
                raise
        assert result is not None
        return result

    def arm_fault(self, stage_name: str, *, occurrence: int = 1) -> None:
        if not stage_name:
            raise ValueError("fault stage name cannot be empty")
        if occurrence <= 0:
            raise ValueError("fault occurrence must be positive")
        self._fault = (stage_name, occurrence)

    def _host_state(self) -> ModelState:
        state = self.model.new_state()
        pinned_buffers: list[Any] = []

        def checkpoint_copy(device_values: Any) -> Any:
            if not device_values.size:
                return self._np.empty(
                    tuple(device_values.shape), dtype=self._np.float32,
                )
            host = self._cuda.pinned_array(
                tuple(device_values.shape), dtype=self._np.float32,
            )
            device_values.copy_to_host(host)
            pinned_buffers.append(host)
            return host

        for device_layer in self.layers:
            host_layer = state.layer(device_layer.layer_index)
            if isinstance(device_layer, _RecurrentLayer):
                if not isinstance(host_layer, RecurrentState):
                    raise TypeError("device recurrent layer lacks host state bridge")
                conv = checkpoint_copy(device_layer.active_conv)
                delta = checkpoint_copy(device_layer.active_delta)
                self._counters["checkpoint_device_to_host_bytes"] += int(
                    conv.nbytes + delta.nbytes
                )
                host_layer.conv_history = array("f", conv.reshape(-1))
                host_layer.delta_matrix = array("f", delta.reshape(-1))
                host_layer.logical_position = self.position
            else:
                if not isinstance(host_layer, FullAttentionState):
                    raise TypeError("device attention layer lacks host state bridge")
                keys = checkpoint_copy(device_layer.active_key[:self.position])
                values = checkpoint_copy(
                    device_layer.active_value[:self.position]
                )
                self._counters["checkpoint_device_to_host_bytes"] += int(
                    keys.nbytes + values.nbytes
                )
                host_layer.key = array("f", keys.reshape(-1))
                host_layer.value = array("f", values.reshape(-1))
                host_layer.logical_length = self.position
        self._counters["synchronizations"] += 1
        state.position = self.position
        state.token_ids = list(self.token_ids)
        state.generation = self.generation
        state._validate_live()
        # Publish pinned buffers only after every component copied and the
        # complete host state validated. A failed export retains the previous
        # last-known-good checkpoint storage.
        self._last_pinned_checkpoint_buffers = tuple(pinned_buffers)
        return state

    def snapshot(self) -> StateSnapshot:
        self._ensure_open()
        if self._dirty:
            raise RuntimeError("cannot snapshot a dirty CUDA graph transaction")
        snapshot = self._host_state().snapshot()
        self._last_complete_host_checkpoint = snapshot
        return snapshot

    checkpoint = snapshot

    def checkpoint_to_host(self) -> StateSnapshot:
        """Copy a complete immutable checkpoint from device authority."""

        return self.snapshot()

    def branch(self) -> StateSnapshot:
        """Return a complete immutable branch point for later restoration."""

        return self.checkpoint_to_host()

    def restore(self, snapshot: StateSnapshot) -> None:
        self._ensure_open()
        if self._dirty:
            raise RuntimeError("cannot restore a dirty CUDA graph transaction")
        host = self.model.new_state()
        host.restore(snapshot)
        prepared: list[tuple[_DeviceLayer, Any, Any]] = []
        transferred = 0
        for device_layer in self.layers:
            host_layer = host.layer(device_layer.layer_index)
            if isinstance(device_layer, _RecurrentLayer):
                assert isinstance(host_layer, RecurrentState)
                conv = self._np.asarray(host_layer.conv_history, dtype=self._np.float32)
                conv = conv.reshape(tuple(device_layer.active_conv.shape))
                delta = self._np.asarray(host_layer.delta_matrix, dtype=self._np.float32)
                delta = delta.reshape(tuple(device_layer.active_delta.shape))
                first = self._cuda.to_device(conv)
                second = self._cuda.to_device(delta)
            else:
                assert isinstance(host_layer, FullAttentionState)
                key_shape = tuple(device_layer.active_key.shape)
                value_shape = tuple(device_layer.active_value.shape)
                keys = self._np.zeros(key_shape, dtype=self._np.float32)
                values = self._np.zeros(value_shape, dtype=self._np.float32)
                if host.position:
                    keys[:host.position] = self._np.asarray(
                        host_layer.key, dtype=self._np.float32,
                    ).reshape((host.position, *key_shape[1:]))
                    values[:host.position] = self._np.asarray(
                        host_layer.value, dtype=self._np.float32,
                    ).reshape((host.position, *value_shape[1:]))
                first = self._cuda.to_device(keys)
                second = self._cuda.to_device(values)
            transferred += int(first.nbytes + second.nbytes)
            prepared.append((device_layer, first, second))
        for device_layer, first, second in prepared:
            if isinstance(device_layer, _RecurrentLayer):
                device_layer.staging_conv = first
                device_layer.staging_delta = second
            else:
                device_layer.staging_key = first
                device_layer.staging_value = second
            device_layer.swap_state()
            if isinstance(device_layer, _AttentionLayer):
                # Re-establish the incremental ping-pong invariant at this
                # explicit checkpoint boundary.  This is device-to-device and
                # outside routine token traffic.
                self._copy(device_layer.active_key, device_layer.staging_key)
                self._copy(device_layer.active_value, device_layer.staging_value)
        self.position = host.position
        self.token_ids = list(host.token_ids)
        self.generation = host.generation
        self.last_fault = ""
        self._attention_staging_prefix = self.position
        self._counters["checkpoint_host_to_device_bytes"] += transferred
        self._counters["synchronizations"] += 1

    def restore_to_device(self, snapshot: StateSnapshot) -> None:
        """Validate and atomically publish a complete host checkpoint."""

        self.restore(snapshot)

    def reset(self, *, clear_prefixes: bool = False) -> None:
        del clear_prefixes  # device graph has no implicit prefix slots
        self.restore(self.model.new_state().snapshot())

    @property
    def state(self) -> ModelState:
        """Materialize host state only for explicit diagnostics/validation."""

        self._ensure_open()
        if self._dirty:
            raise RuntimeError("cannot export state from a dirty CUDA graph")
        return self._host_state()

    def truncate(self, position: int) -> None:
        """Reconstruct a hybrid prefix on the host, then publish it atomically."""

        if isinstance(position, bool) or not isinstance(position, int):
            raise TypeError("truncate position must be an integer")
        if position < 0:
            raise ValueError("truncate position must be non-negative")
        if position > self.position:
            raise ValueError(
                f"cannot extend CUDA state from {self.position} to {position}"
            )
        if position == self.position:
            return
        prefix = tuple(self.token_ids[:position])
        prepared = self.model.new_state()
        for token_id in prefix:
            self.model.forward_token(token_id, prepared)
        self.restore(prepared.snapshot())

    def replay(
        self,
        token_ids: Sequence[int],
        *,
        chunk_size: int | None = None,
    ) -> array | None:
        """Atomically append a replay suffix using complete device chunks."""

        tokens = tuple(int(token_id) for token_id in token_ids)
        if not tokens:
            return None
        return self.prefill(
            tokens,
            chunk_size=len(tokens) if chunk_size is None else chunk_size,
        )

    def state_buffer_ids(self) -> tuple[tuple[int, ...], ...]:
        """Return both persistent ping-pong allocation identities per layer."""

        result: list[tuple[int, ...]] = []
        for layer in self.layers:
            if isinstance(layer, _RecurrentLayer):
                result.append(tuple(sorted((
                    id(layer.active_conv), id(layer.staging_conv),
                    id(layer.active_delta), id(layer.staging_delta),
                ))))
            else:
                result.append(tuple(sorted((
                    id(layer.active_key), id(layer.staging_key),
                    id(layer.active_value), id(layer.staging_value),
                ))))
        return tuple(result)

    def _state_bytes(self) -> dict[str, int]:
        recurrent = convolution = attention = 0
        for layer in self.layers:
            if isinstance(layer, _RecurrentLayer):
                convolution += int(layer.active_conv.nbytes)
                recurrent += int(layer.active_delta.nbytes)
            else:
                attention += int(
                    layer.active_key.nbytes + layer.active_value.nbytes
                )
        logical = recurrent + convolution + attention
        return {
            "recurrent_matrix_bytes": recurrent,
            "convolution_history_bytes": convolution,
            "attention_kv_bytes": attention,
            "logical_device_state_bytes": logical,
            "allocated_ping_pong_state_bytes": 2 * logical,
        }

    def transfer_limits(self) -> dict[str, Any]:
        values: dict[str, Any] = dict(self._counters)
        values.update({
            "routine_host_to_device_bytes": self._counters[
                "routine_token_control_host_to_device_bytes"
            ],
            "routine_device_to_host_bytes": self._counters[
                "routine_logits_device_to_host_bytes"
            ],
            "no_routine_state_transfer": (
                self._counters["routine_state_host_to_device_bytes"] == 0
                and self._counters["routine_state_device_to_host_bytes"] == 0
            ),
            "no_routine_intermediate_transfer": (
                self._counters["routine_intermediate_host_to_device_bytes"] == 0
                and self._counters["routine_intermediate_device_to_host_bytes"] == 0
            ),
        })
        return values

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "production_qualified": self.production_qualified,
            "qualification": self.qualification,
            "simulator": self.simulator,
            "position": self.position,
            "generation": self.generation,
            "dirty_transaction": self._dirty,
            "staged_token_count": len(self._transaction_tokens),
            "last_fault": self.last_fault,
            "checkpoint_policy": {
                "authority": "device-between-explicit-checkpoints",
                "interval_tokens": self.checkpoint_interval,
                "pinned_host_buffers": bool(
                    self._last_pinned_checkpoint_buffers
                ),
                "last_complete_generation": (
                    None if self._last_complete_host_checkpoint is None
                    else self._last_complete_host_checkpoint.generation
                ),
            },
            "device_matrix_count": len(self._matrices),
            "device_matrix_dtypes": sorted({
                matrix.dtype for matrix in self._matrices.values()
            }),
            "all_weights_native_packed": all(
                matrix.descriptor()["native_packed"]
                for matrix in self._matrices.values()
            ),
            "device_weight_bytes": sum(
                int(matrix.device_packed_bytes)
                for matrix in self._matrices.values()
            ),
            "placement_plan": self.placement_plan.descriptor(),
            "serialized_placement_plan": self.serialized_placement_plan,
            "allocation_reconciliation": self.allocation_report,
            "device_state_bytes": self._state_bytes(),
            "transfers": self.transfer_limits(),
        }

    def describe_data(self) -> dict[str, Any]:
        """Expose the common loaded-model diagnostics used by product gates."""

        state = self._state_bytes()
        actual = self.allocation_report
        return {
            "architecture": "qwen35",
            "backend": self.backend_name,
            "backend_selection": dict(getattr(self, "backend_selection", {})),
            "executable": True,
            "context": self.n_ctx,
            "state": {
                "position": self.position,
                "generation": self.generation,
            },
            "memory": {
                "packed_weight_bytes": actual.get("actual_weight_bytes", 0),
                "recurrent_matrix_bytes": state["recurrent_matrix_bytes"],
                "convolution_history_bytes": state[
                    "convolution_history_bytes"
                ],
                "attention_kv_capacity_bytes": state["attention_kv_bytes"],
                "attention_kv_dtype": "f32",
                "live_state_bytes": state["logical_device_state_bytes"],
                "actual_cuda_vram_bytes": actual.get(
                    "actual_total_device_bytes", 0,
                ),
            },
            "placement": self.placement_plan.descriptor(),
            "checkpoint_policy": self.describe()["checkpoint_policy"],
            "cuda": self._kernels["capability"].descriptor(),
            "unsupported_features": self.model.manifest[
                "unsupported_features"
            ],
        }

    def close(self) -> None:
        if self._closed:
            return
        for matrix in self._matrices.values():
            matrix.close()
        self._matrices.clear()
        self.layers.clear()
        self._source.close()
        if self._owns_model:
            self.model.close()
        self._closed = True

    def __enter__(self) -> "Qwen35CudaSimulatorGraph":
        self._ensure_open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


Qwen35CudaDeviceGraph = Qwen35CudaSimulatorGraph


__all__ = [
    "Qwen35CudaDeviceGraph", "Qwen35CudaGraphFault",
    "Qwen35CudaSimulatorGraph",
]
