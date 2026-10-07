# Alpaccaroo - backend-neutral model memory and immutable snapshots.
# MIT License. See LICENSE.
"""State ownership, snapshots, rollback, and prefix-cache accounting.

This module deliberately depends only on the standard library.  Live CPU
state is held in ``array('f')`` buffers.  Snapshots encode those arrays as
immutable little-endian bytes so a prefix slot can never alias writable live
state.

The classes here describe state semantics; architecture backends remain
responsible for doing model math and for advancing every layer before calling
``ModelState.record_token``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from array import array
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence


STATE_SCHEMA_VERSION = 1
MEMORY_REPRESENTATION_VERSION = 1
_F32_ITEMSIZE = array("f").itemsize
if _F32_ITEMSIZE != 4:  # pragma: no cover - CPython's supported platforms
    raise RuntimeError("Alpaccaroo state requires 32-bit array('f') items")


class StateError(ValueError):
    """A live state or snapshot violates its declared schema."""


class SnapshotMismatchError(StateError):
    """A snapshot does not belong to the receiving model state."""


class ReplayRequiredError(StateError):
    """A recurrent state operation needs checkpoint restore plus replay."""


def _strict_nonnegative(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StateError(f"{name} must be a non-negative integer, got {value!r}")


def _strict_positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise StateError(f"{name} must be a positive integer, got {value!r}")


def _float_array(values: Iterable[float], *, expected: int, what: str) -> array:
    try:
        result = array("f", values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise StateError(f"{what} is not a valid float32 buffer: {exc}") from exc
    if len(result) != expected:
        raise StateError(
            f"{what} has {len(result)} float32 values, expected {expected}"
        )
    return result


def _zero_floats(count: int) -> array:
    return array("f", [0.0]) * count


def _f32_to_bytes(values: array) -> bytes:
    if values.typecode != "f":
        raise StateError(f"state buffer type must be 'f', got {values.typecode!r}")
    if sys.byteorder == "little":
        return values.tobytes()
    copied = array("f", values)
    copied.byteswap()
    return copied.tobytes()


def _f32_from_bytes(data: bytes, *, expected: int, what: str) -> array:
    _validate_f32_bytes(data, expected=expected, what=what)
    result = array("f")
    result.frombytes(data)
    if sys.byteorder != "little":
        result.byteswap()
    return result


def _validate_f32_bytes(data: bytes, *, expected: int, what: str) -> None:
    if not isinstance(data, bytes):
        raise StateError(f"{what} payload must be immutable bytes")
    expected_bytes = expected * _F32_ITEMSIZE
    if len(data) != expected_bytes:
        raise StateError(
            f"{what} has {len(data)} bytes, expected {expected_bytes}"
        )


@dataclass(frozen=True, slots=True)
class LayerMemorySpec:
    """Immutable, fingerprinted description of one layer's live memory."""

    layer_index: int
    kind: str
    kv_heads: int = 0
    key_dim: int = 0
    value_dim: int = 0
    conv_channels: int = 0
    conv_kernel: int = 0
    recurrent_key_heads: int = 0
    recurrent_value_heads: int = 0
    recurrent_key_dim: int = 0
    recurrent_value_dim: int = 0

    def __post_init__(self) -> None:
        _strict_nonnegative("layer_index", self.layer_index)
        if self.kind not in ("attention", "recurrent", "none"):
            raise StateError(
                "layer memory kind must be 'attention', 'recurrent', or 'none', "
                f"got {self.kind!r}"
            )
        dimensions = (
            "kv_heads", "key_dim", "value_dim", "conv_channels",
            "conv_kernel", "recurrent_key_heads", "recurrent_value_heads",
            "recurrent_key_dim", "recurrent_value_dim",
        )
        for name in dimensions:
            _strict_nonnegative(name, getattr(self, name))

        attention = (self.kv_heads, self.key_dim, self.value_dim)
        recurrent = (
            self.conv_channels, self.conv_kernel, self.recurrent_key_heads,
            self.recurrent_value_heads, self.recurrent_key_dim,
            self.recurrent_value_dim,
        )
        if self.kind == "attention":
            if not all(attention):
                raise StateError("attention memory requires positive K/V dimensions")
            if any(recurrent):
                raise StateError("attention memory cannot declare recurrent dimensions")
        elif self.kind == "recurrent":
            if not all(recurrent):
                raise StateError("recurrent memory requires positive recurrent dimensions")
            if any(attention):
                raise StateError("recurrent memory cannot declare attention dimensions")
        elif any(attention) or any(recurrent):
            raise StateError("none memory cannot declare attention or recurrent dimensions")

    def descriptor(self) -> dict[str, int | str]:
        return {
            "layer_index": self.layer_index,
            "kind": self.kind,
            "kv_heads": self.kv_heads,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "conv_channels": self.conv_channels,
            "conv_kernel": self.conv_kernel,
            "recurrent_key_heads": self.recurrent_key_heads,
            "recurrent_value_heads": self.recurrent_value_heads,
            "recurrent_key_dim": self.recurrent_key_dim,
            "recurrent_value_dim": self.recurrent_value_dim,
        }


@dataclass(frozen=True, slots=True)
class StateIdentity:
    """All non-token identity fields that make state safe to reuse."""

    architecture: str
    model_fingerprint: str
    weights_identity: str
    adapter_identity: str = ""
    positioning: str = ""
    numerical_mode: str = "f32"
    memory_version: int = MEMORY_REPRESENTATION_VERSION

    def __post_init__(self) -> None:
        for name in (
            "architecture", "model_fingerprint", "weights_identity",
            "numerical_mode",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise StateError(f"{name} must be a non-empty string")
        if not isinstance(self.adapter_identity, str):
            raise StateError("adapter_identity must be a string")
        if not isinstance(self.positioning, str):
            raise StateError("positioning must be a string")
        _strict_positive("memory_version", self.memory_version)

    def descriptor(self) -> dict[str, int | str]:
        return {
            "architecture": self.architecture,
            "model_fingerprint": self.model_fingerprint,
            "weights_identity": self.weights_identity,
            "adapter_identity": self.adapter_identity,
            "positioning": self.positioning,
            "numerical_mode": self.numerical_mode,
            "memory_version": self.memory_version,
        }


def descriptor_fingerprint(
    identity: StateIdentity,
    specs: Sequence[LayerMemorySpec],
    schema_version: int = STATE_SCHEMA_VERSION,
) -> str:
    """Return the stable digest used in snapshot validation and prefix keys."""

    _strict_positive("schema_version", schema_version)
    payload = {
        "identity": identity.descriptor(),
        "schema_version": schema_version,
        "layers": [spec.descriptor() for spec in specs],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class FullAttentionSnapshot:
    layer_index: int
    logical_length: int
    kv_heads: int
    key_dim: int
    value_dim: int
    key: bytes
    value: bytes
    dtype: str = "f32-le"
    kind: str = "attention"

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", bytes(self.key))
        object.__setattr__(self, "value", bytes(self.value))

    def memory_bytes(self) -> int:
        return len(self.key) + len(self.value)


@dataclass(frozen=True, slots=True)
class RecurrentSnapshot:
    layer_index: int
    logical_position: int
    conv_channels: int
    conv_kernel: int
    recurrent_key_heads: int
    recurrent_value_heads: int
    recurrent_key_dim: int
    recurrent_value_dim: int
    conv_history: bytes
    delta_matrix: bytes
    dtype: str = "f32-le"
    kind: str = "recurrent"

    def __post_init__(self) -> None:
        object.__setattr__(self, "conv_history", bytes(self.conv_history))
        object.__setattr__(self, "delta_matrix", bytes(self.delta_matrix))

    def memory_bytes(self) -> int:
        return len(self.conv_history) + len(self.delta_matrix)


@dataclass(frozen=True, slots=True)
class EmptyLayerSnapshot:
    layer_index: int
    kind: str = "none"

    def memory_bytes(self) -> int:
        return 0


LayerSnapshot = FullAttentionSnapshot | RecurrentSnapshot | EmptyLayerSnapshot


@dataclass(frozen=True, slots=True)
class PrefixKey:
    """Complete cache identity; no process-local mutable object is included."""

    architecture: str
    schema_version: int
    model_fingerprint: str
    weights_identity: str
    adapter_identity: str
    positioning: str
    memory_version: int
    numerical_mode: str
    descriptor_fingerprint: str
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_ids", tuple(self.token_ids))

    def memory_bytes(self) -> int:
        text = (
            self.architecture + self.model_fingerprint + self.weights_identity
            + self.adapter_identity + self.positioning + self.numerical_mode
            + self.descriptor_fingerprint
        )
        return len(text.encode("utf-8")) + 16 + 8 * len(self.token_ids)


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    """A complete, deeply immutable host snapshot."""

    identity: StateIdentity
    schema_version: int
    descriptor_fingerprint: str
    descriptors: tuple[LayerMemorySpec, ...]
    position: int
    token_ids: tuple[int, ...]
    generation: int
    layers: tuple[LayerSnapshot, ...]
    byte_cost: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "descriptors", tuple(self.descriptors))
        object.__setattr__(self, "token_ids", tuple(self.token_ids))
        object.__setattr__(self, "layers", tuple(self.layers))

    def memory_bytes(self) -> int:
        return self.byte_cost

    def prefix_key(self) -> PrefixKey:
        return PrefixKey(
            architecture=self.identity.architecture,
            schema_version=self.schema_version,
            model_fingerprint=self.identity.model_fingerprint,
            weights_identity=self.identity.weights_identity,
            adapter_identity=self.identity.adapter_identity,
            positioning=self.identity.positioning,
            memory_version=self.identity.memory_version,
            numerical_mode=self.identity.numerical_mode,
            descriptor_fingerprint=self.descriptor_fingerprint,
            token_ids=self.token_ids,
        )


def _snapshot_metadata_bytes(snapshot: StateSnapshot) -> int:
    layer_metadata: list[dict[str, int | str]] = []
    for layer in snapshot.layers:
        if isinstance(layer, FullAttentionSnapshot):
            layer_metadata.append({
                "kind": layer.kind, "layer_index": layer.layer_index,
                "logical_length": layer.logical_length,
                "kv_heads": layer.kv_heads, "key_dim": layer.key_dim,
                "value_dim": layer.value_dim, "dtype": layer.dtype,
            })
        elif isinstance(layer, RecurrentSnapshot):
            layer_metadata.append({
                "kind": layer.kind, "layer_index": layer.layer_index,
                "logical_position": layer.logical_position,
                "conv_channels": layer.conv_channels,
                "conv_kernel": layer.conv_kernel,
                "recurrent_key_heads": layer.recurrent_key_heads,
                "recurrent_value_heads": layer.recurrent_value_heads,
                "recurrent_key_dim": layer.recurrent_key_dim,
                "recurrent_value_dim": layer.recurrent_value_dim,
                "dtype": layer.dtype,
            })
        else:
            layer_metadata.append({"kind": layer.kind, "layer_index": layer.layer_index})
    metadata = {
        "identity": snapshot.identity.descriptor(),
        "schema_version": snapshot.schema_version,
        "descriptor_fingerprint": snapshot.descriptor_fingerprint,
        "descriptors": [item.descriptor() for item in snapshot.descriptors],
        "position": snapshot.position,
        "generation": snapshot.generation,
        "layers": layer_metadata,
    }
    return len(json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii"))


def _snapshot_byte_cost(snapshot: StateSnapshot) -> int:
    buffers = sum(layer.memory_bytes() for layer in snapshot.layers)
    return buffers + 8 * len(snapshot.token_ids) + _snapshot_metadata_bytes(snapshot)


class FullAttentionState:
    """Owned flattened K/V rows for one conventional attention layer."""

    __slots__ = ("spec", "key", "value", "logical_length")

    def __init__(
        self,
        spec: LayerMemorySpec,
        *,
        key: Iterable[float] = (),
        value: Iterable[float] = (),
        logical_length: int = 0,
    ) -> None:
        if spec.kind != "attention":
            raise StateError("FullAttentionState requires an attention descriptor")
        _strict_nonnegative("logical_length", logical_length)
        self.spec = spec
        self.key = _float_array(
            key, expected=logical_length * self.key_row_size, what="attention key",
        )
        self.value = _float_array(
            value, expected=logical_length * self.value_row_size,
            what="attention value",
        )
        self.logical_length = logical_length

    @property
    def key_row_size(self) -> int:
        return self.spec.kv_heads * self.spec.key_dim

    @property
    def value_row_size(self) -> int:
        return self.spec.kv_heads * self.spec.value_dim

    def append(self, key: Iterable[float], value: Iterable[float]) -> None:
        prepared_key = _float_array(
            key, expected=self.key_row_size, what="attention key row",
        )
        prepared_value = _float_array(
            value, expected=self.value_row_size, what="attention value row",
        )
        self.key.extend(prepared_key)
        self.value.extend(prepared_value)
        self.logical_length += 1

    def reset(self) -> None:
        self.key = array("f")
        self.value = array("f")
        self.logical_length = 0

    def truncate(self, position: int) -> None:
        _strict_nonnegative("position", position)
        if position > self.logical_length:
            raise StateError(
                f"cannot extend attention state from {self.logical_length} to {position}"
            )
        del self.key[position * self.key_row_size:]
        del self.value[position * self.value_row_size:]
        self.logical_length = position

    def clone(self) -> FullAttentionState:
        return FullAttentionState(
            self.spec, key=self.key, value=self.value,
            logical_length=self.logical_length,
        )

    def snapshot(self) -> FullAttentionSnapshot:
        return FullAttentionSnapshot(
            layer_index=self.spec.layer_index,
            logical_length=self.logical_length,
            kv_heads=self.spec.kv_heads,
            key_dim=self.spec.key_dim,
            value_dim=self.spec.value_dim,
            key=_f32_to_bytes(self.key),
            value=_f32_to_bytes(self.value),
        )

    def memory_bytes(self) -> int:
        return (len(self.key) + len(self.value)) * _F32_ITEMSIZE

    def describe(self) -> dict[str, int | str | list[int]]:
        return {
            "kind": "attention", "layer_index": self.spec.layer_index,
            "logical_length": self.logical_length, "dtype": "f32",
            "key_shape": [self.logical_length, self.spec.kv_heads, self.spec.key_dim],
            "value_shape": [
                self.logical_length, self.spec.kv_heads, self.spec.value_dim,
            ],
            "memory_bytes": self.memory_bytes(),
        }


class RecurrentState:
    """Owned convolution history and DeltaNet matrix for one layer."""

    __slots__ = ("spec", "conv_history", "delta_matrix", "logical_position")

    def __init__(
        self,
        spec: LayerMemorySpec,
        *,
        conv_history: Iterable[float] | None = None,
        delta_matrix: Iterable[float] | None = None,
        logical_position: int = 0,
    ) -> None:
        if spec.kind != "recurrent":
            raise StateError("RecurrentState requires a recurrent descriptor")
        _strict_nonnegative("logical_position", logical_position)
        self.spec = spec
        self.conv_history = (
            _zero_floats(self.conv_size) if conv_history is None else
            _float_array(conv_history, expected=self.conv_size, what="conv history")
        )
        self.delta_matrix = (
            _zero_floats(self.delta_size) if delta_matrix is None else
            _float_array(delta_matrix, expected=self.delta_size, what="delta matrix")
        )
        self.logical_position = logical_position

    @property
    def conv_size(self) -> int:
        return (self.spec.conv_kernel - 1) * self.spec.conv_channels

    @property
    def delta_size(self) -> int:
        return (
            self.spec.recurrent_value_heads
            * self.spec.recurrent_value_dim
            * self.spec.recurrent_key_dim
        )

    def advance(
        self,
        *,
        conv_history: Iterable[float] | None = None,
        delta_matrix: Iterable[float] | None = None,
    ) -> None:
        prepared_conv = self.conv_history if conv_history is None else _float_array(
            conv_history, expected=self.conv_size, what="conv history",
        )
        prepared_delta = self.delta_matrix if delta_matrix is None else _float_array(
            delta_matrix, expected=self.delta_size, what="delta matrix",
        )
        if conv_history is not None:
            self.conv_history = prepared_conv
        if delta_matrix is not None:
            self.delta_matrix = prepared_delta
        self.logical_position += 1

    def reset(self) -> None:
        self.conv_history = _zero_floats(self.conv_size)
        self.delta_matrix = _zero_floats(self.delta_size)
        self.logical_position = 0

    def truncate(self, position: int) -> None:
        _strict_nonnegative("position", position)
        if position != self.logical_position:
            raise ReplayRequiredError(
                "recurrent state cannot be truncated by changing its logical "
                "position; restore a complete checkpoint and replay"
            )

    def clone(self) -> RecurrentState:
        return RecurrentState(
            self.spec, conv_history=self.conv_history,
            delta_matrix=self.delta_matrix,
            logical_position=self.logical_position,
        )

    def snapshot(self) -> RecurrentSnapshot:
        return RecurrentSnapshot(
            layer_index=self.spec.layer_index,
            logical_position=self.logical_position,
            conv_channels=self.spec.conv_channels,
            conv_kernel=self.spec.conv_kernel,
            recurrent_key_heads=self.spec.recurrent_key_heads,
            recurrent_value_heads=self.spec.recurrent_value_heads,
            recurrent_key_dim=self.spec.recurrent_key_dim,
            recurrent_value_dim=self.spec.recurrent_value_dim,
            conv_history=_f32_to_bytes(self.conv_history),
            delta_matrix=_f32_to_bytes(self.delta_matrix),
        )

    def memory_bytes(self) -> int:
        return (len(self.conv_history) + len(self.delta_matrix)) * _F32_ITEMSIZE

    def describe(self) -> dict[str, int | str | list[int]]:
        return {
            "kind": "recurrent", "layer_index": self.spec.layer_index,
            "logical_position": self.logical_position, "dtype": "f32",
            "conv_history_shape": [
                self.spec.conv_kernel - 1, self.spec.conv_channels,
            ],
            "delta_matrix_shape": [
                self.spec.recurrent_value_heads,
                self.spec.recurrent_value_dim,
                self.spec.recurrent_key_dim,
            ],
            "memory_bytes": self.memory_bytes(),
        }


class EmptyLayerState:
    __slots__ = ("spec",)

    def __init__(self, spec: LayerMemorySpec) -> None:
        if spec.kind != "none":
            raise StateError("EmptyLayerState requires a none descriptor")
        self.spec = spec

    def reset(self) -> None:
        return None

    def clone(self) -> EmptyLayerState:
        return EmptyLayerState(self.spec)

    def snapshot(self) -> EmptyLayerSnapshot:
        return EmptyLayerSnapshot(layer_index=self.spec.layer_index)

    def memory_bytes(self) -> int:
        return 0

    def describe(self) -> dict[str, int | str]:
        return {"kind": "none", "layer_index": self.spec.layer_index,
                "memory_bytes": 0}


LayerState = FullAttentionState | RecurrentState | EmptyLayerState


def _validate_descriptors(specs: Sequence[LayerMemorySpec]) -> tuple[LayerMemorySpec, ...]:
    result = tuple(specs)
    indices = [spec.layer_index for spec in result]
    if indices != list(range(len(result))):
        raise StateError(
            "layer descriptors must be ordered with contiguous zero-based indices; "
            f"got {indices!r}"
        )
    return result


def _new_layer(spec: LayerMemorySpec) -> LayerState:
    if spec.kind == "attention":
        return FullAttentionState(spec)
    if spec.kind == "recurrent":
        return RecurrentState(spec)
    return EmptyLayerState(spec)


def _validate_layer_snapshot(
    layer: LayerSnapshot,
    spec: LayerMemorySpec,
    position: int,
    *,
    materialize: bool = True,
) -> LayerState | None:
    if getattr(layer, "layer_index", None) != spec.layer_index:
        raise SnapshotMismatchError(
            f"snapshot layer index mismatch at layer {spec.layer_index}"
        )
    if getattr(layer, "kind", None) != spec.kind:
        raise SnapshotMismatchError(
            f"snapshot layer {spec.layer_index} kind differs from descriptor"
        )
    if spec.kind == "attention":
        if not isinstance(layer, FullAttentionSnapshot):
            raise SnapshotMismatchError(
                f"snapshot layer {spec.layer_index} has the wrong record type"
            )
        if layer.dtype != "f32-le":
            raise SnapshotMismatchError(
                f"snapshot layer {spec.layer_index} has unsupported dtype {layer.dtype!r}"
            )
        if (
            layer.logical_length != position
            or layer.kv_heads != spec.kv_heads
            or layer.key_dim != spec.key_dim
            or layer.value_dim != spec.value_dim
        ):
            raise SnapshotMismatchError(
                f"snapshot layer {spec.layer_index} attention shape/position differs"
            )
        key_count = position * spec.kv_heads * spec.key_dim
        value_count = position * spec.kv_heads * spec.value_dim
        _validate_f32_bytes(
            layer.key,
            expected=key_count,
            what=f"snapshot layer {spec.layer_index} key",
        )
        _validate_f32_bytes(
            layer.value,
            expected=value_count,
            what=f"snapshot layer {spec.layer_index} value",
        )
        if not materialize:
            return None
        key = _f32_from_bytes(
            layer.key, expected=key_count,
            what=f"snapshot layer {spec.layer_index} key",
        )
        value = _f32_from_bytes(
            layer.value, expected=value_count,
            what=f"snapshot layer {spec.layer_index} value",
        )
        return FullAttentionState(
            spec, key=key, value=value, logical_length=position,
        )
    if spec.kind == "recurrent":
        if not isinstance(layer, RecurrentSnapshot):
            raise SnapshotMismatchError(
                f"snapshot layer {spec.layer_index} has the wrong record type"
            )
        if layer.dtype != "f32-le":
            raise SnapshotMismatchError(
                f"snapshot layer {spec.layer_index} has unsupported dtype {layer.dtype!r}"
            )
        expected_dimensions = (
            spec.conv_channels, spec.conv_kernel, spec.recurrent_key_heads,
            spec.recurrent_value_heads, spec.recurrent_key_dim,
            spec.recurrent_value_dim,
        )
        actual_dimensions = (
            layer.conv_channels, layer.conv_kernel, layer.recurrent_key_heads,
            layer.recurrent_value_heads, layer.recurrent_key_dim,
            layer.recurrent_value_dim,
        )
        if layer.logical_position != position or actual_dimensions != expected_dimensions:
            raise SnapshotMismatchError(
                f"snapshot layer {spec.layer_index} recurrent shape/position differs"
            )
        conv_count = (spec.conv_kernel - 1) * spec.conv_channels
        delta_count = (
            spec.recurrent_value_heads * spec.recurrent_value_dim
            * spec.recurrent_key_dim
        )
        _validate_f32_bytes(
            layer.conv_history, expected=conv_count,
            what=f"snapshot layer {spec.layer_index} conv history",
        )
        _validate_f32_bytes(
            layer.delta_matrix, expected=delta_count,
            what=f"snapshot layer {spec.layer_index} delta matrix",
        )
        if not materialize:
            return None
        conv = _f32_from_bytes(
            layer.conv_history, expected=conv_count,
            what=f"snapshot layer {spec.layer_index} conv history",
        )
        delta = _f32_from_bytes(
            layer.delta_matrix, expected=delta_count,
            what=f"snapshot layer {spec.layer_index} delta matrix",
        )
        return RecurrentState(
            spec, conv_history=conv, delta_matrix=delta,
            logical_position=position,
        )
    if not isinstance(layer, EmptyLayerSnapshot):
        raise SnapshotMismatchError(
            f"snapshot layer {spec.layer_index} has the wrong record type"
        )
    return None if not materialize else EmptyLayerState(spec)


def _validate_snapshot_record(snapshot: StateSnapshot) -> None:
    if not isinstance(snapshot, StateSnapshot):
        raise SnapshotMismatchError("restore requires a StateSnapshot")
    if not isinstance(snapshot.identity, StateIdentity):
        raise SnapshotMismatchError("snapshot identity record has the wrong type")
    if not isinstance(snapshot.descriptors, tuple) or not all(
        isinstance(spec, LayerMemorySpec) for spec in snapshot.descriptors
    ):
        raise SnapshotMismatchError("snapshot descriptors must be an immutable tuple")
    if not isinstance(snapshot.token_ids, tuple):
        raise SnapshotMismatchError("snapshot token IDs must be an immutable tuple")
    if not isinstance(snapshot.layers, tuple):
        raise SnapshotMismatchError("snapshot layers must be an immutable tuple")
    _strict_positive("snapshot schema_version", snapshot.schema_version)
    _strict_nonnegative("snapshot position", snapshot.position)
    _strict_nonnegative("snapshot generation", snapshot.generation)
    if len(snapshot.token_ids) != snapshot.position:
        raise SnapshotMismatchError(
            "snapshot token count does not match its logical position"
        )
    for token in snapshot.token_ids:
        _strict_nonnegative("snapshot token id", token)
    descriptors = _validate_descriptors(snapshot.descriptors)
    expected_fingerprint = descriptor_fingerprint(
        snapshot.identity, descriptors, snapshot.schema_version,
    )
    if snapshot.descriptor_fingerprint != expected_fingerprint:
        raise SnapshotMismatchError("snapshot descriptor fingerprint is inconsistent")
    if len(snapshot.layers) != len(descriptors):
        raise SnapshotMismatchError("snapshot layer count differs from its descriptors")
    for layer, spec in zip(snapshot.layers, descriptors):
        _validate_layer_snapshot(layer, spec, snapshot.position, materialize=False)
    _strict_nonnegative("snapshot byte_cost", snapshot.byte_cost)
    if snapshot.byte_cost != _snapshot_byte_cost(snapshot):
        raise SnapshotMismatchError("snapshot byte accounting is inconsistent")


class ModelState:
    """One-owner, heterogeneous live state for a single token sequence."""

    __slots__ = (
        "identity", "schema_version", "descriptors", "descriptor_fingerprint",
        "position", "token_ids", "generation", "layer_states",
        "_invalid_components",
    )

    def __init__(
        self,
        identity: StateIdentity,
        descriptors: Sequence[LayerMemorySpec],
        *,
        schema_version: int = STATE_SCHEMA_VERSION,
    ) -> None:
        if not isinstance(identity, StateIdentity):
            raise StateError("identity must be a StateIdentity")
        _strict_positive("schema_version", schema_version)
        self.identity = identity
        self.schema_version = schema_version
        self.descriptors = _validate_descriptors(descriptors)
        self.descriptor_fingerprint = descriptor_fingerprint(
            identity, self.descriptors, schema_version,
        )
        self.position = 0
        self.token_ids: list[int] = []
        self.generation = 0
        self.layer_states: list[LayerState] = [
            _new_layer(spec) for spec in self.descriptors
        ]
        self._invalid_components: set[str] = set()

    def layer(self, layer_index: int) -> LayerState:
        _strict_nonnegative("layer_index", layer_index)
        try:
            return self.layer_states[layer_index]
        except IndexError as exc:
            raise StateError(f"layer index {layer_index} is out of range") from exc

    def require_valid(self) -> None:
        if self._invalid_components:
            names = ", ".join(sorted(self._invalid_components))
            raise StateError(f"model state contains invalid components: {names}")

    def invalidate(
        self, component: str | int = "all", *, generation: int | None = None,
    ) -> bool:
        if generation is not None:
            _strict_nonnegative("generation", generation)
            if generation != self.generation:
                return False
        if isinstance(component, int) and not isinstance(component, bool):
            if component < 0 or component >= len(self.layer_states):
                raise StateError(f"layer index {component} is out of range")
            name = f"layer:{component}"
        elif component in ("all", "attention", "recurrent"):
            name = str(component)
        else:
            raise StateError(
                "component must be 'all', 'attention', 'recurrent', or a layer index"
            )
        self._invalid_components.add(name)
        return True

    def record_token(self, token_id: int) -> None:
        """Publish a token after every stateful layer advanced successfully."""

        _strict_nonnegative("token_id", token_id)
        self.require_valid()
        target = self.position + 1
        for layer in self.layer_states:
            if isinstance(layer, FullAttentionState):
                actual = layer.logical_length
            elif isinstance(layer, RecurrentState):
                actual = layer.logical_position
            else:
                continue
            if actual != target:
                raise StateError(
                    f"layer {layer.spec.layer_index} is at {actual}, expected {target} "
                    "before recording a token"
                )
        self.token_ids.append(token_id)
        self.position = target
        self.generation += 1

    def reset(self) -> None:
        """Reset live memory only; independent prefix stores are unchanged."""

        fresh = [_new_layer(spec) for spec in self.descriptors]
        self.layer_states = fresh
        self.position = 0
        self.token_ids = []
        self.generation += 1
        self._invalid_components.clear()

    def snapshot(self) -> StateSnapshot:
        self.require_valid()
        self._validate_live()
        layers = tuple(layer.snapshot() for layer in self.layer_states)
        incomplete = StateSnapshot(
            identity=self.identity,
            schema_version=self.schema_version,
            descriptor_fingerprint=self.descriptor_fingerprint,
            descriptors=self.descriptors,
            position=self.position,
            token_ids=tuple(self.token_ids),
            generation=self.generation,
            layers=layers,
            byte_cost=0,
        )
        snapshot = StateSnapshot(
            identity=incomplete.identity,
            schema_version=incomplete.schema_version,
            descriptor_fingerprint=incomplete.descriptor_fingerprint,
            descriptors=incomplete.descriptors,
            position=incomplete.position,
            token_ids=incomplete.token_ids,
            generation=incomplete.generation,
            layers=incomplete.layers,
            byte_cost=_snapshot_byte_cost(incomplete),
        )
        # byte_cost changes the object but is deliberately excluded from the
        # serialized metadata it accounts for.
        return snapshot

    def checkpoint(self, position: int | None = None) -> StateSnapshot:
        if position is not None and position != self.position:
            raise ReplayRequiredError(
                "a complete checkpoint can only be captured at the live position"
            )
        return self.snapshot()

    def restore(self, snapshot: StateSnapshot) -> None:
        """Validate and build every buffer before atomically publishing it."""

        _validate_snapshot_record(snapshot)
        if snapshot.identity != self.identity:
            raise SnapshotMismatchError("snapshot model/weights/adapter identity differs")
        if snapshot.schema_version != self.schema_version:
            raise SnapshotMismatchError("snapshot schema version differs")
        if snapshot.descriptors != self.descriptors:
            raise SnapshotMismatchError("snapshot layer descriptors differ")
        if snapshot.descriptor_fingerprint != self.descriptor_fingerprint:
            raise SnapshotMismatchError("snapshot descriptor fingerprint differs")

        prepared = [
            _validate_layer_snapshot(layer, spec, snapshot.position)
            for layer, spec in zip(snapshot.layers, self.descriptors)
        ]
        prepared_tokens = list(snapshot.token_ids)
        new_generation = max(self.generation, snapshot.generation) + 1

        self.layer_states = prepared
        self.token_ids = prepared_tokens
        self.position = snapshot.position
        self.generation = new_generation
        self._invalid_components.clear()

    def clone(self) -> ModelState:
        clone = ModelState(
            self.identity, self.descriptors, schema_version=self.schema_version,
        )
        clone.layer_states = [layer.clone() for layer in self.layer_states]
        clone.token_ids = list(self.token_ids)
        clone.position = self.position
        clone.generation = self.generation
        clone._invalid_components = set(self._invalid_components)
        return clone

    def truncate(
        self,
        position: int,
        *,
        checkpoints: Iterable[StateSnapshot] = (),
        replay: Callable[[ModelState, int], None] | None = None,
    ) -> None:
        """Truncate attention directly; rebuild recurrent state via replay.

        The replay callback must advance all stateful layers and call
        ``record_token`` once for each supplied token.  Work happens on a
        private clone and is published only after full validation.
        """

        _strict_nonnegative("position", position)
        self.require_valid()
        self._validate_live()
        if position > self.position:
            raise StateError(f"cannot extend state from {self.position} to {position}")
        if position == self.position:
            return
        target_tokens = tuple(self.token_ids[:position])
        has_recurrent = any(spec.kind == "recurrent" for spec in self.descriptors)
        if not has_recurrent:
            prepared = self.clone()
            for layer in prepared.layer_states:
                if isinstance(layer, FullAttentionState):
                    layer.truncate(position)
            prepared.token_ids = list(target_tokens)
            prepared.position = position
            prepared.generation = self.generation + 1
            prepared._validate_live()
            self._publish(prepared)
            return

        if position == 0:
            prepared = ModelState(
                self.identity, self.descriptors, schema_version=self.schema_version,
            )
            prepared.generation = self.generation + 1
            self._publish(prepared)
            return
        if replay is None:
            raise ReplayRequiredError(
                "truncating recurrent state requires a replay callback"
            )

        candidates: list[StateSnapshot] = []
        for checkpoint in checkpoints:
            _validate_snapshot_record(checkpoint)
            if (
                checkpoint.identity != self.identity
                or checkpoint.schema_version != self.schema_version
                or checkpoint.descriptors != self.descriptors
            ):
                raise SnapshotMismatchError(
                    "truncate checkpoint does not belong to this model/schema"
                )
            if checkpoint.position <= position:
                if checkpoint.token_ids != target_tokens[:checkpoint.position]:
                    raise SnapshotMismatchError(
                        "truncate checkpoint tokens are not a target-prefix"
                    )
                candidates.append(checkpoint)

        prepared = ModelState(
            self.identity, self.descriptors, schema_version=self.schema_version,
        )
        start = 0
        if candidates:
            chosen = max(candidates, key=lambda item: item.position)
            prepared.restore(chosen)
            start = chosen.position
        for token in target_tokens[start:]:
            before = prepared.position
            replay(prepared, token)
            if prepared.position != before + 1 or prepared.token_ids[-1:] != [token]:
                raise StateError(
                    "replay callback must advance and record exactly the supplied token"
                )
        if tuple(prepared.token_ids) != target_tokens:
            raise StateError("replay callback changed the checkpoint/token prefix")
        prepared._validate_live()
        prepared.generation = self.generation + 1
        self._publish(prepared)

    def replay(
        self, token_ids: Iterable[int],
        step: Callable[[ModelState, int], None],
    ) -> None:
        """Atomically replay new suffix tokens on a private clone."""

        tokens = tuple(token_ids)
        for token in tokens:
            _strict_nonnegative("token_id", token)
        prepared = self.clone()
        for token in tokens:
            before = prepared.position
            step(prepared, token)
            if prepared.position != before + 1 or prepared.token_ids[-1:] != [token]:
                raise StateError(
                    "replay callback must advance and record exactly the supplied token"
                )
        expected_tokens = tuple(self.token_ids) + tokens
        if tuple(prepared.token_ids) != expected_tokens:
            raise StateError("replay callback changed the existing token prefix")
        prepared._validate_live()
        prepared.generation = self.generation + len(tokens)
        self._publish(prepared)

    def memory_bytes(self) -> int:
        return sum(layer.memory_bytes() for layer in self.layer_states) + 8 * len(
            self.token_ids
        )

    def describe(self) -> dict[str, object]:
        layer_descriptions = [layer.describe() for layer in self.layer_states]
        state_buffers = sum(layer.memory_bytes() for layer in self.layer_states)
        token_bytes = 8 * len(self.token_ids)
        return {
            "architecture": self.identity.architecture,
            "model_fingerprint": self.identity.model_fingerprint,
            "weights_identity": self.identity.weights_identity,
            "adapter_identity": self.identity.adapter_identity,
            "positioning": self.identity.positioning,
            "numerical_mode": self.identity.numerical_mode,
            "schema_version": self.schema_version,
            "memory_version": self.identity.memory_version,
            "descriptor_fingerprint": self.descriptor_fingerprint,
            "position": self.position,
            "generation": self.generation,
            "valid": not self._invalid_components,
            "invalid_components": sorted(self._invalid_components),
            "state_buffer_bytes": state_buffers,
            "token_bytes": token_bytes,
            "memory_bytes": state_buffers + token_bytes,
            "layers": layer_descriptions,
        }

    def prefix_key(self, token_ids: Sequence[int] | None = None) -> PrefixKey:
        tokens = tuple(self.token_ids if token_ids is None else token_ids)
        for token in tokens:
            _strict_nonnegative("token_id", token)
        return PrefixKey(
            architecture=self.identity.architecture,
            schema_version=self.schema_version,
            model_fingerprint=self.identity.model_fingerprint,
            weights_identity=self.identity.weights_identity,
            adapter_identity=self.identity.adapter_identity,
            positioning=self.identity.positioning,
            memory_version=self.identity.memory_version,
            numerical_mode=self.identity.numerical_mode,
            descriptor_fingerprint=self.descriptor_fingerprint,
            token_ids=tokens,
        )

    def _publish(self, prepared: ModelState) -> None:
        self.layer_states = prepared.layer_states
        self.token_ids = prepared.token_ids
        self.position = prepared.position
        self.generation = prepared.generation
        self._invalid_components = prepared._invalid_components

    def _validate_live(self) -> None:
        if not isinstance(self.identity, StateIdentity):
            raise StateError("live identity record has the wrong type")
        if self.descriptors != _validate_descriptors(self.descriptors):
            raise StateError("live descriptors are not canonical")
        expected_fingerprint = descriptor_fingerprint(
            self.identity, self.descriptors, self.schema_version,
        )
        if self.descriptor_fingerprint != expected_fingerprint:
            raise StateError("live descriptor fingerprint is inconsistent")
        if len(self.token_ids) != self.position:
            raise StateError("live token count does not match logical position")
        if len(self.layer_states) != len(self.descriptors):
            raise StateError("live layer count does not match descriptors")
        for state, spec in zip(self.layer_states, self.descriptors):
            if state.spec != spec:
                raise StateError(
                    f"live layer {spec.layer_index} descriptor ownership differs"
                )
            if isinstance(state, FullAttentionState):
                if state.logical_length != self.position:
                    raise StateError(
                        f"attention layer {spec.layer_index} position differs"
                    )
                if len(state.key) != self.position * state.key_row_size:
                    raise StateError(f"attention layer {spec.layer_index} key shape differs")
                if len(state.value) != self.position * state.value_row_size:
                    raise StateError(
                        f"attention layer {spec.layer_index} value shape differs"
                    )
            elif isinstance(state, RecurrentState):
                if state.logical_position != self.position:
                    raise StateError(
                        f"recurrent layer {spec.layer_index} position differs"
                    )
                if len(state.conv_history) != state.conv_size:
                    raise StateError(
                        f"recurrent layer {spec.layer_index} conv shape differs"
                    )
                if len(state.delta_matrix) != state.delta_size:
                    raise StateError(
                        f"recurrent layer {spec.layer_index} delta shape differs"
                    )


class PrefixSnapshotStore:
    """Immutable prefix snapshots in byte-budgeted least-recently-used order."""

    __slots__ = ("max_bytes", "max_slots", "_slots", "_costs", "_bytes",
                 "hits", "misses", "evictions")

    def __init__(self, max_bytes: int, *, max_slots: int | None = None) -> None:
        _strict_nonnegative("max_bytes", max_bytes)
        if max_slots is not None:
            _strict_nonnegative("max_slots", max_slots)
        self.max_bytes = max_bytes
        self.max_slots = max_slots
        self._slots: OrderedDict[PrefixKey, StateSnapshot] = OrderedDict()
        self._costs: dict[PrefixKey, int] = {}
        self._bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    @property
    def current_bytes(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._slots)

    def keys(self) -> tuple[PrefixKey, ...]:
        """Return keys from least to most recently used."""

        return tuple(self._slots)

    @staticmethod
    def slot_bytes(key: PrefixKey, snapshot: StateSnapshot) -> int:
        return key.memory_bytes() + snapshot.memory_bytes()

    def put(self, snapshot: StateSnapshot) -> bool:
        _validate_snapshot_record(snapshot)
        key = snapshot.prefix_key()
        cost = self.slot_bytes(key, snapshot)
        if cost > self.max_bytes or self.max_slots == 0:
            return False

        old = self._costs.get(key, 0)
        projected = self._bytes - old + cost
        # Updating one key is atomic with respect to an oversized replacement.
        if projected > self.max_bytes and len(self._slots) == 1 and key in self._slots:
            return False
        if key in self._slots:
            del self._slots[key]
            self._bytes -= old
            del self._costs[key]
        self._slots[key] = snapshot
        self._costs[key] = cost
        self._bytes += cost
        self._evict_to_budget()
        return key in self._slots

    def save(self, state: ModelState) -> bool:
        return self.put(state.snapshot())

    def get(self, key: PrefixKey) -> StateSnapshot | None:
        snapshot = self._slots.get(key)
        if snapshot is None:
            self.misses += 1
            return None
        self._slots.move_to_end(key)
        self.hits += 1
        return snapshot

    def restore(self, state: ModelState, token_ids: Sequence[int]) -> bool:
        snapshot = self.get(state.prefix_key(token_ids))
        if snapshot is None:
            return False
        state.restore(snapshot)
        return True

    def remove(self, key: PrefixKey) -> bool:
        if key not in self._slots:
            return False
        del self._slots[key]
        self._bytes -= self._costs.pop(key)
        return True

    def clear(self) -> None:
        self._slots.clear()
        self._costs.clear()
        self._bytes = 0

    def invalidate_model(self, model_fingerprint: str) -> int:
        keys = [
            key for key in self._slots
            if key.model_fingerprint == model_fingerprint
        ]
        for key in keys:
            self.remove(key)
        return len(keys)

    def describe(self) -> dict[str, int | str | None]:
        return {
            "accounting": "serialized-payload-and-key-bytes",
            "max_bytes": self.max_bytes,
            "max_slots": self.max_slots,
            "current_bytes": self._bytes,
            "slots": len(self._slots),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
        }

    def _evict_to_budget(self) -> None:
        while self._slots and (
            self._bytes > self.max_bytes
            or (self.max_slots is not None and len(self._slots) > self.max_slots)
        ):
            key, _ = self._slots.popitem(last=False)
            self._bytes -= self._costs.pop(key)
            self.evictions += 1


class ConventionalKVStateAdapter:
    """Compatibility owner for the conventional transformer's K/V arrays.

    The existing decoder writes its NumPy arrays (or pure-Python row lists)
    directly.  Replacing those hot-loop data structures would add overhead and
    risk old-model regressions, so this adapter defines the generic state
    boundary around them: snapshots are deeply immutable, restore validates
    every layer before publishing, and ordinary truncation remains O(layers)
    for NumPy and O(discarded rows) for the pure tier.
    """

    __slots__ = (
        "model", "identity", "descriptors", "generation", "_last_position",
    )

    def __init__(self, model: Any, identity: StateIdentity) -> None:
        self.model = model
        self.identity = identity
        hp = model.hp
        self.descriptors = tuple(
            LayerMemorySpec(
                layer_index=index,
                kind="attention",
                kv_heads=hp.n_kv,
                key_dim=hp.head_dim,
                value_dim=hp.head_dim,
            )
            for index in range(hp.n_layer)
        )
        self.generation = 0
        self._last_position = int(model.n_past)

    @staticmethod
    def _flatten_rows(rows: Any, count: int) -> array:
        result = array("f")
        if hasattr(rows, "dtype") and hasattr(rows, "tobytes"):
            result.frombytes(rows[:count].tobytes(order="C"))
            if sys.byteorder != "little":
                result.byteswap()
            return result
        for row in rows[:count]:
            result.extend(row)
        return result

    def _observe_live(self) -> None:
        position = int(self.model.n_past)
        if position != self._last_position:
            self.generation += 1
            self._last_position = position

    def model_state(self) -> ModelState:
        """Return an owned generic-state copy of the live conventional cache."""

        self._observe_live()
        model = self.model
        position = int(model.n_past)
        if position != len(model.cached_ids):
            raise StateError(
                "conventional K/V position and cached token IDs are out of step"
            )
        staged = ModelState(self.identity, self.descriptors)
        staged.layer_states = [
            FullAttentionState(
                spec,
                key=self._flatten_rows(model.cache_k[index], position),
                value=self._flatten_rows(model.cache_v[index], position),
                logical_length=position,
            )
            for index, spec in enumerate(self.descriptors)
        ]
        staged.position = position
        staged.token_ids = list(model.cached_ids)
        staged.generation = self.generation
        staged._validate_live()
        return staged

    def snapshot(self) -> StateSnapshot:
        return self.model_state().snapshot()

    def restore(self, snapshot: StateSnapshot) -> None:
        """Validate and materialize every row before changing the live model."""

        staged = ModelState(self.identity, self.descriptors)
        staged.restore(snapshot)
        if staged.position > self.model.n_ctx:
            raise StateError(
                f"snapshot position {staged.position} exceeds context "
                f"{self.model.n_ctx}"
            )

        # Prepare detached, correctly-shaped payloads first.  The assignments
        # below cannot discover a bad snapshot halfway through publication.
        numpy_cache = bool(
            self.model.cache_k and hasattr(self.model.cache_k[0], "dtype")
        )
        prepared_k: list[Any] = []
        prepared_v: list[Any] = []
        if numpy_cache:
            import numpy as np

            shape = (staged.position, self.model.hp.n_kv, self.model.hp.head_dim)
            for layer in staged.layer_states:
                assert isinstance(layer, FullAttentionState)
                prepared_k.append(
                    np.frombuffer(layer.key, dtype=np.float32).reshape(shape).copy()
                )
                prepared_v.append(
                    np.frombuffer(layer.value, dtype=np.float32).reshape(shape).copy()
                )
        else:
            row_size = self.model.hp.n_kv * self.model.hp.head_dim
            for layer in staged.layer_states:
                assert isinstance(layer, FullAttentionState)
                keys = list(layer.key)
                values = list(layer.value)
                prepared_k.append([
                    keys[offset:offset + row_size]
                    for offset in range(0, len(keys), row_size)
                ])
                prepared_v.append([
                    values[offset:offset + row_size]
                    for offset in range(0, len(values), row_size)
                ])

        self.model._chain_invalidate(0)
        if numpy_cache:
            for index in range(len(self.descriptors)):
                self.model.cache_k[index][:staged.position] = prepared_k[index]
                self.model.cache_v[index][:staged.position] = prepared_v[index]
        else:
            self.model.cache_k = prepared_k
            self.model.cache_v = prepared_v
        self.model.n_past = staged.position
        self.model.cached_ids = list(staged.token_ids)
        self.model.last_prefill_forwarded = 0
        self.generation = max(self.generation, staged.generation) + 1
        self._last_position = staged.position

    def reset(self, *, clear_prefixes: bool = False) -> None:
        self.model._init_cache()
        self.generation += 1
        self._last_position = 0
        if clear_prefixes:
            self.clear_prefixes()

    def truncate(self, position: int) -> None:
        if isinstance(position, bool) or not isinstance(position, int):
            raise StateError(f"position must be an integer, got {position!r}")
        position = max(0, min(position, int(self.model.n_past)))
        if hasattr(self.model.cache_k[0], "dtype") if self.model.cache_k else False:
            self.model.n_past = position
        else:
            for index in range(len(self.descriptors)):
                del self.model.cache_k[index][position:]
                del self.model.cache_v[index][position:]
            self.model.n_past = position
        del self.model.cached_ids[position:]
        self.model._chain_invalidate(position)
        self.generation += 1
        self._last_position = position

    def clear_prefixes(self) -> None:
        slots = getattr(self.model, "_prefix_slots", None)
        if slots is not None:
            slots.clear()
        self.model._prefix_bytes = 0

    def describe(self) -> dict[str, Any]:
        self._observe_live()
        row_values = self.model.hp.n_kv * self.model.hp.head_dim
        return {
            "identity": self.identity.descriptor(),
            "schema_version": STATE_SCHEMA_VERSION,
            "descriptor_fingerprint": descriptor_fingerprint(
                self.identity, self.descriptors,
            ),
            "position": int(self.model.n_past),
            "generation": self.generation,
            "layers": len(self.descriptors),
            "attention_bytes": (
                int(self.model.n_past) * len(self.descriptors) * row_values * 2 * 4
            ),
            "storage": "numpy-f32-capacity" if (
                self.model.cache_k and hasattr(self.model.cache_k[0], "dtype")
            ) else "python-f32-rows",
        }


__all__ = [
    "STATE_SCHEMA_VERSION", "MEMORY_REPRESENTATION_VERSION",
    "StateError", "SnapshotMismatchError", "ReplayRequiredError",
    "LayerMemorySpec", "StateIdentity", "FullAttentionState",
    "RecurrentState", "ModelState", "StateSnapshot", "PrefixKey",
    "PrefixSnapshotStore", "ConventionalKVStateAdapter",
    "descriptor_fingerprint",
]
