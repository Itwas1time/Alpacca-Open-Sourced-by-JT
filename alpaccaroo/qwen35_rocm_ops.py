# Alpaccaroo - device-resident Qwen3.5/Qwen3.8 ROCm semantic primitives.
# MIT License. See LICENSE.
"""Decomposed Qwen35 tensor contracts for the dedicated ROCm chain.

PyTorch is imported lazily only when :class:`Qwen35RocmOps` is constructed.
Every public tensor argument must already be a contiguous float32 tensor on
the selected HIP device (PyTorch intentionally calls that device ``cuda``).
The API has no host-download helper and never calls ``cpu()``, ``item()``, or
``tolist()``.  These primitives intentionally remain decomposed until live
gfx1151 measurements justify a fused Triton implementation.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any, Sequence

from .qwen35_rocm import Qwen35RocmUnavailable, RocmCapability, rocm_capability


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _shape(tensor: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in tensor.shape)


def _tensor_descriptor(tensor: Any) -> dict[str, Any]:
    device = tensor.device
    index = getattr(device, "index", None)
    return {
        "shape": list(_shape(tensor)),
        "dtype": str(tensor.dtype),
        "device": f"{device.type}:{0 if index is None else index}",
        "contiguous": bool(tensor.is_contiguous()),
    }


@dataclass(frozen=True, slots=True)
class RocmRecurrentWorkspace:
    """Zero-initialized mutable recurrent state, entirely on the HIP device."""

    convolution_history: Any
    delta_state: Any

    def descriptor(self) -> dict[str, Any]:
        return {
            "convolution_history": _tensor_descriptor(self.convolution_history),
            "delta_state": _tensor_descriptor(self.delta_state),
            "delta_state_orientation": "batch,value_head,value,key",
            "initialization": "deterministic-zero",
        }


@dataclass(frozen=True, slots=True)
class RocmAttentionWorkspace:
    """Preallocated device K/V capacity; the chain owns the logical length."""

    key_cache: Any
    value_cache: Any

    def descriptor(self) -> dict[str, Any]:
        return {
            "key_cache": _tensor_descriptor(self.key_cache),
            "value_cache": _tensor_descriptor(self.value_cache),
            "layout": "capacity,batch,kv_head,head_dimension",
            "initialization": "deterministic-zero",
        }


class Qwen35RocmOps:
    """Device-only implementation of the frozen Qwen35 semantic boundaries."""

    def __init__(
        self,
        *,
        device_index: int | None = None,
        _torch: Any | None = None,
        _capability: RocmCapability | None = None,
    ) -> None:
        capability = rocm_capability() if _capability is None else _capability
        if not capability.available:
            raise Qwen35RocmUnavailable(capability.reason)
        selected = capability.device_index if device_index is None else device_index
        if selected is None:
            raise Qwen35RocmUnavailable("ROCm capability has no target device index")
        _nonnegative_integer("device_index", selected)
        if (
            capability.device_index is not None
            and selected != capability.device_index
        ):
            raise ValueError(
                f"device_index {selected} differs from qualified target "
                f"{capability.device_index}"
            )
        self.torch = importlib.import_module("torch") if _torch is None else _torch
        self.capability = capability
        self.device_index = selected
        self.device = f"cuda:{selected}"

    def descriptor(self) -> dict[str, Any]:
        return {
            "backend": "pytorch-rocm-decomposed",
            "device": self.device,
            "gcn_arch_name": self.capability.gcn_arch_name,
            "tensor_dtype": "float32",
            "host_transfer_api": False,
            "state_orientation": "batch,value_head,value,key",
            "recurrent_key_mapping": "value_head % key_head_count",
            "attention_key_mapping": "contiguous-query-head-groups",
            "attention_algorithm": "stable-online-softmax",
            "supported_attention_head_dimensions": "positive; live gate includes 256",
            "triton_fusion": "deferred-until-measured",
        }

    def _tensor(
        self,
        value: Any,
        name: str,
        *,
        dimensions: int | None = None,
        allow_integer: bool = False,
    ) -> Any:
        torch = self.torch
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a PyTorch tensor")
        device = value.device
        if getattr(device, "type", None) != "cuda":
            raise TypeError(f"{name} must remain on the ROCm device")
        index = getattr(device, "index", None)
        if index not in (None, self.device_index):
            raise ValueError(
                f"{name} is on device {index}, expected {self.device_index}"
            )
        valid_dtypes = (torch.float32,)
        if allow_integer:
            valid_dtypes = valid_dtypes + (torch.int64,)
        if value.dtype not in valid_dtypes:
            expected = "float32 or int64" if allow_integer else "float32"
            raise TypeError(f"{name} must use {expected}")
        if dimensions is not None and len(value.shape) != dimensions:
            raise ValueError(
                f"{name} has {_shape(value)}, expected {dimensions} dimensions"
            )
        if not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        return value

    @staticmethod
    def _same_shape(left: Any, right: Any, what: str) -> None:
        if _shape(left) != _shape(right):
            raise ValueError(
                f"{what} shapes differ: {_shape(left)} != {_shape(right)}"
            )

    def zeros(self, shape: Sequence[int]) -> Any:
        normalized = tuple(
            _positive_integer("workspace dimension", value) for value in shape
        )
        if not normalized:
            raise ValueError("workspace shape cannot be empty")
        return self.torch.zeros(
            normalized,
            dtype=self.torch.float32,
            device=self.device,
        )

    def initialize_recurrent_workspace(
        self,
        *,
        batch_size: int,
        convolution_kernel: int,
        convolution_channels: int,
        value_heads: int,
        value_dimension: int,
        key_dimension: int,
    ) -> RocmRecurrentWorkspace:
        batch = _positive_integer("batch_size", batch_size)
        kernel = _positive_integer("convolution_kernel", convolution_kernel)
        channels = _positive_integer("convolution_channels", convolution_channels)
        heads = _positive_integer("value_heads", value_heads)
        value_width = _positive_integer("value_dimension", value_dimension)
        key_width = _positive_integer("key_dimension", key_dimension)
        history = self.torch.zeros(
            (batch, kernel - 1, channels),
            dtype=self.torch.float32,
            device=self.device,
        )
        state = self.torch.zeros(
            (batch, heads, value_width, key_width),
            dtype=self.torch.float32,
            device=self.device,
        )
        return RocmRecurrentWorkspace(history, state)

    def initialize_attention_workspace(
        self,
        *,
        capacity: int,
        batch_size: int,
        kv_heads: int,
        key_dimension: int,
        value_dimension: int,
    ) -> RocmAttentionWorkspace:
        length = _positive_integer("capacity", capacity)
        batch = _positive_integer("batch_size", batch_size)
        heads = _positive_integer("kv_heads", kv_heads)
        key_width = _positive_integer("key_dimension", key_dimension)
        value_width = _positive_integer("value_dimension", value_dimension)
        keys = self.torch.zeros(
            (length, batch, heads, key_width),
            dtype=self.torch.float32,
            device=self.device,
        )
        values = self.torch.zeros(
            (length, batch, heads, value_width),
            dtype=self.torch.float32,
            device=self.device,
        )
        return RocmAttentionWorkspace(keys, values)

    def state_copy(self, source: Any) -> Any:
        source = self._tensor(source, "state source")
        return source.clone()

    def state_restore_(self, target: Any, snapshot: Any) -> Any:
        target = self._tensor(target, "state target")
        snapshot = self._tensor(snapshot, "state snapshot")
        self._same_shape(target, snapshot, "state restore")
        target.copy_(snapshot)
        return target

    def rms_norm(self, values: Any, weight: Any, epsilon: float) -> Any:
        values = self._tensor(values, "RMSNorm input")
        weight = self._tensor(weight, "RMSNorm weight", dimensions=1)
        epsilon = _positive_finite("epsilon", epsilon)
        if not values.shape or int(values.shape[-1]) != int(weight.shape[0]):
            raise ValueError("RMSNorm weight must match the input's final dimension")
        mean_square = self.torch.mean(values * values, dim=-1, keepdim=True)
        return values * self.torch.rsqrt(mean_square + epsilon) * weight

    def l2_normalize(self, values: Any, epsilon: float) -> Any:
        values = self._tensor(values, "L2 input")
        epsilon = _positive_finite("epsilon", epsilon)
        if not values.shape or int(values.shape[-1]) <= 0:
            raise ValueError("L2 input final dimension must be nonempty")
        magnitude = self.torch.sqrt(
            self.torch.sum(values * values, dim=-1, keepdim=True)
        )
        return values / self.torch.clamp_min(magnitude, epsilon)

    def residual(self, residual: Any, update: Any) -> Any:
        residual = self._tensor(residual, "residual input")
        update = self._tensor(update, "residual update")
        self._same_shape(residual, update, "residual")
        return residual + update

    def silu(self, values: Any) -> Any:
        values = self._tensor(values, "SiLU input")
        return values * self.torch.sigmoid(values)

    def swiglu(self, gate: Any, up: Any) -> Any:
        gate = self._tensor(gate, "SwiGLU gate")
        up = self._tensor(up, "SwiGLU up projection")
        self._same_shape(gate, up, "SwiGLU")
        return self.silu(gate) * up

    def sigmoid_gate(self, values: Any, gate: Any) -> Any:
        values = self._tensor(values, "sigmoid-gate values")
        gate = self._tensor(gate, "sigmoid-gate logits")
        self._same_shape(values, gate, "sigmoid gate")
        return values * self.torch.sigmoid(gate)

    def recurrent_gate(self, values: Any, weight: Any, gate: Any, epsilon: float) -> Any:
        normalized = self.rms_norm(values, weight, epsilon)
        gate = self._tensor(gate, "recurrent SiLU gate")
        self._same_shape(normalized, gate, "recurrent gate")
        return normalized * self.silu(gate)

    def split_query_gate(
        self,
        projected: Any,
        *,
        query_heads: int,
        head_dimension: int,
    ) -> tuple[Any, Any]:
        projected = self._tensor(projected, "joint Q/gate projection")
        heads = _positive_integer("query_heads", query_heads)
        width = _positive_integer("head_dimension", head_dimension)
        expected = heads * 2 * width
        actual = int(projected.shape[-1]) if projected.shape else 0
        if actual != expected:
            raise ValueError(
                f"joint Q/gate width is {actual}, expected "
                f"{expected} ({heads} heads * Q/gate stride {2 * width})"
            )
        reshaped = projected.reshape(_shape(projected)[:-1] + (heads, 2 * width))
        query = reshaped[..., :width].contiguous()
        gate = reshaped[..., width:].contiguous()
        return query, gate

    def causal_convolution_step_(
        self,
        current: Any,
        weights: Any,
        history: Any,
    ) -> Any:
        current = self._tensor(current, "convolution current", dimensions=2)
        weights = self._tensor(weights, "convolution weights", dimensions=2)
        history = self._tensor(history, "convolution history", dimensions=3)
        batch, channels = _shape(current)
        kernel, weight_channels = _shape(weights)
        if weight_channels != channels:
            raise ValueError("convolution weights must match current channels")
        if _shape(history) != (batch, kernel - 1, channels):
            raise ValueError(
                "convolution history must have shape "
                f"{(batch, kernel - 1, channels)}, got {_shape(history)}"
            )
        window = self.torch.cat((history, current.unsqueeze(1)), dim=1)
        output = self.torch.sum(window * weights.unsqueeze(0), dim=1)
        history.copy_(window[:, 1:, :])
        return output

    def recurrent_parameters(
        self,
        beta_logits: Any,
        alpha_logits: Any,
        dt_bias: Any,
        ssm_a: Any,
    ) -> tuple[Any, Any]:
        beta_logits = self._tensor(beta_logits, "beta logits", dimensions=2)
        alpha_logits = self._tensor(alpha_logits, "alpha logits", dimensions=2)
        dt_bias = self._tensor(dt_bias, "time-step bias", dimensions=1)
        ssm_a = self._tensor(ssm_a, "converted ssm_a", dimensions=1)
        self._same_shape(beta_logits, alpha_logits, "recurrent parameter logits")
        heads = int(beta_logits.shape[1])
        if _shape(dt_bias) != (heads,) or _shape(ssm_a) != (heads,):
            raise ValueError("time-step bias and ssm_a must match value heads")
        beta = self.torch.sigmoid(beta_logits)
        time_step = self.torch.nn.functional.softplus(alpha_logits + dt_bias)
        return beta, time_step * ssm_a

    def gated_delta_net_step_(
        self,
        state: Any,
        query: Any,
        key: Any,
        value: Any,
        beta: Any,
        log_decay: Any,
    ) -> Any:
        state = self._tensor(state, "GDN state", dimensions=4)
        query = self._tensor(query, "GDN query", dimensions=3)
        key = self._tensor(key, "GDN key", dimensions=3)
        value = self._tensor(value, "GDN value", dimensions=3)
        beta = self._tensor(beta, "GDN beta", dimensions=2)
        log_decay = self._tensor(log_decay, "GDN log decay", dimensions=2)
        batch, value_heads, value_width, key_width = _shape(state)
        if _shape(query) != _shape(key):
            raise ValueError("GDN query and key shapes must agree")
        if _shape(query)[0] != batch or _shape(query)[2] != key_width:
            raise ValueError("GDN query/key must match state batch and key dimension")
        key_heads = int(query.shape[1])
        if key_heads <= 0 or value_heads % key_heads:
            raise ValueError("GDN value heads must be a multiple of key heads")
        if _shape(value) != (batch, value_heads, value_width):
            raise ValueError("GDN value must match state batch/value-head/value axes")
        if _shape(beta) != (batch, value_heads):
            raise ValueError("GDN beta must match batch and value heads")
        if _shape(log_decay) != (batch, value_heads):
            raise ValueError("GDN log decay must match batch and value heads")

        mapping = self.torch.arange(
            value_heads,
            dtype=self.torch.int64,
            device=self.device,
        ) % key_heads
        mapped_query = query.index_select(1, mapping)
        mapped_key = key.index_select(1, mapping)
        state.mul_(self.torch.exp(log_decay).unsqueeze(-1).unsqueeze(-1))
        prediction = self.torch.sum(
            state * mapped_key.unsqueeze(-2), dim=-1,
        )
        delta = beta.unsqueeze(-1) * (value - prediction)
        state.add_(delta.unsqueeze(-1) * mapped_key.unsqueeze(-2))
        return self.torch.sum(
            state * mapped_query.unsqueeze(-2), dim=-1,
        ) * (1.0 / math.sqrt(key_width))

    def imrope(
        self,
        values: Any,
        position_lanes: Any,
        sections: Sequence[int],
        *,
        frequency_base: float,
        frequency_scale: float = 1.0,
    ) -> Any:
        values = self._tensor(values, "IMRoPE input")
        position_lanes = self._tensor(
            position_lanes,
            "IMRoPE position lanes",
            dimensions=1,
            allow_integer=True,
        )
        if position_lanes.dtype != self.torch.int64 or _shape(position_lanes) != (4,):
            raise ValueError("IMRoPE position lanes must be int64 shape (4,)")
        if len(sections) != 4:
            raise ValueError("IMRoPE requires four sections")
        section = tuple(
            _nonnegative_integer("IMRoPE section", item) for item in sections
        )
        rotary_pairs = sum(section)
        if rotary_pairs <= 0:
            raise ValueError("IMRoPE requires at least one rotary pair")
        rotary_dimensions = 2 * rotary_pairs
        if not values.shape or int(values.shape[-1]) < rotary_dimensions:
            raise ValueError(
                "IMRoPE input is shorter than twice the section sum"
            )
        base = _positive_finite("frequency_base", frequency_base)
        scale = _positive_finite("frequency_scale", frequency_scale)

        pair = self.torch.arange(
            rotary_pairs,
            dtype=self.torch.int64,
            device=self.device,
        )
        sector = pair % rotary_pairs
        lane = self.torch.full(
            (rotary_pairs,), 3, dtype=self.torch.int64, device=self.device,
        )
        lane = self.torch.where(
            (sector % 3 == 1) & (sector < 3 * section[1]), 1, lane,
        )
        lane = self.torch.where(
            (sector % 3 == 2) & (sector < 3 * section[2]), 2, lane,
        )
        lane = self.torch.where(
            (sector % 3 == 0) & (sector < 3 * section[0]), 0, lane,
        )
        positions = position_lanes.index_select(0, lane).to(dtype=values.dtype)
        exponent = pair.to(dtype=values.dtype) * (-2.0 / rotary_dimensions)
        theta = positions * scale * self.torch.pow(base, exponent)
        cosine = self.torch.cos(theta)
        sine = self.torch.sin(theta)
        first = values[..., :rotary_pairs]
        second = values[..., rotary_pairs:rotary_dimensions]
        output = values.clone()
        output[..., :rotary_pairs] = first * cosine - second * sine
        output[..., rotary_pairs:rotary_dimensions] = (
            first * sine + second * cosine
        )
        return output

    def attention_append_(
        self,
        key_cache: Any,
        value_cache: Any,
        position: int,
        key: Any,
        value: Any,
    ) -> int:
        key_cache = self._tensor(key_cache, "attention key cache", dimensions=4)
        value_cache = self._tensor(value_cache, "attention value cache", dimensions=4)
        key = self._tensor(key, "attention key", dimensions=3)
        value = self._tensor(value, "attention value", dimensions=3)
        position = _nonnegative_integer("attention cache position", position)
        if int(key_cache.shape[0]) != int(value_cache.shape[0]):
            raise ValueError("attention K/V cache capacities differ")
        if position >= int(key_cache.shape[0]):
            raise ValueError("attention cache position exceeds capacity")
        if _shape(key_cache)[1:3] != _shape(value_cache)[1:3]:
            raise ValueError("attention K/V cache batch and head axes differ")
        if _shape(key) != _shape(key_cache)[1:]:
            raise ValueError("attention key does not match cache token shape")
        if _shape(value) != _shape(value_cache)[1:]:
            raise ValueError("attention value does not match cache token shape")
        key_cache[position].copy_(key)
        value_cache[position].copy_(value)
        return position + 1

    def online_attention(
        self,
        query: Any,
        key_cache: Any,
        value_cache: Any,
        *,
        length: int,
        scale: float | None = None,
    ) -> Any:
        query = self._tensor(query, "attention query", dimensions=3)
        key_cache = self._tensor(key_cache, "attention key cache", dimensions=4)
        value_cache = self._tensor(value_cache, "attention value cache", dimensions=4)
        length = _positive_integer("attention cache length", length)
        capacity, batch, kv_heads, key_width = _shape(key_cache)
        value_shape = _shape(value_cache)
        if length > capacity:
            raise ValueError("attention cache length exceeds capacity")
        if value_shape[:3] != (capacity, batch, kv_heads):
            raise ValueError("attention K/V cache capacity, batch, or heads differ")
        query_batch, query_heads, query_width = _shape(query)
        if query_batch != batch or query_width != key_width:
            raise ValueError("attention query does not match K cache batch/head width")
        if kv_heads <= 0 or query_heads % kv_heads:
            raise ValueError("attention query heads must be a multiple of KV heads")
        scale_value = (
            1.0 / math.sqrt(key_width)
            if scale is None else _positive_finite("attention scale", scale)
        )
        mapping = self.torch.arange(
            query_heads,
            dtype=self.torch.int64,
            device=self.device,
        ) // (query_heads // kv_heads)
        maximum = self.torch.full(
            (batch, query_heads),
            -math.inf,
            dtype=self.torch.float32,
            device=self.device,
        )
        denominator = self.torch.zeros(
            (batch, query_heads),
            dtype=self.torch.float32,
            device=self.device,
        )
        accumulator = self.torch.zeros(
            (batch, query_heads, value_shape[3]),
            dtype=self.torch.float32,
            device=self.device,
        )
        for position in range(length):
            keys = key_cache[position].index_select(1, mapping)
            values = value_cache[position].index_select(1, mapping)
            score = self.torch.sum(query * keys, dim=-1) * scale_value
            next_maximum = self.torch.maximum(maximum, score)
            old_scale = self.torch.exp(maximum - next_maximum)
            new_scale = self.torch.exp(score - next_maximum)
            accumulator = (
                accumulator * old_scale.unsqueeze(-1)
                + values * new_scale.unsqueeze(-1)
            )
            denominator = denominator * old_scale + new_scale
            maximum = next_maximum
        return accumulator / denominator.unsqueeze(-1)

    def append_online_attention_(
        self,
        query: Any,
        key: Any,
        value: Any,
        key_cache: Any,
        value_cache: Any,
        *,
        position: int,
        scale: float | None = None,
    ) -> tuple[Any, int]:
        length = self.attention_append_(
            key_cache, value_cache, position, key, value,
        )
        return (
            self.online_attention(
                query,
                key_cache,
                value_cache,
                length=length,
                scale=scale,
            ),
            length,
        )


__all__ = [
    "Qwen35RocmOps",
    "RocmAttentionWorkspace",
    "RocmRecurrentWorkspace",
]
