# Alpaccaroo - native GGUF packed tensor sources and CUDA placement plans.
# MIT License. See LICENSE.
"""Packed storage boundary shared by Qwen35 planning and future GPU kernels.

Unlike :mod:`alpaccaroo.qmatrix`, this layer never expands a complete matrix
to universal integer codes.  It retains the exact GGUF bytes, block geometry,
shape, dtype and offsets.  The host row-dot method is intentionally a small
reference; a CUDA backend may upload ``packed_row_bytes`` directly.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .gguf import GGML_BLOCK_INFO, GGUFFile
from .quants import _PURE_DECODERS


PACKED_LAYOUT_DTYPES = frozenset(_PURE_DECODERS)
DEFAULT_DEVICE_ALIGNMENT = 256
DEFAULT_AMD_APU_OS_RESERVE_BYTES = 8 * 1024**3


def _align(value: int, alignment: int) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("device alignment must be a positive power of two")
    return (value + alignment - 1) // alignment * alignment


def _nonnegative_bytes(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer byte count")
    return value


@dataclass(frozen=True, slots=True)
class AmdApuMemorySnapshot:
    """One live view of the overlapping host and accelerator memory pool.

    ``mem_available_bytes`` is Linux's live ``MemAvailable`` value. Cached and
    reclaimable bytes are retained for diagnostics but are not added to that
    value: the kernel already includes the reclaimable portion in its estimate.
    Accelerator allocations reduce the aperture headroom only; subtracting
    them from ``MemAvailable`` again would double-charge shared physical pages.
    """

    physical_total_bytes: int
    mem_available_bytes: int
    cached_bytes: int
    reclaimable_slab_bytes: int
    swap_total_bytes: int
    swap_free_bytes: int
    accelerator_aperture_bytes: int
    accelerator_allocated_bytes: int
    source: str = "injected"

    def __post_init__(self) -> None:
        for name in (
            "physical_total_bytes", "mem_available_bytes", "cached_bytes",
            "reclaimable_slab_bytes", "swap_total_bytes", "swap_free_bytes",
            "accelerator_aperture_bytes", "accelerator_allocated_bytes",
        ):
            _nonnegative_bytes(name, getattr(self, name))
        if self.physical_total_bytes == 0:
            raise ValueError("physical_total_bytes must be positive")
        if self.accelerator_aperture_bytes == 0:
            raise ValueError("accelerator_aperture_bytes must be positive")
        if self.mem_available_bytes > self.physical_total_bytes:
            raise ValueError(
                "mem_available_bytes cannot exceed physical_total_bytes"
            )
        if self.swap_free_bytes > self.swap_total_bytes:
            raise ValueError("swap_free_bytes cannot exceed swap_total_bytes")
        if self.accelerator_allocated_bytes > self.accelerator_aperture_bytes:
            raise ValueError(
                "accelerator_allocated_bytes cannot exceed the GPU-visible aperture"
            )
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("memory snapshot source must be a non-empty string")

    @property
    def page_cache_bytes(self) -> int:
        return self.cached_bytes

    @property
    def reclaimable_cache_and_slab_bytes(self) -> int:
        return self.cached_bytes + self.reclaimable_slab_bytes

    @property
    def swap_used_bytes(self) -> int:
        return self.swap_total_bytes - self.swap_free_bytes

    @property
    def accelerator_available_bytes(self) -> int:
        return self.accelerator_aperture_bytes - self.accelerator_allocated_bytes

    def descriptor(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "physical_total_bytes": self.physical_total_bytes,
            "mem_available_bytes": self.mem_available_bytes,
            "cached_bytes": self.cached_bytes,
            "reclaimable_slab_bytes": self.reclaimable_slab_bytes,
            "page_cache_bytes": self.page_cache_bytes,
            "reclaimable_cache_and_slab_bytes": (
                self.reclaimable_cache_and_slab_bytes
            ),
            "swap_total_bytes": self.swap_total_bytes,
            "swap_free_bytes": self.swap_free_bytes,
            "swap_used_bytes": self.swap_used_bytes,
            "accelerator_aperture_bytes": self.accelerator_aperture_bytes,
            "accelerator_allocated_bytes": self.accelerator_allocated_bytes,
            "accelerator_available_bytes": self.accelerator_available_bytes,
        }


def amd_apu_memory_snapshot_from_bytes(
    memory: Mapping[str, int],
    *,
    accelerator_aperture_bytes: int,
    accelerator_allocated_bytes: int = 0,
    source: str = "injected",
) -> AmdApuMemorySnapshot:
    """Build a snapshot from injected Linux memory fields expressed in bytes."""

    required = ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")
    missing = [name for name in required if name not in memory]
    if missing:
        raise ValueError(f"memory snapshot lacks required fields: {missing}")
    return AmdApuMemorySnapshot(
        physical_total_bytes=_nonnegative_bytes("MemTotal", memory["MemTotal"]),
        mem_available_bytes=_nonnegative_bytes(
            "MemAvailable", memory["MemAvailable"],
        ),
        cached_bytes=_nonnegative_bytes("Cached", memory.get("Cached", 0)),
        reclaimable_slab_bytes=_nonnegative_bytes(
            "SReclaimable", memory.get("SReclaimable", 0),
        ),
        swap_total_bytes=_nonnegative_bytes("SwapTotal", memory["SwapTotal"]),
        swap_free_bytes=_nonnegative_bytes("SwapFree", memory["SwapFree"]),
        accelerator_aperture_bytes=_nonnegative_bytes(
            "accelerator_aperture_bytes", accelerator_aperture_bytes,
        ),
        accelerator_allocated_bytes=_nonnegative_bytes(
            "accelerator_allocated_bytes", accelerator_allocated_bytes,
        ),
        source=source,
    )


def read_linux_amd_apu_memory_snapshot(
    *,
    accelerator_aperture_bytes: int,
    accelerator_allocated_bytes: int = 0,
    meminfo_path: str | Path = "/proc/meminfo",
) -> AmdApuMemorySnapshot:
    """Read a live Linux snapshot while leaving accelerator probes injectable."""

    path = Path(meminfo_path)
    fields: dict[str, int] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="ascii").splitlines(), start=1,
    ):
        if ":" not in line:
            continue
        name, raw = line.split(":", 1)
        parts = raw.split()
        if not parts:
            continue
        try:
            value = int(parts[0])
        except ValueError as exc:
            raise ValueError(
                f"{path}:{line_number}: invalid {name} byte count {parts[0]!r}"
            ) from exc
        if value < 0 or (len(parts) > 1 and parts[1] != "kB"):
            raise ValueError(
                f"{path}:{line_number}: unsupported {name} memory value {raw.strip()!r}"
            )
        fields[name] = value * 1024 if len(parts) > 1 else value
    return amd_apu_memory_snapshot_from_bytes(
        fields,
        accelerator_aperture_bytes=accelerator_aperture_bytes,
        accelerator_allocated_bytes=accelerator_allocated_bytes,
        source=str(path),
    )


@dataclass(frozen=True, slots=True)
class PackedTensorSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]
    relative_offset: int
    absolute_offset: int
    packed_bytes: int
    gguf_alignment: int
    block_elements: int
    block_bytes: int

    @property
    def columns(self) -> int:
        if len(self.shape) != 2:
            raise ValueError(f"tensor {self.name} is not a matrix")
        return self.shape[0]

    @property
    def rows(self) -> int:
        if len(self.shape) != 2:
            raise ValueError(f"tensor {self.name} is not a matrix")
        return self.shape[1]

    @property
    def packed_row_bytes(self) -> int:
        if self.columns % self.block_elements:
            raise ValueError(
                f"tensor {self.name} row width {self.columns} is not divisible "
                f"by {self.dtype} block size {self.block_elements}"
            )
        return self.columns // self.block_elements * self.block_bytes

    def device_bytes(self, alignment: int = DEFAULT_DEVICE_ALIGNMENT) -> int:
        return _align(self.packed_bytes, alignment)

    def descriptor(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "relative_offset": self.relative_offset,
            "absolute_offset": self.absolute_offset,
            "packed_bytes": self.packed_bytes,
            "gguf_alignment": self.gguf_alignment,
            "block_elements": self.block_elements,
            "block_bytes": self.block_bytes,
        }


class PackedTensorSource:
    """Owner of a read-only GGUF mmap and exact packed tensor descriptors."""

    __slots__ = ("path", "_gguf", "_closed", "specs")

    def __init__(self, gguf: GGUFFile) -> None:
        self.path = gguf.path.resolve()
        self._gguf = gguf
        self._closed = False
        specs: dict[str, PackedTensorSpec] = {}
        for name, info in gguf.tensors.items():
            if info.dtype not in GGML_BLOCK_INFO:
                continue
            block_elements, block_bytes = GGML_BLOCK_INFO[info.dtype]
            specs[name] = PackedTensorSpec(
                name=name,
                dtype=info.dtype,
                shape=info.shape,
                relative_offset=info.offset,
                absolute_offset=gguf.data_start + info.offset,
                packed_bytes=info.n_bytes,
                gguf_alignment=gguf.alignment,
                block_elements=block_elements,
                block_bytes=block_bytes,
            )
        self.specs = specs

    @classmethod
    def open(cls, path: str | Path) -> "PackedTensorSource":
        return cls(GGUFFile.open(path, prefetch=False))

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("packed tensor source is closed")

    def packed_bytes(self, name: str, *, start: int = 0, count: int | None = None) -> bytes:
        """Return an owned packed slice; no mmap-export survives the call."""

        self._ensure_open()
        spec = self.specs[name]
        if count is None:
            count = spec.packed_bytes - start
        if start < 0 or count < 0 or start + count > spec.packed_bytes:
            raise ValueError(
                f"packed slice [{start}, {start + count}) exceeds {name} "
                f"payload {spec.packed_bytes}"
            )
        view = self._gguf.tensor_bytes(name)
        result = bytes(view[start:start + count])
        del view
        return result

    def matrix(self, name: str) -> "PackedHostMatrix":
        self._ensure_open()
        spec = self.specs[name]
        if len(spec.shape) != 2:
            raise ValueError(f"tensor {name} has shape {spec.shape}, expected matrix")
        if spec.dtype not in PACKED_LAYOUT_DTYPES:
            raise ValueError(
                f"tensor {name} uses {spec.dtype}, which has no packed row decoder"
            )
        return PackedHostMatrix(self, spec)

    def close(self) -> None:
        if self._closed:
            return
        self._gguf.close()
        self._closed = True

    def __enter__(self) -> "PackedTensorSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class PackedHostMatrix:
    source: PackedTensorSource
    spec: PackedTensorSpec

    def row(self, row_index: int) -> list[float]:
        if row_index < 0:
            row_index += self.spec.rows
        if row_index < 0 or row_index >= self.spec.rows:
            raise IndexError(f"row {row_index} is out of range for {self.spec.name}")
        row_bytes = self.spec.packed_row_bytes
        payload = self.source.packed_bytes(
            self.spec.name, start=row_index * row_bytes, count=row_bytes,
        )
        return _PURE_DECODERS[self.spec.dtype](payload, self.spec.columns)

    def row_dot(self, row_index: int, values: Sequence[float]) -> float:
        if len(values) != self.spec.columns:
            raise ValueError(
                f"{self.spec.name} expects {self.spec.columns} values, got {len(values)}"
            )
        return math.fsum(
            weight * float(value) for weight, value in zip(self.row(row_index), values)
        )

    def matvec(self, values: Sequence[float]) -> list[float]:
        return [self.row_dot(row, values) for row in range(self.spec.rows)]


@dataclass(frozen=True, slots=True)
class PlacementSubgroup:
    name: str
    kind: str
    tensor_names: tuple[str, ...]
    packed_bytes: int
    device_bytes: int
    logical_parameters: int

    def descriptor(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "tensor_names": list(self.tensor_names),
            "packed_bytes": self.packed_bytes,
            "device_bytes": self.device_bytes,
            "logical_parameters": self.logical_parameters,
        }


@dataclass(frozen=True, slots=True)
class PlacementGroup:
    name: str
    kind: str
    tensor_names: tuple[str, ...]
    dtypes: tuple[str, ...]
    packed_bytes: int
    device_bytes: int
    logical_parameters: int
    token_frequency: float
    estimated_boundary_bytes: int
    score: float
    selection_rank: int
    compute_subgroups: tuple[PlacementSubgroup, ...]
    placement: str
    reason: str

    def descriptor(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "tensor_names": list(self.tensor_names),
            "dtypes": list(self.dtypes),
            "packed_bytes": self.packed_bytes,
            "device_bytes": self.device_bytes,
            "logical_parameters": self.logical_parameters,
            "token_frequency": self.token_frequency,
            "estimated_boundary_bytes": self.estimated_boundary_bytes,
            "score": self.score,
            "selection_rank": self.selection_rank,
            "compute_subgroups": [
                subgroup.descriptor() for subgroup in self.compute_subgroups
            ],
            "placement": self.placement,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class Qwen35PlacementPlan:
    schema_version: int
    model_fingerprint: str
    device_alignment: int
    budget_bytes: int
    reserve_bytes: int
    state_copies: int
    logical_state_bytes: int
    allocated_state_bytes: int
    activation_workspace_bytes: int
    state_kv_workspace_bytes: int
    available_weight_bytes: int
    planned_device_weight_bytes: int
    planned_host_weight_bytes: int
    all_weights_device_resident: bool
    required_dtypes: tuple[str, ...]
    qualified_dtypes: tuple[str, ...]
    unsupported_dtypes: tuple[str, ...]
    groups: tuple[PlacementGroup, ...]

    @property
    def planned_total_device_bytes(self) -> int:
        return (
            self.reserve_bytes + self.state_kv_workspace_bytes
            + self.planned_device_weight_bytes
        )

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_fingerprint": self.model_fingerprint,
            "device_alignment": self.device_alignment,
            "budget_bytes": self.budget_bytes,
            "reserve_bytes": self.reserve_bytes,
            "state_copies": self.state_copies,
            "logical_state_bytes": self.logical_state_bytes,
            "allocated_state_bytes": self.allocated_state_bytes,
            "activation_workspace_bytes": self.activation_workspace_bytes,
            "state_kv_workspace_bytes": self.state_kv_workspace_bytes,
            "available_weight_bytes": self.available_weight_bytes,
            "planned_device_weight_bytes": self.planned_device_weight_bytes,
            "planned_host_weight_bytes": self.planned_host_weight_bytes,
            "planned_total_device_bytes": self.planned_total_device_bytes,
            "all_weights_device_resident": self.all_weights_device_resident,
            "required_dtypes": list(self.required_dtypes),
            "qualified_dtypes": list(self.qualified_dtypes),
            "unsupported_dtypes": list(self.unsupported_dtypes),
            "groups": [group.descriptor() for group in self.groups],
        }


@dataclass(frozen=True, slots=True)
class AmdApuAdmissionPlan:
    """Fail-closed admission result for an overlapping AMD APU memory pool."""

    schema_version: int
    model_fingerprint: str
    placement_schema_version: int
    snapshot: AmdApuMemorySnapshot
    os_reserve_bytes: int
    require_swap_clear: bool
    host_weight_mapping_bytes: int
    host_runtime_bytes: int
    device_weight_bytes: int
    device_state_workspace_bytes: int
    device_driver_reserve_bytes: int
    device_required_bytes: int
    duplicated_weight_payload_bytes: int
    combined_incremental_physical_bytes: int
    live_physical_budget_bytes: int
    accelerator_aperture_budget_bytes: int
    projected_mem_available_bytes: int
    projected_reserve_headroom_bytes: int
    projected_accelerator_available_bytes: int
    largest_device_groups: tuple[tuple[str, int], ...]
    admitted: bool
    rejection_reasons: tuple[str, ...]

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_fingerprint": self.model_fingerprint,
            "placement_schema_version": self.placement_schema_version,
            "memory_pool_relationship": "overlapping-unified-not-additive",
            "snapshot": self.snapshot.descriptor(),
            "os_reserve_bytes": self.os_reserve_bytes,
            "require_swap_clear": self.require_swap_clear,
            "host_weight_mapping_bytes": self.host_weight_mapping_bytes,
            "host_runtime_bytes": self.host_runtime_bytes,
            "device_weight_bytes": self.device_weight_bytes,
            "device_state_workspace_bytes": self.device_state_workspace_bytes,
            "device_driver_reserve_bytes": self.device_driver_reserve_bytes,
            "device_required_bytes": self.device_required_bytes,
            "duplicated_weight_payload_bytes": self.duplicated_weight_payload_bytes,
            "weight_residency_assumption": (
                "host mapping and accelerator allocation are distinct physical "
                "residents; no page-sharing credit"
            ),
            "combined_incremental_physical_bytes": (
                self.combined_incremental_physical_bytes
            ),
            "live_physical_budget_bytes": self.live_physical_budget_bytes,
            "accelerator_aperture_budget_bytes": (
                self.accelerator_aperture_budget_bytes
            ),
            "projected_mem_available_bytes": self.projected_mem_available_bytes,
            "projected_reserve_headroom_bytes": (
                self.projected_reserve_headroom_bytes
            ),
            "projected_accelerator_available_bytes": (
                self.projected_accelerator_available_bytes
            ),
            "swap_contributes_capacity_bytes": 0,
            "largest_device_groups": [
                {"name": name, "device_bytes": byte_count}
                for name, byte_count in self.largest_device_groups
            ],
            "admitted": self.admitted,
            "rejection_reasons": list(self.rejection_reasons),
        }

    def require_admitted(self) -> "AmdApuAdmissionPlan":
        if self.admitted:
            return self
        largest = ", ".join(
            f"{name}={byte_count}" for name, byte_count in self.largest_device_groups
        ) or "none"
        raise MemoryError(
            "AMD APU shared-memory preflight rejected before allocation: "
            + "; ".join(self.rejection_reasons)
            + f"; largest_device_groups={largest}; "
            f"state_workspace={self.device_state_workspace_bytes}; "
            "close memory-heavy processes or reduce context; use validated F16 "
            "K/V or a lower quantization only after its correctness gate, "
            "otherwise select the optimized CPU backend"
        )


def build_amd_apu_admission_plan(
    placement: Qwen35PlacementPlan,
    snapshot: AmdApuMemorySnapshot,
    *,
    host_weight_mapping_bytes: int | None = None,
    host_runtime_bytes: int = 0,
    os_reserve_bytes: int = DEFAULT_AMD_APU_OS_RESERVE_BYTES,
    require_swap_clear: bool = True,
) -> AmdApuAdmissionPlan:
    """Plan one target-device run against live overlapping shared memory.

    The GPU-visible aperture and physical RAM are independent constraints on
    one pool, never capacities to add. The source GGUF mapping and accelerator
    weight allocation are charged separately because an APU does not imply
    that those pages share physical backing. Swap is diagnostic only and never
    contributes capacity.
    """

    if not isinstance(placement, Qwen35PlacementPlan):
        raise TypeError("placement must be a Qwen35PlacementPlan")
    if not isinstance(snapshot, AmdApuMemorySnapshot):
        raise TypeError("snapshot must be an AmdApuMemorySnapshot")
    if not isinstance(require_swap_clear, bool):
        raise TypeError("require_swap_clear must be a bool")
    os_reserve_bytes = _nonnegative_bytes("os_reserve_bytes", os_reserve_bytes)
    host_runtime_bytes = _nonnegative_bytes("host_runtime_bytes", host_runtime_bytes)

    classified_host_weights = sum(group.packed_bytes for group in placement.groups)
    if host_weight_mapping_bytes is None:
        host_weight_mapping_bytes = classified_host_weights
    host_weight_mapping_bytes = _nonnegative_bytes(
        "host_weight_mapping_bytes", host_weight_mapping_bytes,
    )
    if host_weight_mapping_bytes < classified_host_weights:
        raise ValueError(
            "host_weight_mapping_bytes cannot be smaller than the classified "
            f"GGUF weight payload ({classified_host_weights})"
        )

    device_weight_bytes = placement.planned_device_weight_bytes
    device_state_workspace_bytes = placement.state_kv_workspace_bytes
    device_driver_reserve_bytes = placement.reserve_bytes
    device_required_bytes = placement.planned_total_device_bytes
    duplicated_weight_payload_bytes = sum(
        group.packed_bytes
        for group in placement.groups if group.placement == "cuda"
    )
    combined_incremental = (
        host_weight_mapping_bytes + host_runtime_bytes + device_required_bytes
    )
    physical_budget = max(
        0, snapshot.mem_available_bytes - os_reserve_bytes,
    )
    aperture_budget = snapshot.accelerator_available_bytes
    projected_available = snapshot.mem_available_bytes - combined_incremental
    projected_reserve = projected_available - os_reserve_bytes
    projected_aperture = aperture_budget - device_required_bytes

    reasons: list[str] = []
    if require_swap_clear and snapshot.swap_used_bytes:
        reasons.append(
            f"swap is in use ({snapshot.swap_used_bytes} bytes); swap cannot "
            "back active weights or state"
        )
    if combined_incremental > physical_budget:
        reasons.append(
            "overlapping physical pool lacks headroom: "
            f"required={combined_incremental}, live_budget_after_os_reserve="
            f"{physical_budget}, mem_available={snapshot.mem_available_bytes}, "
            f"os_reserve={os_reserve_bytes}"
        )
    if device_required_bytes > aperture_budget:
        reasons.append(
            "GPU-visible aperture lacks headroom: "
            f"required={device_required_bytes}, available={aperture_budget}, "
            f"aperture={snapshot.accelerator_aperture_bytes}, "
            f"already_allocated={snapshot.accelerator_allocated_bytes}"
        )

    largest = tuple(sorted(
        (
            (group.name, group.device_bytes)
            for group in placement.groups if group.placement == "cuda"
        ),
        key=lambda item: (-item[1], item[0]),
    )[:5])
    return AmdApuAdmissionPlan(
        schema_version=1,
        model_fingerprint=placement.model_fingerprint,
        placement_schema_version=placement.schema_version,
        snapshot=snapshot,
        os_reserve_bytes=os_reserve_bytes,
        require_swap_clear=require_swap_clear,
        host_weight_mapping_bytes=host_weight_mapping_bytes,
        host_runtime_bytes=host_runtime_bytes,
        device_weight_bytes=device_weight_bytes,
        device_state_workspace_bytes=device_state_workspace_bytes,
        device_driver_reserve_bytes=device_driver_reserve_bytes,
        device_required_bytes=device_required_bytes,
        duplicated_weight_payload_bytes=duplicated_weight_payload_bytes,
        combined_incremental_physical_bytes=combined_incremental,
        live_physical_budget_bytes=physical_budget,
        accelerator_aperture_budget_bytes=aperture_budget,
        projected_mem_available_bytes=projected_available,
        projected_reserve_headroom_bytes=projected_reserve,
        projected_accelerator_available_bytes=projected_aperture,
        largest_device_groups=largest,
        admitted=not reasons,
        rejection_reasons=tuple(reasons),
    )


def _manifest_fingerprint(manifest: dict[str, Any]) -> str:
    source_hash = manifest["source"].get("sha256")
    if source_hash:
        return f"sha256:{source_hash}"
    payload = {
        "source": {
            "path": manifest["source"].get("path"),
            "byte_size": manifest["source"]["byte_size"],
        },
        "model": manifest["model"],
        "tensors": [
            (item["name"], item["dtype"], item["shape"], item["packed_bytes"])
            for item in manifest["tensors"]
        ],
    }
    return "manifest:" + hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")).hexdigest()


def build_qwen35_placement_plan(
    manifest: dict[str, Any],
    *,
    budget_bytes: int,
    qualified_dtypes: Iterable[str],
    device_alignment: int = DEFAULT_DEVICE_ALIGNMENT,
    state_copies: int = 1,
) -> Qwen35PlacementPlan:
    """Plan scored, indivisible compute groups after fixed reservations.

    Layers remain atomic so a single attention/recurrent block never alternates
    CPU and CUDA projections. The global stack is split at real execution
    boundaries (embedding, final norm, and output projection) rather than
    hidden behind one all-or-nothing bucket. Scores are deterministic benefit
    density: logical work plus avoided activation-boundary traffic, multiplied
    by token frequency, divided by aligned device bytes.
    """

    if budget_bytes < 0:
        raise ValueError("device budget cannot be negative")
    if isinstance(state_copies, bool) or not isinstance(state_copies, int):
        raise TypeError("state_copies must be an integer")
    if state_copies <= 0:
        raise ValueError("state_copies must be positive")
    _align(0, device_alignment)
    qualified = frozenset(qualified_dtypes)
    memory = manifest["memory_plan"]
    reserve = int(memory["cuda_reserve_bytes"])
    logical_state = sum(int(memory[name]) for name in (
        "recurrent_matrix_bytes", "convolution_history_bytes",
        "attention_kv_bytes",
    ))
    allocated_state = state_copies * logical_state
    activation_workspace = int(memory["logits_and_activation_bytes"])
    state_workspace = allocated_state + activation_workspace
    available = max(0, budget_bytes - reserve - state_workspace)

    records = [
        item for item in manifest["tensors"]
        if item["role"] != "unsupported-auxiliary"
        and not str(item["name"]).startswith("nextn.")
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in records:
        layer = item.get("layer_index")
        tensor_name = str(item["name"])
        if layer is not None:
            group_name = f"layer.{layer}"
        elif tensor_name == "token_embd.weight":
            group_name = "embeddings"
        elif tensor_name == "output_norm.weight":
            group_name = "final_norm"
        elif tensor_name == "output.weight":
            group_name = "output_projection"
        else:
            group_name = "global_auxiliary"
        grouped.setdefault(group_name, []).append(item)

    layer_count = int(manifest["model"]["main_layer_count"])
    hidden = int(manifest["model"]["embedding_length"])
    vocabulary = int(manifest["model"]["vocabulary_size"])

    def execution_order(name: str) -> tuple[int, int]:
        if name == "embeddings":
            return (0, -1)
        if name.startswith("layer."):
            return (1, int(name.split(".")[1]))
        if name == "final_norm":
            return (2, layer_count)
        if name == "output_projection":
            return (3, layer_count)
        return (4, layer_count)

    def kind_and_boundary(name: str) -> tuple[str, int]:
        if name == "embeddings":
            return "embedding", hidden * 4
        if name.startswith("layer."):
            return "complete_layer", hidden * 4 * 2
        if name == "final_norm":
            return "final_norm", hidden * 4 * 2
        if name == "output_projection":
            return "output_projection", (hidden + vocabulary) * 4
        return "global_auxiliary", hidden * 4 * 2

    def subgroup(
        name: str,
        kind: str,
        tensors: Sequence[dict[str, Any]],
    ) -> PlacementSubgroup:
        return PlacementSubgroup(
            name=name,
            kind=kind,
            tensor_names=tuple(str(item["name"]) for item in tensors),
            packed_bytes=sum(int(item["packed_bytes"]) for item in tensors),
            device_bytes=sum(
                _align(int(item["packed_bytes"]), device_alignment)
                for item in tensors
            ),
            logical_parameters=sum(
                int(item["logical_parameters"]) for item in tensors
            ),
        )

    candidates: list[dict[str, Any]] = []
    for name in sorted(grouped, key=execution_order):
        tensors = sorted(grouped[name], key=lambda item: str(item["name"]))
        packed = sum(int(item["packed_bytes"]) for item in tensors)
        device = sum(
            _align(int(item["packed_bytes"]), device_alignment) for item in tensors
        )
        logical = sum(int(item["logical_parameters"]) for item in tensors)
        kind, boundary = kind_and_boundary(name)
        if name.startswith("layer."):
            ffn_roles = {
                "post_attention_norm.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
            }
            ffn = [item for item in tensors if item["role"] in ffn_roles]
            attention = [item for item in tensors if item["role"] not in ffn_roles]
            subgroups = (
                subgroup(
                    name + ".attention_or_recurrent",
                    "attention_or_recurrent",
                    attention,
                ),
                subgroup(name + ".ffn", "ffn", ffn),
            )
        else:
            subgroups = (subgroup(name, kind, tensors),)
        frequency = 1.0
        score = frequency * (logical + boundary) / max(1, device)
        candidates.append({
            "name": name,
            "kind": kind,
            "tensors": tensors,
            "packed": packed,
            "device": device,
            "logical": logical,
            "frequency": frequency,
            "boundary": boundary,
            "score": score,
            "subgroups": subgroups,
        })

    # Select by scored benefit density. Equal-score layers retain execution
    # order, yielding a contiguous prefix instead of token-by-token ping-pong.
    selection = sorted(
        candidates,
        key=lambda item: (
            -float(item["score"]),
            execution_order(str(item["name"])),
        ),
    )
    used = 0
    placements: dict[str, tuple[str, str, int]] = {}
    for rank, candidate in enumerate(selection):
        name = str(candidate["name"])
        tensors = candidate["tensors"]
        dtypes = tuple(sorted({str(item["dtype"]) for item in tensors}))
        device = int(candidate["device"])
        missing = sorted(set(dtypes) - qualified)
        if missing:
            placement = "cpu"
            reason = "unqualified packed dtype(s): " + ", ".join(missing)
        elif used + device > available:
            placement = "cpu"
            reason = "complete group exceeds remaining device budget"
        else:
            placement = "cuda"
            reason = "scored complete group fits with every dtype qualified"
            used += device
        placements[name] = placement, reason, rank

    groups: list[PlacementGroup] = []
    for candidate in sorted(candidates, key=lambda item: execution_order(str(item["name"]))):
        name = str(candidate["name"])
        tensors = candidate["tensors"]
        dtypes = tuple(sorted({str(item["dtype"]) for item in tensors}))
        placement, reason, rank = placements[name]
        groups.append(PlacementGroup(
            name=name,
            kind=str(candidate["kind"]),
            tensor_names=tuple(str(item["name"]) for item in tensors),
            dtypes=dtypes,
            packed_bytes=int(candidate["packed"]),
            device_bytes=int(candidate["device"]),
            logical_parameters=int(candidate["logical"]),
            token_frequency=float(candidate["frequency"]),
            estimated_boundary_bytes=int(candidate["boundary"]),
            score=float(candidate["score"]),
            selection_rank=rank,
            compute_subgroups=tuple(candidate["subgroups"]),
            placement=placement,
            reason=reason,
        ))

    required = tuple(sorted({str(item["dtype"]) for item in records}))
    unsupported = tuple(sorted(set(required) - qualified))
    host = sum(group.packed_bytes for group in groups if group.placement == "cpu")
    all_device = bool(groups) and all(group.placement == "cuda" for group in groups)
    return Qwen35PlacementPlan(
        schema_version=3,
        model_fingerprint=_manifest_fingerprint(manifest),
        device_alignment=device_alignment,
        budget_bytes=budget_bytes,
        reserve_bytes=reserve,
        state_copies=state_copies,
        logical_state_bytes=logical_state,
        allocated_state_bytes=allocated_state,
        activation_workspace_bytes=activation_workspace,
        state_kv_workspace_bytes=state_workspace,
        available_weight_bytes=available,
        planned_device_weight_bytes=used,
        planned_host_weight_bytes=host,
        all_weights_device_resident=all_device,
        required_dtypes=required,
        qualified_dtypes=tuple(sorted(qualified)),
        unsupported_dtypes=unsupported,
        groups=tuple(groups),
    )


def verify_actual_allocations(
    plan: Qwen35PlacementPlan,
    allocations: dict[str, int],
    *,
    documented_overhead_bytes: int = 0,
) -> dict[str, Any]:
    """Reconcile actual group allocations and reject unexplained overruns."""

    documented_overhead_bytes = int(documented_overhead_bytes)
    if documented_overhead_bytes < 0:
        raise ValueError("documented allocation overhead cannot be negative")
    if documented_overhead_bytes > plan.reserve_bytes:
        raise MemoryError(
            "documented CUDA allocation overhead exceeds the reserved bytes: "
            f"{documented_overhead_bytes} > {plan.reserve_bytes}"
        )
    expected = {
        group.name: group.device_bytes
        for group in plan.groups if group.placement == "cuda"
    }
    if set(allocations) != set(expected):
        missing = sorted(set(expected) - set(allocations))
        extra = sorted(set(allocations) - set(expected))
        raise ValueError(f"allocation groups differ; missing={missing}, extra={extra}")
    overruns = {
        name: int(allocations[name]) - expected[name]
        for name in expected if int(allocations[name]) > expected[name]
    }
    if overruns:
        largest = sorted(
            (
                (group.name, group.device_bytes)
                for group in plan.groups if group.placement == "cuda"
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:5]
        raise MemoryError(
            f"unexplained CUDA allocation overrun: {overruns}; "
            f"largest_planned_groups={largest}; "
            f"allocated_state_bytes={plan.allocated_state_bytes}; "
            f"activation_workspace_bytes={plan.activation_workspace_bytes}; "
            "reduce context/device budget pressure or select the CPU backend"
        )
    actual = sum(int(value) for value in allocations.values())
    total = plan.reserve_bytes + plan.state_kv_workspace_bytes + actual
    if total > plan.budget_bytes:
        raise MemoryError(
            f"actual CUDA plan uses {total} bytes, budget is {plan.budget_bytes}; "
            f"weights={actual}, allocated_state={plan.allocated_state_bytes}, "
            f"activation_workspace={plan.activation_workspace_bytes}, "
            f"reserve={plan.reserve_bytes}; reduce context or use CPU"
        )
    return {
        "planned_weight_bytes": plan.planned_device_weight_bytes,
        "actual_weight_bytes": actual,
        "actual_documented_overhead_bytes": documented_overhead_bytes,
        "reserve_headroom_after_documented_overhead_bytes": (
            plan.reserve_bytes - documented_overhead_bytes
        ),
        "actual_total_device_bytes": total,
        "budget_bytes": plan.budget_bytes,
        "headroom_bytes": plan.budget_bytes - total,
    }


__all__ = [
    "AmdApuAdmissionPlan", "AmdApuMemorySnapshot",
    "DEFAULT_AMD_APU_OS_RESERVE_BYTES", "DEFAULT_DEVICE_ALIGNMENT",
    "PACKED_LAYOUT_DTYPES", "PackedHostMatrix", "PackedTensorSource",
    "PackedTensorSpec", "PlacementGroup", "PlacementSubgroup",
    "Qwen35PlacementPlan", "build_qwen35_placement_plan",
    "amd_apu_memory_snapshot_from_bytes", "build_amd_apu_admission_plan",
    "read_linux_amd_apu_memory_snapshot", "verify_actual_allocations",
]
