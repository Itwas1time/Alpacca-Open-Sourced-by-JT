# Alpaccaroo - native-packed Qwen35 ROCm matrices.
# MIT License. See LICENSE.
"""Exact GGUF packed storage and Triton matvec dispatch for ROCm.

Each matrix retains one uint8 device tensor with the exact GGUF payload.  It
never constructs a matrix-sized code, scale, or float expansion.  PyTorch and
Triton are imported only when a real matrix is constructed, preserving the
dependency-free CPU tiers and allowing geometry/ownership tests offline.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from .gguf import GGML_BLOCK_INFO
from .qwen35_rocm import Qwen35RocmUnavailable, RocmCapability, rocm_capability


ROCM_PACKED_IMPLEMENTED_DTYPES = frozenset((
    "IQ3_S", "IQ4_NL", "IQ4_XS",
    "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0",
))
# Every exact-artifact quant format passed the opt-in Triton gate on gfx1151
# with deterministic synthetic blocks and a native matrix from the pinned 27B
# GGUF. This qualifies matrix primitives, not the model chain.
ROCM_PACKED_QUALIFIED_DTYPES = ROCM_PACKED_IMPLEMENTED_DTYPES
_DEVICE_CODEBOOK_BYTES = {"IQ3_S": 512 * 4, "IQ4_NL": 16, "IQ4_XS": 16}

IQ3_S_BLOCK_ELEMENTS, IQ3_S_BLOCK_BYTES = GGML_BLOCK_INFO["IQ3_S"]
IQ4_NL_BLOCK_ELEMENTS, IQ4_NL_BLOCK_BYTES = GGML_BLOCK_INFO["IQ4_NL"]
IQ4_XS_BLOCK_ELEMENTS, IQ4_XS_BLOCK_BYTES = GGML_BLOCK_INFO["IQ4_XS"]
Q3_K_BLOCK_ELEMENTS, Q3_K_BLOCK_BYTES = GGML_BLOCK_INFO["Q3_K"]
Q4_K_BLOCK_ELEMENTS, Q4_K_BLOCK_BYTES = GGML_BLOCK_INFO["Q4_K"]
Q5_K_BLOCK_ELEMENTS, Q5_K_BLOCK_BYTES = GGML_BLOCK_INFO["Q5_K"]
Q6_K_BLOCK_ELEMENTS, Q6_K_BLOCK_BYTES = GGML_BLOCK_INFO["Q6_K"]
Q8_0_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES = GGML_BLOCK_INFO["Q8_0"]


@dataclass(frozen=True, slots=True)
class PackedGeometry:
    dtype: str
    rows: int
    columns: int
    block_elements: int
    block_bytes: int
    blocks_per_row: int
    packed_row_bytes: int
    packed_bytes: int

    def descriptor(self) -> dict[str, int | str]:
        return {
            "dtype": self.dtype,
            "rows": self.rows,
            "columns": self.columns,
            "block_elements": self.block_elements,
            "block_bytes": self.block_bytes,
            "blocks_per_row": self.blocks_per_row,
            "packed_row_bytes": self.packed_row_bytes,
            "packed_bytes": self.packed_bytes,
        }


# Compatibility name retained for the first public Q4_K slice.
Q4KGeometry = PackedGeometry


def packed_geometry(dtype: str, rows: int, columns: int) -> PackedGeometry:
    """Validate one supported GGUF matrix without importing an accelerator."""

    if dtype not in ROCM_PACKED_IMPLEMENTED_DTYPES:
        supported = ", ".join(sorted(ROCM_PACKED_IMPLEMENTED_DTYPES))
        raise ValueError(
            f"packed ROCm matvec does not implement {dtype}; supported: "
            f"{supported}"
        )
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise ValueError("packed ROCm matrix rows must be a positive integer")
    if isinstance(columns, bool) or not isinstance(columns, int) or columns <= 0:
        raise ValueError("packed ROCm matrix columns must be a positive integer")
    block_elements, block_bytes = GGML_BLOCK_INFO[dtype]
    if columns % block_elements:
        raise ValueError(
            f"{dtype} matrix columns {columns} are not divisible by block "
            f"size {block_elements}"
        )
    blocks = columns // block_elements
    row_bytes = blocks * block_bytes
    return PackedGeometry(
        dtype=dtype,
        rows=rows,
        columns=columns,
        block_elements=block_elements,
        block_bytes=block_bytes,
        blocks_per_row=blocks,
        packed_row_bytes=row_bytes,
        packed_bytes=rows * row_bytes,
    )


def q4_k_geometry(rows: int, columns: int) -> PackedGeometry:
    """Compatibility wrapper for callers of the original Q4_K-only slice."""

    return packed_geometry("Q4_K", rows, columns)


class _PackedRuntime(Protocol):
    capability: RocmCapability
    device_label: str

    def upload_packed(self, payload: bytes) -> Any: ...
    def tensor_bytes(self, tensor: Any) -> int: ...
    def storage_bytes(self, tensor: Any) -> int: ...
    def prepare_dtype(self, dtype: str) -> int: ...
    def input_from_host(self, values: Sequence[float]) -> Any: ...
    def new_output(self, rows: int) -> Any: ...
    def validate_vector(self, tensor: Any, length: int, what: str) -> None: ...
    def launch_packed(
        self,
        packed: Any,
        values: Any,
        output: Any,
        geometry: PackedGeometry,
    ) -> None: ...
    def download(self, tensor: Any) -> list[float]: ...
    def close(self) -> None: ...


class _TorchTritonRuntime:
    """Small PyTorch/Triton adapter kept behind strict ROCm capability."""

    def __init__(self) -> None:
        capability = rocm_capability()
        if not capability.available:
            raise Qwen35RocmUnavailable(capability.reason)
        if capability.device_index is None:
            raise Qwen35RocmUnavailable("ROCm capability has no target device index")
        self.capability = capability
        self.torch = importlib.import_module("torch")
        kernels = importlib.import_module("alpaccaroo.qwen35_rocm_packed_triton")
        self._launch = kernels.launch_packed_matvec
        self.device_index = capability.device_index
        self.device_label = f"cuda:{self.device_index}"
        self._codebook = None

    def upload_packed(self, payload: bytes):
        # The bytearray is one temporary exact packed host copy. ``to`` is
        # synchronous here; neither it nor its tensor view survives this call.
        owner = bytearray(payload)
        host = self.torch.frombuffer(owner, dtype=self.torch.uint8)
        device = host.to(device=self.device_label, non_blocking=False)
        if device.dtype != self.torch.uint8 or int(device.numel()) != len(payload):
            raise RuntimeError("ROCm upload changed packed payload geometry")
        return device

    @staticmethod
    def tensor_bytes(tensor: Any) -> int:
        return int(tensor.numel()) * int(tensor.element_size())

    @staticmethod
    def storage_bytes(tensor: Any) -> int:
        storage = getattr(tensor, "untyped_storage", None)
        if storage is None:
            return int(tensor.numel()) * int(tensor.element_size())
        return int(storage().nbytes())

    def prepare_dtype(self, dtype: str) -> int:
        """Upload only the small immutable IQ codebook required by ``dtype``."""

        self._codebook = None
        if dtype not in _DEVICE_CODEBOOK_BYTES:
            return 0
        quants = importlib.import_module("alpaccaroo.quants")
        if dtype == "IQ3_S":
            values = quants._IQ3S_GRID
            tensor_dtype = self.torch.int32
            expected_bytes = _DEVICE_CODEBOOK_BYTES[dtype]
        else:
            values = quants._IQ4_NL_VALUES
            tensor_dtype = self.torch.int8
            expected_bytes = _DEVICE_CODEBOOK_BYTES[dtype]
        codebook = self.torch.tensor(
            values, dtype=tensor_dtype, device=self.device_label,
        )
        tensor_bytes = self.tensor_bytes(codebook)
        storage_bytes = self.storage_bytes(codebook)
        if tensor_bytes != expected_bytes or storage_bytes != expected_bytes:
            raise RuntimeError(
                f"{dtype} ROCm codebook storage differs: tensor={tensor_bytes}, "
                f"storage={storage_bytes}, expected={expected_bytes}"
            )
        self._codebook = codebook
        return expected_bytes

    def input_from_host(self, values: Sequence[float]):
        return self.torch.tensor(
            tuple(float(value) for value in values),
            dtype=self.torch.float32,
            device=self.device_label,
        )

    def new_output(self, rows: int):
        return self.torch.zeros(
            (rows,), dtype=self.torch.float32, device=self.device_label,
        )

    def validate_vector(self, tensor: Any, length: int, what: str) -> None:
        if not self.torch.is_tensor(tensor):
            raise TypeError(f"{what} must be a PyTorch tensor")
        if tensor.device.type != "cuda":
            raise TypeError(f"{what} must be on the ROCm device")
        if tensor.device.index not in (None, self.device_index):
            raise ValueError(
                f"{what} is on device {tensor.device.index}, expected "
                f"{self.device_index}"
            )
        if tensor.dtype != self.torch.float32:
            raise TypeError(f"{what} must use float32")
        if tuple(tensor.shape) != (length,):
            raise ValueError(
                f"{what} has shape {tuple(tensor.shape)}, expected ({length},)"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{what} must be contiguous")

    def launch_packed(
        self,
        packed: Any,
        values: Any,
        output: Any,
        geometry: PackedGeometry,
    ) -> None:
        # Independent column tiles accumulate into one scalar per row.
        output.zero_()
        self._launch(
            packed,
            values,
            output,
            packed if self._codebook is None else self._codebook,
            dtype=geometry.dtype,
            rows=geometry.rows,
            columns=geometry.columns,
            packed_row_bytes=geometry.packed_row_bytes,
        )

    def download(self, tensor: Any) -> list[float]:
        self.torch.cuda.synchronize(self.device_index)
        return [float(value) for value in tensor.detach().cpu().tolist()]

    def close(self) -> None:
        self._codebook = None


class PackedRocmMatrix:
    """One immutable native-packed ROCm matrix."""

    __slots__ = (
        "name", "dtype", "rows", "columns", "packed_bytes",
        "device_packed_bytes", "device_storage_bytes",
        "device_codebook_bytes", "geometry",
        "capability", "_runtime", "_device_weights", "_closed",
    )

    def __init__(
        self,
        packed: bytes | bytearray | memoryview,
        dtype: str,
        *,
        rows: int,
        columns: int,
        name: str = "packed-rocm-matrix",
        _runtime: _PackedRuntime | None = None,
    ) -> None:
        geometry = packed_geometry(dtype, rows, columns)
        try:
            payload = bytes(packed)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "packed ROCm weights must be a contiguous byte buffer"
            ) from exc
        if len(payload) != geometry.packed_bytes:
            raise ValueError(
                f"{name} has {len(payload)} packed bytes, expected "
                f"{geometry.packed_bytes} for {rows}x{columns} {dtype}"
            )

        runtime = _TorchTritonRuntime() if _runtime is None else _runtime
        device_weights = runtime.upload_packed(payload)
        device_bytes = runtime.tensor_bytes(device_weights)
        if device_bytes != len(payload):
            raise RuntimeError(
                f"{name} device payload uses {device_bytes} bytes, expected the "
                f"exact {len(payload)} packed bytes"
            )
        storage_bytes = runtime.storage_bytes(device_weights)
        if storage_bytes != len(payload):
            raise RuntimeError(
                f"{name} device storage exposes {storage_bytes} bytes, expected "
                f"the exact {len(payload)} packed bytes"
            )
        codebook_bytes = runtime.prepare_dtype(dtype)
        expected_codebook_bytes = _DEVICE_CODEBOOK_BYTES.get(dtype, 0)
        if codebook_bytes != expected_codebook_bytes:
            raise RuntimeError(
                f"{name} runtime codebook uses {codebook_bytes!r} bytes, "
                f"expected {expected_codebook_bytes} for {dtype}"
            )

        self.name = name
        self.dtype = dtype
        self.rows = rows
        self.columns = columns
        self.packed_bytes = len(payload)
        self.device_packed_bytes = device_bytes
        self.device_storage_bytes = storage_bytes
        self.device_codebook_bytes = codebook_bytes
        self.geometry = geometry
        self.capability = runtime.capability
        self._runtime = runtime
        self._device_weights = device_weights
        self._closed = False

    @classmethod
    def from_source(
        cls,
        source: Any,
        tensor_name: str,
        *,
        _runtime: _PackedRuntime | None = None,
    ) -> "PackedRocmMatrix":
        """Upload one GGUF matrix without decoding or widening its blocks."""

        source._ensure_open()
        try:
            spec = source.specs[tensor_name]
        except KeyError:
            raise KeyError(f"GGUF has no packed tensor named {tensor_name!r}") from None
        if len(spec.shape) != 2:
            raise ValueError(
                f"tensor {tensor_name} has shape {spec.shape}, expected matrix"
            )
        geometry = packed_geometry(spec.dtype, spec.rows, spec.columns)
        if (
            spec.block_elements != geometry.block_elements
            or spec.block_bytes != geometry.block_bytes
        ):
            raise ValueError(f"tensor {tensor_name} GGUF block geometry differs")
        if spec.packed_bytes != geometry.packed_bytes:
            raise ValueError(
                f"tensor {tensor_name} declares {spec.packed_bytes} bytes, "
                f"expected {geometry.packed_bytes} from {spec.dtype} geometry"
            )
        return cls(
            source.packed_bytes(tensor_name),
            spec.dtype,
            rows=spec.rows,
            columns=spec.columns,
            name=tensor_name,
            _runtime=_runtime,
        )

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed or self._device_weights is None:
            raise RuntimeError(f"packed ROCm matrix {self.name} is closed")

    def matvec_device(self, values_device: Any, output_device: Any | None = None):
        """Run matvec while keeping input and output on the HIP device."""

        self._ensure_open()
        runtime = self._runtime
        runtime.validate_vector(
            values_device, self.columns, f"{self.name} matvec input",
        )
        if output_device is None:
            output_device = runtime.new_output(self.rows)
        else:
            runtime.validate_vector(
                output_device, self.rows, f"{self.name} matvec output",
            )
        runtime.launch_packed(
            self._device_weights,
            values_device,
            output_device,
            self.geometry,
        )
        return output_device

    def matvec(self, values: Sequence[float]) -> list[float]:
        """Upload one F32 vector and download only the resulting row vector."""

        self._ensure_open()
        if len(values) != self.columns:
            raise ValueError(
                f"{self.name} expects {self.columns} matvec values, got "
                f"{len(values)}"
            )
        device_values = self._runtime.input_from_host(values)
        return self._runtime.download(self.matvec_device(device_values))

    def descriptor(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": [self.columns, self.rows],
            "packed_bytes": self.packed_bytes,
            "device_packed_bytes": self.device_packed_bytes,
            "device_storage_bytes": self.device_storage_bytes,
            "device_codebook_bytes": self.device_codebook_bytes,
            "native_packed": True,
            "expanded_device_weight_bytes": 0,
            "device": self._runtime.device_label,
            "gcn_arch_name": self.capability.gcn_arch_name,
            "real_device_qualified": (
                self.dtype in ROCM_PACKED_QUALIFIED_DTYPES
            ),
            "geometry": self.geometry.descriptor(),
        }

    def close(self) -> None:
        self._device_weights = None
        self._runtime.close()
        self._closed = True

    def __enter__(self) -> "PackedRocmMatrix":
        self._ensure_open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = [
    "PackedGeometry", "PackedRocmMatrix", "Q4KGeometry",
    "IQ3_S_BLOCK_BYTES", "IQ3_S_BLOCK_ELEMENTS",
    "IQ4_NL_BLOCK_BYTES", "IQ4_NL_BLOCK_ELEMENTS",
    "IQ4_XS_BLOCK_BYTES", "IQ4_XS_BLOCK_ELEMENTS",
    "Q3_K_BLOCK_BYTES", "Q3_K_BLOCK_ELEMENTS",
    "Q4_K_BLOCK_BYTES", "Q4_K_BLOCK_ELEMENTS",
    "Q5_K_BLOCK_BYTES", "Q5_K_BLOCK_ELEMENTS",
    "Q6_K_BLOCK_BYTES", "Q6_K_BLOCK_ELEMENTS",
    "Q8_0_BLOCK_BYTES", "Q8_0_BLOCK_ELEMENTS",
    "ROCM_PACKED_IMPLEMENTED_DTYPES", "ROCM_PACKED_QUALIFIED_DTYPES",
    "packed_geometry", "q4_k_geometry",
]
