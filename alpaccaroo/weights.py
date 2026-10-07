# Alpaccaroo - shared, allocation-conscious GGUF weight helpers.
# MIT License. See LICENSE.
"""Strict GGUF identity, tensor-table, and read-only weight helpers.

The manifest functions prove that a file's header, offsets, shapes and encoded
byte counts are coherent before allocating model memory.  ``WeightStore`` is
the small shared execution layer: it owns the GGUF read-only mmap and exposes
immutable F32/F16 views without expanding a tensor into Python float objects.
"""

from __future__ import annotations

import hashlib
import os
import struct
from array import array
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .gguf import (
    GGUFFile,
    T_ARRAY,
    T_BOOL,
    T_FLOAT32,
    T_FLOAT64,
    T_INT8,
    T_INT16,
    T_INT32,
    T_INT64,
    T_STRING,
    T_UINT8,
    T_UINT16,
    T_UINT32,
    T_UINT64,
)


class ManifestError(ValueError):
    """The GGUF header contradicts the selected architecture contract."""


class UnsupportedExecutionDType(ManifestError):
    """A tensor encoding has no execution view in this weight loader."""


GGUF_VALUE_TYPE_NAMES = {
    T_UINT8: "UINT8",
    T_INT8: "INT8",
    T_UINT16: "UINT16",
    T_INT16: "INT16",
    T_UINT32: "UINT32",
    T_INT32: "INT32",
    T_FLOAT32: "FLOAT32",
    T_BOOL: "BOOL",
    T_STRING: "STRING",
    T_ARRAY: "ARRAY",
    T_UINT64: "UINT64",
    T_INT64: "INT64",
    T_FLOAT64: "FLOAT64",
}

INTEGER_TYPES = frozenset({
    T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32, T_UINT64, T_INT64,
})
FLOAT_TYPES = frozenset({T_FLOAT32, T_FLOAT64})

# This first shared execution slice intentionally supports only dense types.
# Quantized matrices need packed-block implementations rather than accidental
# whole-matrix expansion through this API.
DENSE_EXECUTION_DTYPES = frozenset({"F32", "F16"})
_DENSE_VALUE_STRUCTS = {
    "F32": struct.Struct("<f"),
    "F16": struct.Struct("<e"),
}


def stream_sha256(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash *path* without constructing a second in-memory copy."""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while True:
            chunk = source.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def metadata_type_name(encoded: tuple[int, int | None]) -> str:
    value_type, element_type = encoded
    name = GGUF_VALUE_TYPE_NAMES.get(value_type, f"UNKNOWN_{value_type}")
    if value_type == T_ARRAY:
        item = GGUF_VALUE_TYPE_NAMES.get(
            int(element_type) if element_type is not None else -1,
            f"UNKNOWN_{element_type}",
        )
        return f"ARRAY[{item}]"
    return name


def metadata_inventory(gguf: GGUFFile) -> dict[str, dict[str, Any]]:
    """Return metadata in stable key order, including its encoded GGUF type."""
    return {
        key: {
            "type": metadata_type_name(gguf.metadata_types[key]),
            "value": gguf.metadata[key],
        }
        for key in sorted(gguf.metadata)
    }


def validate_tensor_layout(
    gguf: GGUFFile,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Validate all table byte ranges and return inventory plus dtype census."""
    if gguf.duplicate_metadata_keys:
        names = ", ".join(sorted(set(gguf.duplicate_metadata_keys)))
        raise ManifestError(f"duplicate GGUF metadata key(s): {names}")
    if gguf.duplicate_tensor_names:
        names = ", ".join(sorted(set(gguf.duplicate_tensor_names)))
        raise ManifestError(f"duplicate GGUF tensor name(s): {names}")
    if gguf.data_start > gguf.file_size:
        raise ManifestError(
            f"GGUF tensor data starts at {gguf.data_start}, beyond file size {gguf.file_size}"
        )

    records: list[dict[str, Any]] = []
    byte_census: Counter[str] = Counter()
    tensor_census: Counter[str] = Counter()
    parameter_census: Counter[str] = Counter()
    occupied: list[tuple[int, int, str]] = []

    for name in sorted(gguf.tensors):
        info = gguf.tensors[name]
        if any(int(dimension) <= 0 for dimension in info.shape):
            raise ManifestError(f"tensor {name} has non-positive shape {info.shape}")
        try:
            packed_bytes = info.n_bytes
        except ValueError as exc:
            raise ManifestError(str(exc)) from None
        if info.offset % gguf.alignment:
            raise ManifestError(
                f"tensor {name} offset {info.offset} is not aligned to {gguf.alignment} bytes"
            )
        start = gguf.data_start + info.offset
        end = start + packed_bytes
        if start < gguf.data_start or end > gguf.file_size:
            raise ManifestError(
                f"tensor {name} byte range [{start}, {end}) is outside file size {gguf.file_size}"
            )
        occupied.append((start, end, name))
        byte_census[info.dtype] += packed_bytes
        tensor_census[info.dtype] += 1
        parameter_census[info.dtype] += info.n_elements
        records.append({
            "name": name,
            "shape": list(info.shape),
            "dtype": info.dtype,
            "offset": info.offset,
            "absolute_offset": start,
            "packed_bytes": packed_bytes,
            "logical_parameters": info.n_elements,
            "alignment": gguf.alignment,
        })

    occupied.sort()
    for previous, current in zip(occupied, occupied[1:]):
        if current[0] < previous[1]:
            raise ManifestError(
                f"tensor byte ranges overlap: {previous[2]} [{previous[0]}, {previous[1]}) "
                f"and {current[2]} [{current[0]}, {current[1]})"
            )

    dtypes = {
        dtype: {
            "tensors": tensor_census[dtype],
            "logical_parameters": parameter_census[dtype],
            "packed_bytes": byte_census[dtype],
        }
        for dtype in sorted(byte_census)
    }
    return records, dtypes


@dataclass(frozen=True, slots=True)
class TensorView(Sequence[float]):
    """Immutable, flat F32/F16 view into an owning :class:`WeightStore`.

    The view stores byte offsets rather than an exported ``memoryview``.  That
    distinction lets ``WeightStore.close()`` release its mmap even while a
    caller retains a view; any subsequent access then fails deterministically
    instead of reading dangling storage or making ``mmap.close`` raise
    ``BufferError``.
    """

    _store: "WeightStore"
    name: str
    dtype: str
    shape: tuple[int, ...]
    _absolute_offset: int
    _element_offset: int
    _element_count: int

    @property
    def n_elements(self) -> int:
        return self._element_count

    @property
    def item_bytes(self) -> int:
        return _DENSE_VALUE_STRUCTS[self.dtype].size

    @property
    def packed_bytes(self) -> int:
        return self._element_count * self.item_bytes

    def __len__(self) -> int:
        return self._element_count

    def __getitem__(self, index: int | slice) -> float | tuple[float, ...]:
        if isinstance(index, slice):
            # Slices are explicitly materialized as an immutable tuple.  Hot
            # paths use matrix.row(), which remains a zero-copy offset view.
            return tuple(self[position] for position in range(*index.indices(len(self))))
        if not isinstance(index, int):
            raise TypeError(
                "tensor indices must be integers or slices, not "
                f"{type(index).__name__}"
            )
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"tensor index {index} is out of range for {self.name}")
        return self._store._read_dense_value(
            self.name,
            self.dtype,
            self._absolute_offset,
            self._element_offset + index,
        )

    def __iter__(self) -> Iterator[float]:
        # Resolve the live mmap once per iteration, but never return it to the
        # caller as an exported buffer.
        buffer = self._store._execution_buffer(self.name)
        unpack = _DENSE_VALUE_STRUCTS[self.dtype].unpack_from
        item_bytes = self.item_bytes
        byte_offset = self._absolute_offset + self._element_offset * item_bytes
        for index in range(self._element_count):
            value, = unpack(buffer, byte_offset + index * item_bytes)
            yield float(value)


@dataclass(frozen=True, slots=True)
class MatrixView:
    """Immutable GGML-order matrix view.

    GGUF stores ``shape[0]`` contiguously.  Consequently a tensor with GGUF
    shape ``(columns, rows)`` has ``rows`` output rows, and matvec computes
    ``row dot x`` for each contiguous row.  Embedding tables use the identical
    layout, so a token id selects one contiguous row.
    """

    _tensor: TensorView

    def __post_init__(self) -> None:
        if len(self._tensor.shape) != 2:
            raise ValueError(
                f"tensor {self._tensor.name} has shape {self._tensor.shape}, expected a matrix"
            )

    @property
    def name(self) -> str:
        return self._tensor.name

    @property
    def dtype(self) -> str:
        return self._tensor.dtype

    @property
    def shape(self) -> tuple[int, int]:
        return self._tensor.shape  # type: ignore[return-value]

    @property
    def columns(self) -> int:
        return self.shape[0]

    @property
    def rows(self) -> int:
        return self.shape[1]

    @property
    def n_elements(self) -> int:
        return self._tensor.n_elements

    @property
    def packed_bytes(self) -> int:
        return self._tensor.packed_bytes

    def row(self, row_index: int) -> TensorView:
        """Return one zero-copy contiguous row as an immutable vector view."""
        if row_index < 0:
            row_index += self.rows
        if row_index < 0 or row_index >= self.rows:
            raise IndexError(f"matrix row {row_index} is out of range for {self.name}")
        return TensorView(
            self._tensor._store,
            self.name,
            self.dtype,
            (self.columns,),
            self._tensor._absolute_offset,
            self._tensor._element_offset + row_index * self.columns,
            self.columns,
        )

    def embedding_row(self, token_id: int) -> TensorView:
        """Return the contiguous embedding vector selected by ``token_id``."""
        return self.row(token_id)

    def __getitem__(self, index: int | tuple[int, int]) -> TensorView | float:
        if isinstance(index, tuple):
            if len(index) != 2:
                raise IndexError("matrix indexing requires (row, column)")
            row_index, column_index = index
            return self.row(row_index)[column_index]
        return self.row(index)

    def matvec(self, values: Sequence[float]) -> array:
        """Compute a pure-stdlib matrix-vector product in GGML orientation."""
        if len(values) != self.columns:
            raise ValueError(
                f"{self.name} expects {self.columns} matvec inputs, got {len(values)}"
            )
        buffer = self._tensor._store._execution_buffer(self.name)
        unpack = _DENSE_VALUE_STRUCTS[self.dtype].unpack_from
        item_bytes = self._tensor.item_bytes
        matrix_start = (
            self._tensor._absolute_offset
            + self._tensor._element_offset * item_bytes
        )
        result = array("f", [0.0]) * self.rows
        for row_index in range(self.rows):
            total = 0.0
            row_start = matrix_start + row_index * self.columns * item_bytes
            for column_index in range(self.columns):
                weight, = unpack(buffer, row_start + column_index * item_bytes)
                total += float(weight) * float(values[column_index])
            result[row_index] = total
        return result

    def matvec_numpy(self, values: Sequence[float]):
        """Compute a float32 NumPy matvec without retaining an mmap export.

        NumPy is imported only when this explicitly optimized method is used;
        importing the weight layer remains standard-library-only.  The matrix
        view is local and the returned result is an owned copy, so closing the
        store after this call cannot leave a dangling exported mmap buffer.
        """

        if len(values) != self.columns:
            raise ValueError(
                f"{self.name} expects {self.columns} matvec inputs, got {len(values)}"
            )
        import numpy as np

        buffer = self._tensor._store._execution_buffer(self.name)
        dtype = np.dtype("<f4" if self.dtype == "F32" else "<f2")
        byte_offset = (
            self._tensor._absolute_offset
            + self._tensor._element_offset * self._tensor.item_bytes
        )
        matrix = np.frombuffer(
            buffer,
            dtype=dtype,
            count=self.n_elements,
            offset=byte_offset,
        ).reshape(self.rows, self.columns)
        vector = np.asarray(values, dtype=np.float32)
        result = np.asarray(matrix @ vector, dtype=np.float32).copy()
        # Drop every mmap-exporting ndarray before returning the owned result.
        del matrix
        return result

    def embedding_row_numpy(self, token_id: int):
        """Return one owned float32 NumPy embedding row."""

        import numpy as np

        row = self.row(token_id)
        buffer = row._store._execution_buffer(self.name)
        dtype = np.dtype("<f4" if row.dtype == "F32" else "<f2")
        byte_offset = row._absolute_offset + row._element_offset * row.item_bytes
        view = np.frombuffer(
            buffer, dtype=dtype, count=row.n_elements, offset=byte_offset,
        )
        result = np.asarray(view, dtype=np.float32).copy()
        del view
        return result


@dataclass(frozen=True, slots=True)
class PackedMatrixView:
    """Immutable adapter around Alpaccaroo's owned packed matrix storage."""

    _store: "PackedWeightStore"
    name: str
    dtype: str
    shape: tuple[int, int]
    _matrix: Any
    packed_bytes: int

    @property
    def columns(self) -> int:
        return self.shape[0]

    @property
    def rows(self) -> int:
        return self.shape[1]

    @property
    def n_elements(self) -> int:
        return self.columns * self.rows

    def matvec(self, values: Sequence[float]):
        self._store._ensure_open(self.name)
        if self._matrix is not None:
            return self._matrix.matvec(values)
        matrix = self._store._new_quant_matrix(self.name)
        return matrix.matvec(values)

    def matvec_numpy(self, values: Sequence[float]):
        self._store._ensure_open(self.name)
        if self._matrix is not None:
            return self._matrix.matvec(values)
        matrix = self._store._new_quant_matrix(self.name)
        return matrix.matvec(values)

    def embedding_row(self, token_id: int):
        self._store._ensure_open(self.name)
        if self._matrix is not None:
            return self._matrix.row(token_id)
        return self._store._quant_row(self.name, token_id)

    def embedding_row_numpy(self, token_id: int):
        self._store._ensure_open(self.name)
        row = (
            self._matrix.row(token_id) if self._matrix is not None
            else self._store._quant_row(self.name, token_id)
        )
        try:
            import numpy as np
        except Exception as exc:  # pragma: no cover - selected only with NumPy
            raise RuntimeError("NumPy embedding row requested without NumPy") from exc
        return np.asarray(row, dtype=np.float32).copy()

class WeightStore:
    """Owner of a validated read-only GGUF mapping and its immutable views.

    Opening performs the complete dtype preflight before any tensor values are
    allocated or expanded.  The store currently accepts only dense F32/F16
    execution files.  Callers must keep the store alive for as long as they use
    its views, normally with a ``with WeightStore.open(path) as weights`` block.
    """

    __slots__ = ("_gguf", "_closed")

    def __init__(self, gguf: GGUFFile):
        self._gguf = gguf
        self._closed = False

    @classmethod
    def open(cls, path: str | Path, *, prefetch: bool = False) -> "WeightStore":
        gguf = GGUFFile.open(path, prefetch=prefetch)
        try:
            # Check dtypes before layout validation asks for encoded sizes.  In
            # particular, unknown and not-yet-executable types get one stable,
            # explicit error rather than falling into a dequantizing fallback.
            unsupported = Counter(
                info.dtype for info in gguf.tensors.values()
                if info.dtype not in DENSE_EXECUTION_DTYPES
            )
            if unsupported:
                summary = ", ".join(
                    f"{dtype} ({unsupported[dtype]} tensor(s))"
                    for dtype in sorted(unsupported)
                )
                example_names = [
                    name for name in sorted(gguf.tensors)
                    if gguf.tensors[name].dtype not in DENSE_EXECUTION_DTYPES
                ][:6]
                examples = ", ".join(
                    f"{name}:{gguf.tensors[name].dtype}" for name in example_names
                )
                if sum(unsupported.values()) > len(example_names):
                    examples += ", ..."
                raise UnsupportedExecutionDType(
                    "unsupported execution tensor dtype(s) before weight allocation: "
                    f"{summary}; tensors include {examples}"
                )
            validate_tensor_layout(gguf)
            return cls(gguf)
        except Exception:
            gguf.close()
            raise

    @property
    def path(self) -> Path:
        return self._gguf.path

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def tensor_names(self) -> tuple[str, ...]:
        self._ensure_open()
        return tuple(sorted(self._gguf.tensors))

    def __len__(self) -> int:
        self._ensure_open()
        return len(self._gguf.tensors)

    def __contains__(self, name: object) -> bool:
        self._ensure_open()
        return isinstance(name, str) and name in self._gguf.tensors

    def _ensure_open(self, tensor_name: str = "") -> None:
        if self._closed or self._gguf._mm is None:
            subject = f" for tensor {tensor_name}" if tensor_name else ""
            raise RuntimeError(f"weight store is closed{subject}; its views are invalid")

    def _execution_buffer(self, tensor_name: str) -> Any:
        self._ensure_open(tensor_name)
        # This private reference never escapes the call stack as a memoryview;
        # struct.unpack_from reads directly from the read-only mmap.
        return self._gguf._mm

    def _read_dense_value(
        self,
        tensor_name: str,
        dtype: str,
        absolute_offset: int,
        element_index: int,
    ) -> float:
        buffer = self._execution_buffer(tensor_name)
        encoded = _DENSE_VALUE_STRUCTS[dtype]
        value, = encoded.unpack_from(
            buffer, absolute_offset + element_index * encoded.size
        )
        return float(value)

    def tensor(self, name: str) -> TensorView:
        """Return an immutable flat view without copying tensor data."""
        self._ensure_open(name)
        try:
            info = self._gguf.tensors[name]
        except KeyError:
            raise KeyError(f"GGUF has no tensor named {name!r}") from None
        if info.dtype not in DENSE_EXECUTION_DTYPES:
            # Normally unreachable because open() preflights the whole file;
            # retain the local guard so the invariant survives future stores
            # that also carry opaque packed tensors.
            raise UnsupportedExecutionDType(
                f"tensor {name} has unsupported execution dtype {info.dtype}"
            )
        return TensorView(
            self,
            info.name,
            info.dtype,
            info.shape,
            self._gguf.data_start + info.offset,
            0,
            info.n_elements,
        )

    def vector(self, name: str) -> TensorView:
        view = self.tensor(name)
        if len(view.shape) != 1:
            raise ValueError(f"tensor {name} has shape {view.shape}, expected a vector")
        return view

    def matrix(self, name: str) -> MatrixView:
        return MatrixView(self.tensor(name))

    def packed_bytes(self, name: str) -> int:
        self._ensure_open(name)
        try:
            return self._gguf.tensors[name].n_bytes
        except KeyError:
            raise KeyError(f"GGUF has no tensor named {name!r}") from None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._gguf.close()

    def __enter__(self) -> "WeightStore":
        self._ensure_open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort fallback for callers that do not use the context manager.
        try:
            self.close()
        except Exception:
            pass


class PackedWeightStore(WeightStore):
    """Validated dense-vector and packed/dense-matrix execution store.

    Quantized matrices are decoded once into :class:`QuantMatrix`'s compact
    execution representation.  Norms, biases and convolution vectors remain
    zero-copy dense views.  Unsupported encodings still fail as a complete
    preflight before any packed matrix is materialized.
    """

    __slots__ = ("_matrices", "_streaming")

    def __init__(self, gguf: GGUFFile):
        super().__init__(gguf)
        self._matrices: dict[str, PackedMatrixView | MatrixView] = {}
        self._streaming = (
            os.environ.get("ALPACCAROO_QWEN35_STREAM_WEIGHTS", "")
            .strip().lower() in ("1", "on", "yes", "true")
        )

    @classmethod
    def open(cls, path: str | Path, *, prefetch: bool = False) -> "PackedWeightStore":
        from .quants import QUANT_GEOMETRY

        gguf = GGUFFile.open(path, prefetch=prefetch)
        try:
            unsupported: list[str] = []
            for name, info in gguf.tensors.items():
                if info.dtype in DENSE_EXECUTION_DTYPES:
                    continue
                if len(info.shape) == 2 and info.dtype in QUANT_GEOMETRY:
                    continue
                unsupported.append(f"{name}:{info.dtype}")
            if unsupported:
                preview = ", ".join(sorted(unsupported)[:8])
                if len(unsupported) > 8:
                    preview += ", ..."
                raise UnsupportedExecutionDType(
                    "unsupported packed execution tensor(s) before weight "
                    f"allocation: {preview}"
                )
            validate_tensor_layout(gguf)
            return cls(gguf)
        except Exception:
            gguf.close()
            raise

    def matrix(self, name: str) -> PackedMatrixView | MatrixView:
        self._ensure_open(name)
        cached = self._matrices.get(name)
        if cached is not None:
            return cached
        try:
            info = self._gguf.tensors[name]
        except KeyError:
            raise KeyError(f"GGUF has no tensor named {name!r}") from None
        if len(info.shape) != 2:
            raise ValueError(f"tensor {name} has shape {info.shape}, expected a matrix")
        if info.dtype in DENSE_EXECUTION_DTYPES:
            result: PackedMatrixView | MatrixView = MatrixView(super().tensor(name))
        else:
            matrix = None if self._streaming else self._new_quant_matrix(name)
            result = PackedMatrixView(
                self, name, info.dtype, info.shape, matrix, info.n_bytes,
            )
        self._matrices[name] = result
        return result

    def _new_quant_matrix(self, name: str):
        """Materialize one compact CPU representation from native GGUF bytes.

        Streaming correctness mode deliberately does this per operation and
        releases it with the call; the normal decode mode caches the returned
        object in :class:`PackedMatrixView` exactly as before.
        """

        from .qmatrix import QuantMatrix

        self._ensure_open(name)
        info = self._gguf.tensors[name]
        start = self._gguf.data_start + info.offset
        raw = memoryview(self._execution_buffer(name))[start:start + info.n_bytes]
        try:
            return QuantMatrix(
                raw, info.dtype, rows=info.shape[1], cols=info.shape[0],
                # Qwen35's frozen llama.cpp log-probability gates require
                # float32 activations.  The conventional integer-activation
                # path preserved IDs but failed the 0.8B and 27B frozen
                # distribution tolerances, so it is never selected here.
                allow_int_dot=False,
            )
        finally:
            raw.release()

    def _quant_row(self, name: str, row_index: int):
        """Decode one native row for streaming embedding/conv gathers."""

        from .quants import _PURE_DECODERS

        self._ensure_open(name)
        info = self._gguf.tensors[name]
        rows = info.shape[1]
        if row_index < 0:
            row_index += rows
        if row_index < 0 or row_index >= rows:
            raise IndexError(f"row {row_index} is out of range for {name}")
        row_bytes = info.n_bytes // rows
        start = self._gguf.data_start + info.offset + row_index * row_bytes
        raw = memoryview(self._execution_buffer(name))[start:start + row_bytes]
        try:
            payload = bytes(raw)
        finally:
            raw.release()
        return _PURE_DECODERS[info.dtype](payload, info.shape[0])

    def close(self) -> None:
        if self._closed:
            return
        self._matrices.clear()
        super().close()
