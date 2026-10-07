# Alpaccaroo - Qwen35 hybrid execution composition seam.
# MIT License. See LICENSE.
"""Simulator-only Qwen35 orchestration with injectable projections.

The CUDA boundary currently has semantic kernels and state transactions, but
no production-qualified packed projection launcher.  This module supplies the
small seam needed to qualify complete token composition without weakening that
boundary: projections are injected, while :class:`Qwen35Model` remains the
authority for layer ordering, recurrent/KV state mutation, and atomic publish.

``PackedHostProjectionExecutor`` exercises the exact packed GGUF tensor source
for an F32/quantized host reference.  It is not a CUDA executor.  A future CUDA
projection provider can implement the same two-method protocol and then be
qualified independently on real hardware.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from .memory import ModelState
from .packed_gpu import PackedHostMatrix, PackedTensorSource
from .qwen35 import inspect_qwen35
from .qwen35_runtime import Qwen35Model, TraceCallback


class ProjectionContractError(RuntimeError):
    """An injected projection provider violated the composition contract."""


class InjectedProjectionFault(RuntimeError):
    """Deterministic one-shot fault used to qualify transaction recovery."""


class ProjectionMatrix(Protocol):
    """Matrix metadata and reference operations visible to a provider."""

    name: str
    dtype: str
    columns: int
    rows: int

    def matvec(self, values: Sequence[float]) -> Sequence[float]: ...

    def embedding_row(self, token_id: int) -> Sequence[float]: ...


class ProjectionExecutor(Protocol):
    """Injectable packed/dense projection boundary for one-token execution.

    Implementations must return exactly ``matrix.rows`` values for ``matvec``
    and ``matrix.columns`` values for ``embedding``.  Returned values are
    frozen to float32 by the composing model before any semantic operation.
    The model owns transactions and state; providers must not mutate it.
    """

    backend_label: str

    def matvec(
        self, matrix: ProjectionMatrix, values: Sequence[float],
    ) -> Sequence[float]: ...

    def embedding(
        self, matrix: ProjectionMatrix, token_id: int,
    ) -> Sequence[float]: ...


@dataclass(frozen=True, slots=True)
class ProjectionCall:
    operation: str
    tensor_name: str
    dtype: str
    input_width: int
    output_width: int


class ReferenceProjectionExecutor:
    """Recording reference provider with an optional one-shot fault.

    This is useful for composition tests and for hosts without the pinned CUDA
    stack.  It deliberately calls the matrix's existing reference operation;
    its ``backend_label`` must never be interpreted as CUDA qualification.
    """

    backend_label = "reference-projections"

    def __init__(self) -> None:
        self.calls: list[ProjectionCall] = []
        self._fault: tuple[str, str, int] | None = None

    def arm_fault(
        self,
        tensor_name: str,
        *,
        operation: str = "matvec",
        occurrence: int = 1,
    ) -> None:
        if operation not in ("matvec", "embedding"):
            raise ValueError("projection fault operation must be matvec or embedding")
        if occurrence <= 0:
            raise ValueError("projection fault occurrence must be positive")
        self._fault = (operation, tensor_name, occurrence)

    def _record(self, call: ProjectionCall) -> None:
        self.calls.append(call)
        if self._fault is None:
            return
        operation, tensor_name, remaining = self._fault
        if (call.operation, call.tensor_name) != (operation, tensor_name):
            return
        remaining -= 1
        if remaining:
            self._fault = (operation, tensor_name, remaining)
            return
        self._fault = None
        raise InjectedProjectionFault(
            f"injected {operation} fault for {tensor_name}"
        )

    def matvec(
        self, matrix: ProjectionMatrix, values: Sequence[float],
    ) -> Sequence[float]:
        self._record(ProjectionCall(
            "matvec", matrix.name, matrix.dtype, len(values), matrix.rows,
        ))
        return matrix.matvec(values)

    def embedding(
        self, matrix: ProjectionMatrix, token_id: int,
    ) -> Sequence[float]:
        self._record(ProjectionCall(
            "embedding", matrix.name, matrix.dtype, 1, matrix.columns,
        ))
        return matrix.embedding_row(token_id)


class PackedHostProjectionExecutor:
    """Reference executor backed by exact packed GGUF row payloads.

    The source owns a separate read-only GGUF mapping.  Callers own this
    executor and must close it after the composed model has stopped executing.
    """

    backend_label = "packed-host-reference"

    def __init__(self, source: PackedTensorSource) -> None:
        self.source = source
        self._matrices: dict[str, PackedHostMatrix] = {}

    @classmethod
    def open(cls, path: str | Path) -> "PackedHostProjectionExecutor":
        return cls(PackedTensorSource.open(path))

    def _matrix(self, expected: ProjectionMatrix) -> PackedHostMatrix:
        packed = self._matrices.get(expected.name)
        if packed is None:
            packed = self.source.matrix(expected.name)
            self._matrices[expected.name] = packed
        spec = packed.spec
        if (
            spec.dtype != expected.dtype
            or spec.columns != expected.columns
            or spec.rows != expected.rows
        ):
            raise ProjectionContractError(
                f"packed tensor {expected.name} metadata differs from model view"
            )
        return packed

    def matvec(
        self, matrix: ProjectionMatrix, values: Sequence[float],
    ) -> Sequence[float]:
        return self._matrix(matrix).matvec(values)

    def embedding(
        self, matrix: ProjectionMatrix, token_id: int,
    ) -> Sequence[float]:
        return self._matrix(matrix).row(token_id)

    def close(self) -> None:
        self._matrices.clear()
        self.source.close()

    def __enter__(self) -> "PackedHostProjectionExecutor":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class PackedCudaProjectionExecutor:
    """Simulator-only provider using native-packed CUDA matrix operations.

    Every embedding lookup and matvec is dispatched through a cached
    ``PackedCudaMatrix``.  The remaining RMS/attention/recurrent/FFN semantic
    operations are still supplied by :class:`Qwen35Model`; this adapter proves
    projection and layer composition, not a device-resident model backend.
    """

    backend_label = "packed-cuda-simulator-projections"
    production_qualified = False
    qualification = "CUDA-simulator packed projections; host semantic composition"

    def __init__(self, source: PackedTensorSource) -> None:
        # Import lazily so hosts without the pinned CUDA packages can still use
        # every other composition provider in this module.
        from .qwen35_cuda import Qwen35CudaUnavailable, cuda_capability

        capability = cuda_capability(allow_simulator=True)
        if not capability.available:
            raise Qwen35CudaUnavailable(capability.reason)
        if not capability.simulator:
            raise Qwen35CudaUnavailable(
                "PackedCudaProjectionExecutor is restricted to the CUDA simulator"
            )
        self.source = source
        self.calls: list[ProjectionCall] = []
        self._matrices: dict[str, Any] = {}

    @classmethod
    def open(cls, path: str | Path) -> "PackedCudaProjectionExecutor":
        source = PackedTensorSource.open(path)
        try:
            return cls(source)
        except Exception:
            source.close()
            raise

    def _matrix(self, expected: ProjectionMatrix):
        matrix = self._matrices.get(expected.name)
        if matrix is None:
            from .qwen35_cuda import PackedCudaMatrix

            matrix = PackedCudaMatrix.from_source(
                self.source, expected.name, allow_simulator=True,
            )
            self._matrices[expected.name] = matrix
        if (
            matrix.dtype != expected.dtype
            or matrix.columns != expected.columns
            or matrix.rows != expected.rows
        ):
            raise ProjectionContractError(
                f"packed CUDA tensor {expected.name} metadata differs from model view"
            )
        return matrix

    def matvec(
        self, matrix: ProjectionMatrix, values: Sequence[float],
    ) -> Sequence[float]:
        self.calls.append(ProjectionCall(
            "matvec", matrix.name, matrix.dtype, len(values), matrix.rows,
        ))
        return self._matrix(matrix).matvec(values)

    def embedding(
        self, matrix: ProjectionMatrix, token_id: int,
    ) -> Sequence[float]:
        self.calls.append(ProjectionCall(
            "embedding", matrix.name, matrix.dtype, 1, matrix.columns,
        ))
        return self._matrix(matrix).embedding_row(token_id)

    def descriptor(self) -> dict[str, Any]:
        return {
            "backend": self.backend_label,
            "production_qualified": self.production_qualified,
            "qualification": self.qualification,
            "cached_matrices": len(self._matrices),
            "device_packed_bytes": sum(
                int(matrix.device_packed_bytes)
                for matrix in self._matrices.values()
            ),
            "all_native_packed": all(
                matrix.descriptor()["native_packed"]
                for matrix in self._matrices.values()
            ),
            "all_simulator": all(
                matrix.descriptor()["simulator"]
                for matrix in self._matrices.values()
            ),
        }

    def close(self) -> None:
        for matrix in self._matrices.values():
            matrix.close()
        self._matrices.clear()
        self.source.close()

    def __enter__(self) -> "PackedCudaProjectionExecutor":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _projection_result(
    values: Sequence[float], expected: int, operation: str,
) -> array:
    try:
        result = array("f", values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProjectionContractError(
            f"{operation} returned values that cannot be represented as float32"
        ) from exc
    if len(result) != expected:
        raise ProjectionContractError(
            f"{operation} returned {len(result)} values, expected {expected}"
        )
    return result


class ProjectedQwen35Model(Qwen35Model):
    """Complete hybrid semantic model whose matrix work is injected."""

    backend_name = "qwen35-composition-simulator"
    numerical_mode = "qwen35-composition-f32"

    def __init__(
        self,
        path: str | Path,
        manifest: dict[str, Any],
        *,
        projections: ProjectionExecutor,
        n_ctx: int = 0,
        progress: bool = True,
    ) -> None:
        super().__init__(path, manifest, n_ctx=n_ctx, progress=progress)
        self.projections = projections
        self.backend_name = f"qwen35-composed-{projections.backend_label}"
        self.weight_storage["backend"] = self.backend_name
        self.weight_storage["projection_executor"] = projections.backend_label

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        projections: ProjectionExecutor,
        n_ctx: int = 0,
        progress: bool = True,
        manifest: dict[str, Any] | None = None,
    ) -> "ProjectedQwen35Model":
        if manifest is None:
            manifest = inspect_qwen35(path, context=n_ctx, calculate_hash=True)
        return cls(
            path, manifest, projections=projections,
            n_ctx=n_ctx, progress=progress,
        )

    def _matvec(
        self, matrix: ProjectionMatrix, values: Sequence[float],
    ) -> array:
        return _projection_result(
            self.projections.matvec(matrix, values),
            matrix.rows,
            f"matvec {matrix.name}",
        )

    def _embedding(self, token_id: int) -> array:
        matrix = self.tok_embd
        return _projection_result(
            self.projections.embedding(matrix, token_id),
            matrix.columns,
            f"embedding {matrix.name}",
        )


class SimulatedHybridTokenExecutor:
    """Callable adapter for :class:`Qwen35CudaChain` transaction tests.

    This qualifies complete orchestration only.  It neither checks CUDA
    capability nor advertises production/device execution.
    """

    production_qualified = False
    qualification = "simulator/reference composition only"

    def __init__(
        self,
        model: ProjectedQwen35Model,
        *,
        trace: TraceCallback | None = None,
    ) -> None:
        self.model = model
        self.trace = trace

    def __call__(self, token_id: int, state: ModelState):
        return self.model.forward_token(token_id, state, trace=self.trace)


def simulated_hybrid_chain(
    model: ProjectedQwen35Model,
    *,
    checkpoint_interval: int = 128,
    trace: TraceCallback | None = None,
):
    """Build the existing checkpoint/replay chain around full composition."""

    from .qwen35_cuda import Qwen35CudaChain

    executor = SimulatedHybridTokenExecutor(model, trace=trace)
    return Qwen35CudaChain(
        model,
        executor,
        checkpoint_interval=checkpoint_interval,
        backend_label="qwen35-hybrid-composition-simulator",
    )


__all__ = [
    "InjectedProjectionFault",
    "PackedCudaProjectionExecutor",
    "PackedHostProjectionExecutor",
    "ProjectedQwen35Model",
    "ProjectionCall",
    "ProjectionContractError",
    "ProjectionExecutor",
    "ProjectionMatrix",
    "ReferenceProjectionExecutor",
    "SimulatedHybridTokenExecutor",
    "simulated_hybrid_chain",
]
