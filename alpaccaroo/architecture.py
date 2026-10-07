# Alpaccaroo - architecture backend registry and compatibility adapter.
# MIT License. See LICENSE.
"""Architecture-owned loading and execution contracts.

The conventional transformer remains a single optimized specialization.  New
architectures register once at file load instead of adding conditionals to its
layer loop.  Qwen35 owns a separate hybrid runtime and heterogeneous state;
the conventional transformer hot loops remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ArchitectureBackend(Protocol):
    architecture: str
    executable: bool

    def validate_manifest(self, path: str) -> dict[str, Any]: ...
    def load_model(self, model_type, path: str, *, n_ctx: int, progress: bool): ...
    def new_state(self, model, *, context: int, sequences: int, backend: str): ...
    def forward_token(self, model, token_id: int, state, *, trace=None): ...
    def prefill(self, model, token_ids, state, *, chunk_size: int, trace=None): ...
    def estimate_memory(self, path: str, *, context: int, sequences: int) -> dict[str, Any]: ...
    def describe(self, model) -> dict[str, Any]: ...


@dataclass(frozen=True)
class StandardArchitectureBackend:
    architecture: str
    executable: bool = True

    def validate_manifest(self, path: str) -> dict[str, Any]:
        from .gguf import GGUFFile

        with GGUFFile.open(path, prefetch=False) as gguf:
            if gguf.architecture != self.architecture:
                raise ValueError(
                    f"architecture dispatch expected {self.architecture!r}, "
                    f"got {gguf.architecture!r}"
                )
            return {
                "architecture": self.architecture,
                "block_count": gguf.get(f"{self.architecture}.block_count"),
                "tensor_count": len(gguf.tensors),
            }

    def load_model(self, model_type, path: str, *, n_ctx: int, progress: bool):
        return model_type._load_standard(path, n_ctx=n_ctx, progress=progress)

    def new_state(self, model, *, context: int, sequences: int, backend: str):
        if sequences != 1:
            raise ValueError("the conventional compatibility backend supports one live sequence")
        model.reset()
        return model

    def forward_token(self, model, token_id: int, state, *, trace=None):
        if state is not model:
            raise ValueError("standard backend state does not belong to this model")
        return model.forward(token_id)

    def prefill(self, model, token_ids, state, *, chunk_size: int, trace=None):
        if state is not model:
            raise ValueError("standard backend state does not belong to this model")
        return model.prefill(token_ids)

    def estimate_memory(self, path: str, *, context: int, sequences: int) -> dict[str, Any]:
        from .model import auto_budget_fit_mb

        estimate = auto_budget_fit_mb(path, n_ctx=context)
        return {
            "architecture": self.architecture,
            "sequences": sequences,
            "eligible_dense_mb": None if estimate is None else estimate[0],
            "fixed_mb": None if estimate is None else estimate[1],
        }

    def describe(self, model) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "backend": "standard-transformer-compatibility",
            "executable": True,
        }


@dataclass(frozen=True)
class Qwen35ArchitectureBackend:
    architecture: str = "qwen35"
    executable: bool = True

    def validate_manifest(self, path: str) -> dict[str, Any]:
        from .qwen35 import inspect_qwen35

        return inspect_qwen35(path, calculate_hash=False)

    def load_model(self, model_type, path: str, *, n_ctx: int, progress: bool):
        manifest = self.validate_manifest(path)
        import os
        from . import tensor as tensor_backend
        from .qwen35_runtime import Qwen35Model

        requested = os.environ.get("ALPACCAROO_QWEN35_BACKEND", "auto").strip().lower()
        if requested not in (
            "", "auto", "pure", "pure-python", "numpy", "numba", "rocm", "cuda",
        ):
            raise ValueError(
                "ALPACCAROO_QWEN35_BACKEND must be auto, pure, numpy, numba, rocm, "
                "or cuda; "
                f"got {requested!r}"
            )
        requested_label = requested or "auto"
        rocm_mode = os.environ.get("ALPACCAROO_ROCM", "").strip().lower()
        if rocm_mode not in (
            "", "0", "off", "no", "false", "force",
        ):
            raise ValueError("ALPACCAROO_ROCM must be 0 or force")
        rocm_forced = rocm_mode == "force"
        rocm_disabled = rocm_mode in ("0", "off", "no", "false") or bool(
            os.environ.get("ALPACCAROO_PURE")
        )

        from .qwen35_rocm import (
            Qwen35RocmChain,
            Qwen35RocmUnavailable,
            rocm_capability,
        )

        rocm_status = None
        select_rocm = requested == "rocm" or (
            requested_label == "auto"
            and not rocm_disabled
            and (rocm_forced or Qwen35RocmChain.production_qualified)
        )
        if select_rocm:
            # The first inspection validates the architecture.  ROCm admission
            # must use the runtime's actual context (whose zero/default policy
            # is 4096), not the model's potentially much larger trained limit.
            rocm_context = n_ctx or min(
                int(manifest["model"]["context_length"]), 4096,
            )
            if int(manifest["memory_plan"]["requested_context"]) != rocm_context:
                from .qwen35 import inspect_qwen35

                manifest = inspect_qwen35(
                    path,
                    context=rocm_context,
                    calculate_hash=False,
                )
            try:
                rocm_status = Qwen35RocmChain.preflight(manifest)
            except Qwen35RocmUnavailable as exc:
                raise RuntimeError(f"Qwen35 ROCm backend unavailable: {exc}") from None
            base = Qwen35Model.load(
                path, n_ctx=n_ctx, progress=progress, manifest=manifest,
            )
            try:
                graph = Qwen35RocmChain.create_production(
                    base, owns_model=True, capability=rocm_status,
                )
            except Qwen35RocmUnavailable as exc:
                base.close()
                raise RuntimeError(f"Qwen35 ROCm backend unavailable: {exc}") from None
            except Exception:
                base.close()
                raise
            graph.backend_selection = {
                "requested": requested_label,
                "selected": graph.backend_name,
                "fallback": False,
                "reason": (
                    f"ROCm {rocm_status.hip_version} on "
                    f"{rocm_status.device_name} ({rocm_status.gcn_arch_name}); "
                    f"shared-memory budget {rocm_status.memory_budget_bytes} bytes"
                ),
            }
            graph.rocm_status = rocm_status.descriptor()
            return graph
        if requested == "cuda":
            from .qwen35_cuda import Qwen35CudaChain, Qwen35CudaUnavailable

            base = Qwen35Model.load(
                path, n_ctx=n_ctx, progress=progress, manifest=manifest,
            )
            try:
                graph = Qwen35CudaChain.create_production(
                    base, owns_model=True,
                )
            except Qwen35CudaUnavailable as exc:
                base.close()
                raise RuntimeError(f"Qwen35 CUDA backend unavailable: {exc}") from None
            except Exception:
                base.close()
                raise
            graph.backend_selection = {
                "requested": "cuda",
                "selected": graph.backend_name,
                "fallback": False,
                "reason": (
                    "explicit CUDA qualification request satisfied; "
                    "automatic production selection remains disabled"
                ),
            }
            return graph
        if requested == "numba":
            from .qwen35_kernels import available

            if not available():
                raise RuntimeError(
                    "Qwen35 Numba backend was requested but pinned numba==0.65.1 "
                    "is unavailable or disabled"
                )
        use_numpy = requested in ("numpy", "numba") or (
            requested in ("", "auto") and tensor_backend.HAS_NUMPY
        )
        if use_numpy and not tensor_backend.HAS_NUMPY:
            raise RuntimeError(
                "Qwen35 NumPy backend was requested but NumPy is unavailable "
                "or ALPACCAROO_PURE=1 is set"
            )
        runtime_type = Qwen35Model
        if use_numpy:
            from .qwen35_numpy import Qwen35NumpyModel

            runtime_type = Qwen35NumpyModel
        model = runtime_type.load(
            path, n_ctx=n_ctx, progress=progress, manifest=manifest,
        )
        if requested_label == "auto":
            rocm_status = rocm_status or rocm_capability()
            reason = (
                f"auto selected {model.backend_name}; packed ROCm was not "
                f"selected: {rocm_status.reason}"
            )
        else:
            reason = f"explicit {requested_label} backend request satisfied"
        model.backend_selection = {
            "requested": requested_label,
            "selected": model.backend_name,
            "fallback": requested_label == "auto",
            "reason": reason,
        }
        return model

    def new_state(self, model, *, context: int, sequences: int, backend: str):
        if sequences != 1:
            raise ValueError("Qwen35 currently supports one sequence per ModelState")
        if backend not in (
            "", "auto", "pure", "pure-python", "numpy", "numba", "rocm", "cuda",
        ):
            raise ValueError(
                f"Qwen35 backend {backend!r} is unavailable; available: pure-python, numpy"
            )
        if backend == "numpy" and not model.backend_name.startswith(("numpy", "numba")):
            raise ValueError("the loaded Qwen35 model does not use the NumPy backend")
        if backend == "numba" and not model.backend_name.startswith("numba"):
            raise ValueError("the loaded Qwen35 model does not use the Numba backend")
        if backend == "cuda":
            raise ValueError("Qwen35 CUDA state requires a qualified CUDA model load")
        if backend == "rocm":
            raise ValueError("Qwen35 ROCm state requires a qualified ROCm model load")
        if backend in ("pure", "pure-python") and not model.backend_name.startswith("pure"):
            raise ValueError("the loaded Qwen35 model does not use the pure backend")
        if context not in (0, model.n_ctx):
            raise ValueError(
                "Qwen35 state context is fixed by model load; "
                f"requested {context}, loaded {model.n_ctx}"
            )
        return model.new_state()

    def forward_token(self, model, token_id: int, state, *, trace=None):
        return model.forward_token(token_id, state, trace=trace)

    def prefill(self, model, token_ids, state, *, chunk_size: int, trace=None):
        return model.prefill_state(
            token_ids, state, chunk_size=chunk_size, trace=trace,
        )

    def estimate_memory(self, path: str, *, context: int, sequences: int) -> dict[str, Any]:
        return self.validate_manifest(path)["memory_plan"]

    def describe(self, model) -> dict[str, Any]:
        return model.describe_data()


_REGISTRY: dict[str, ArchitectureBackend] = {}


def register_backend(backend: ArchitectureBackend) -> None:
    name = backend.architecture
    if not name or name in _REGISTRY:
        raise ValueError(f"architecture backend already registered or unnamed: {name!r}")
    _REGISTRY[name] = backend


def backend_for(architecture: str) -> ArchitectureBackend | None:
    return _REGISTRY.get(architecture)


def registered_architectures(*, executable_only: bool = False) -> tuple[str, ...]:
    return tuple(sorted(
        name for name, backend in _REGISTRY.items()
        if not executable_only or backend.executable
    ))


for _standard_architecture in (
    "llama", "mistral", "qwen2", "qwen3", "stablelm", "gemma", "gemma3",
):
    register_backend(StandardArchitectureBackend(_standard_architecture))
register_backend(Qwen35ArchitectureBackend())
