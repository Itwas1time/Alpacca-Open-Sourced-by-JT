# Alpaccaroo - Qwen35 target-device ROCm capability boundary.
# MIT License. See LICENSE.
"""Fail-closed ROCm environment detection for the Qwen35 target backend.

PyTorch deliberately retains its ``torch.cuda`` namespace when built for HIP.
This module therefore verifies the HIP runtime and the AMD architecture instead
of treating ``torch.cuda.is_available()`` as sufficient evidence.  Imports are
lazy so the standard-library backend never acquires a PyTorch dependency.

The packed Qwen35 ROCm chain is not implemented in this first slice.  A working
ROCm installation is reported separately from production execution readiness,
and :class:`Qwen35RocmChain` refuses selection until both are true.
"""

from __future__ import annotations

import importlib
import os
import platform
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping


PYTHON_PIN = (3, 12)
PYTORCH_PIN = "2.9.1"
TRITON_PIN = "3.5.1"
ROCM_RELEASE = "7.2.1"
HIP_SERIES = "7.2"
TARGET_GFX = "gfx1151"
TARGET_DEVICE = "Radeon 8060S / Ryzen AI Max+ 395"
SUPPORTED_DISTRO = "ubuntu"
SUPPORTED_DISTRO_VERSION = "24.04"
MINIMUM_KERNEL = (6, 14)
SHARED_MEMORY_RESERVE_BYTES = 8 * 1024**3

_FALSE_VALUES = frozenset(("0", "off", "no", "false"))
_TRUE_VALUES = frozenset(("force",))


class Qwen35RocmUnavailable(RuntimeError):
    """Raised when an explicit ROCm request cannot be honored exactly."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostics = None if diagnostics is None else dict(diagnostics)


@dataclass(frozen=True, slots=True)
class RocmCapability:
    available: bool
    reason: str
    python_version: str
    distro: str | None = None
    distro_version: str | None = None
    kernel: str | None = None
    torch_version: str | None = None
    hip_version: str | None = None
    triton_version: str | None = None
    device_index: int | None = None
    device_name: str | None = None
    gcn_arch_name: str | None = None
    device_memory_bytes: int | None = None
    device_available_bytes: int | None = None
    device_allocated_bytes: int | None = None
    physical_memory_bytes: int | None = None
    shared_pool_bytes: int | None = None
    memory_budget_bytes: int | None = None
    reserve_bytes: int = SHARED_MEMORY_RESERVE_BYTES
    kfd_accessible: bool = False
    render_accessible: bool = False
    compute_smoke: bool = False
    production_qualified: bool = False
    admission: dict[str, Any] | None = None

    def descriptor(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "python_version": self.python_version,
            "distro": self.distro,
            "distro_version": self.distro_version,
            "kernel": self.kernel,
            "torch_version": self.torch_version,
            "hip_version": self.hip_version,
            "triton_version": self.triton_version,
            "device_index": self.device_index,
            "device_name": self.device_name,
            "gcn_arch_name": self.gcn_arch_name,
            "target_gfx": TARGET_GFX,
            "target_device": TARGET_DEVICE,
            "rocm_release": ROCM_RELEASE,
            "device_memory_bytes": self.device_memory_bytes,
            "device_available_bytes": self.device_available_bytes,
            "device_allocated_bytes": self.device_allocated_bytes,
            "physical_memory_bytes": self.physical_memory_bytes,
            "shared_pool_bytes": self.shared_pool_bytes,
            "memory_budget_bytes": self.memory_budget_bytes,
            "reserve_bytes": self.reserve_bytes,
            "kfd_accessible": self.kfd_accessible,
            "render_accessible": self.render_accessible,
            "compute_smoke": self.compute_smoke,
            "production_qualified": self.production_qualified,
            "admission": self.admission,
        }


@dataclass(frozen=True, slots=True)
class RocmSwapActivity:
    """Linux swap-I/O counter delta observed across the ROCm preflight."""

    pswpin_before: int
    pswpin_after: int
    pswpout_before: int
    pswpout_after: int
    source: str = "injected"

    def __post_init__(self) -> None:
        for name in (
            "pswpin_before", "pswpin_after", "pswpout_before", "pswpout_after",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer counter")
        if self.pswpin_after < self.pswpin_before:
            raise ValueError("pswpin counter moved backwards")
        if self.pswpout_after < self.pswpout_before:
            raise ValueError("pswpout counter moved backwards")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("swap activity source must be a non-empty string")

    @property
    def pswpin_delta(self) -> int:
        return self.pswpin_after - self.pswpin_before

    @property
    def pswpout_delta(self) -> int:
        return self.pswpout_after - self.pswpout_before

    @property
    def active(self) -> bool:
        return bool(self.pswpin_delta or self.pswpout_delta)

    def descriptor(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "counter_unit": "pages",
            "pswpin_before": self.pswpin_before,
            "pswpin_after": self.pswpin_after,
            "pswpin_delta": self.pswpin_delta,
            "pswpout_before": self.pswpout_before,
            "pswpout_after": self.pswpout_after,
            "pswpout_delta": self.pswpout_delta,
            "active": self.active,
        }


def _version_prefix(version: Any, expected: str) -> bool:
    """Accept the pinned release plus vendor/build suffixes, never later releases."""

    value = str(version or "")
    return value == expected or value.startswith(expected + "+")


def _kernel_pair(release: str) -> tuple[int, int]:
    values = re.findall(r"\d+", release)
    if len(values) < 2:
        return (0, 0)
    return int(values[0]), int(values[1])


def _physical_memory_bytes() -> int | None:
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    total = pages * page_size
    return total if total > 0 else None


def _read_linux_swap_counters(
    vmstat_path: str | Path = "/proc/vmstat",
) -> tuple[int, int]:
    path = Path(vmstat_path)
    values: dict[str, int] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="ascii").splitlines(), start=1,
    ):
        parts = line.split()
        if not parts or parts[0] not in ("pswpin", "pswpout"):
            continue
        if len(parts) != 2:
            raise ValueError(f"{path}:{line_number}: malformed swap counter")
        try:
            value = int(parts[1])
        except ValueError as exc:
            raise ValueError(
                f"{path}:{line_number}: invalid {parts[0]} counter {parts[1]!r}"
            ) from exc
        if value < 0:
            raise ValueError(f"{path}:{line_number}: negative {parts[0]} counter")
        values[parts[0]] = value
    missing = sorted({"pswpin", "pswpout"} - values.keys())
    if missing:
        raise ValueError(f"{path}: missing swap counters {missing}")
    return values["pswpin"], values["pswpout"]


def _live_accelerator_memory(cuda: Any, device_index: int) -> tuple[int, int]:
    try:
        free_bytes, total_bytes = cuda.mem_get_info(device_index)
    except Exception as exc:
        raise RuntimeError(f"torch.cuda.mem_get_info failed: {exc}") from exc
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (free_bytes, total_bytes)
    ):
        raise RuntimeError("torch.cuda.mem_get_info returned non-integer byte counts")
    if total_bytes <= 0 or free_bytes < 0 or free_bytes > total_bytes:
        raise RuntimeError(
            "torch.cuda.mem_get_info returned invalid aperture values: "
            f"free={free_bytes}, total={total_bytes}"
        )
    return free_bytes, total_bytes


def _device_files() -> tuple[bool, bool]:
    mode = os.R_OK | os.W_OK
    kfd = os.access("/dev/kfd", mode)
    render = any(os.access(path, mode) for path in Path("/dev/dri").glob("renderD*"))
    return kfd, render


def _memory_limit_bytes(environ: Mapping[str, str]) -> tuple[int | None, str | None]:
    raw = environ.get("ALPACCAROO_ROCM_MEMORY_MB", "").strip()
    if not raw:
        return None, None
    try:
        mib = int(raw)
    except ValueError:
        return None, "ALPACCAROO_ROCM_MEMORY_MB must be a positive integer"
    if mib <= 0:
        return None, "ALPACCAROO_ROCM_MEMORY_MB must be a positive integer"
    return mib * 1024**2, None


def _architecture(properties: Any) -> str:
    for name in ("gcnArchName", "gcn_arch_name", "arch"):
        value = getattr(properties, name, None)
        if value:
            return str(value).split(":", 1)[0].strip().lower()
    return ""


def _unavailable(
    reason: str,
    *,
    python_version: str,
    details: Mapping[str, Any] | None = None,
) -> RocmCapability:
    values = dict(details or {})
    return RocmCapability(
        available=False,
        reason=reason,
        python_version=python_version,
        **values,
    )


def _run_compute_smoke(device_index: int) -> None:
    """Run one PyTorch allocation and one Triton JIT kernel lazily."""

    from .qwen35_rocm_probe import run_probe

    run_probe(device_index)


def rocm_capability(
    *,
    probe_compute: bool = False,
    _environ: Mapping[str, str] | None = None,
    _import_module: Callable[[str], Any] = importlib.import_module,
    _python_version: tuple[int, int] | None = None,
    _python_version_text: str | None = None,
    _system: str | None = None,
    _os_release: Mapping[str, str] | None = None,
    _kernel: str | None = None,
    _device_files_result: tuple[bool, bool] | None = None,
    _physical_bytes: int | None = None,
    _accelerator_memory_result: tuple[int, int] | None = None,
    _compute_probe: Callable[[int], None] | None = None,
) -> RocmCapability:
    """Inspect the pinned ROCm userspace and exact ``gfx1151`` target.

    The private keyword hooks keep offline tests deterministic without faking
    modules globally.  Production callers use the defaults.
    """

    environ = os.environ if _environ is None else _environ
    version_pair = sys.version_info[:2] if _python_version is None else _python_version
    version_text = (
        platform.python_version()
        if _python_version_text is None else _python_version_text
    )
    mode = environ.get("ALPACCAROO_ROCM", "").strip().lower()
    if mode and mode not in _FALSE_VALUES | _TRUE_VALUES:
        return _unavailable(
            "ALPACCAROO_ROCM must be 0 or force",
            python_version=version_text,
        )
    if mode in _FALSE_VALUES or environ.get("ALPACCAROO_PURE"):
        return _unavailable(
            "ROCm disabled by environment",
            python_version=version_text,
        )
    if tuple(version_pair) != PYTHON_PIN:
        return _unavailable(
            f"Python {version_text} is not the pinned Python 3.12 ROCm tier",
            python_version=version_text,
        )

    system = platform.system() if _system is None else _system
    try:
        release = (
            platform.freedesktop_os_release()
            if _os_release is None else dict(_os_release)
        )
    except OSError as exc:
        return _unavailable(
            f"cannot identify the ROCm qualification userspace: {exc}",
            python_version=version_text,
        )
    distro = str(release.get("ID", "")).lower() or None
    distro_version = str(release.get("VERSION_ID", "")) or None
    kernel = platform.release() if _kernel is None else _kernel
    details: dict[str, Any] = {
        "distro": distro,
        "distro_version": distro_version,
        "kernel": kernel,
    }
    if system != "Linux":
        return _unavailable(
            f"ROCm target requires Linux, got {system}",
            python_version=version_text,
            details=details,
        )
    if distro != SUPPORTED_DISTRO or not str(distro_version or "").startswith(
        SUPPORTED_DISTRO_VERSION
    ):
        return _unavailable(
            "ROCm qualification requires Ubuntu 24.04 userspace",
            python_version=version_text,
            details=details,
        )
    if _kernel_pair(kernel) < MINIMUM_KERNEL:
        return _unavailable(
            f"kernel {kernel} is older than the required 6.14 target tier",
            python_version=version_text,
            details=details,
        )

    kfd, render = _device_files() if _device_files_result is None else _device_files_result
    details.update(kfd_accessible=kfd, render_accessible=render)
    if not kfd or not render:
        return _unavailable(
            "ROCm target requires read/write /dev/kfd and a /dev/dri/renderD* node",
            python_version=version_text,
            details=details,
        )

    requested_limit, limit_error = _memory_limit_bytes(environ)
    if limit_error:
        return _unavailable(
            limit_error,
            python_version=version_text,
            details=details,
        )
    try:
        torch = _import_module("torch")
        triton = _import_module("triton")
    except Exception as exc:
        return _unavailable(
            f"pinned ROCm dependencies unavailable: {exc}",
            python_version=version_text,
            details=details,
        )

    torch_version = str(getattr(torch, "__version__", ""))
    triton_version = str(getattr(triton, "__version__", ""))
    hip_version = str(getattr(getattr(torch, "version", None), "hip", "") or "")
    details.update(
        torch_version=torch_version,
        triton_version=triton_version,
        hip_version=hip_version or None,
    )
    if not _version_prefix(torch_version, PYTORCH_PIN):
        return _unavailable(
            f"PyTorch {torch_version or 'unknown'} != pinned {PYTORCH_PIN}",
            python_version=version_text,
            details=details,
        )
    if not _version_prefix(triton_version, TRITON_PIN):
        return _unavailable(
            f"Triton {triton_version or 'unknown'} != pinned {TRITON_PIN}",
            python_version=version_text,
            details=details,
        )
    if not hip_version:
        return _unavailable(
            "PyTorch is not a HIP/ROCm build",
            python_version=version_text,
            details=details,
        )
    if not hip_version.startswith(HIP_SERIES):
        return _unavailable(
            f"HIP {hip_version} is not the pinned ROCm {ROCM_RELEASE} series",
            python_version=version_text,
            details=details,
        )

    try:
        cuda = torch.cuda
        if not cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is false for the HIP runtime")
        count = int(cuda.device_count())
        target: tuple[int, Any, str] | None = None
        detected: list[str] = []
        for index in range(count):
            properties = cuda.get_device_properties(index)
            architecture = _architecture(properties)
            name = str(cuda.get_device_name(index))
            detected.append(f"{name} ({architecture or 'unknown-arch'})")
            if architecture == TARGET_GFX and target is None:
                target = index, properties, name
        if target is None:
            raise RuntimeError(
                f"required {TARGET_GFX} device not found; detected: "
                + (", ".join(detected) if detected else "none")
            )
    except Exception as exc:
        return _unavailable(
            f"ROCm device discovery failed: {exc}",
            python_version=version_text,
            details=details,
        )

    device_index, properties, device_name = target
    architecture = _architecture(properties)

    compute_smoke = False
    if probe_compute:
        try:
            (_run_compute_smoke if _compute_probe is None else _compute_probe)(device_index)
            compute_smoke = True
        except Exception as exc:
            details["compute_smoke"] = False
            return _unavailable(
                f"ROCm PyTorch/Triton compute smoke failed: {exc}",
                python_version=version_text,
                details=details,
            )
    details["compute_smoke"] = compute_smoke

    property_memory = int(getattr(properties, "total_memory", 0) or 0)
    try:
        free_memory, device_memory = (
            _live_accelerator_memory(cuda, device_index)
            if _accelerator_memory_result is None
            else _accelerator_memory_result
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (free_memory, device_memory)
        ):
            raise RuntimeError("accelerator memory probe returned non-integer byte counts")
        if device_memory <= 0 or free_memory < 0 or free_memory > device_memory:
            raise RuntimeError(
                "accelerator memory probe returned invalid aperture values: "
                f"free={free_memory}, total={device_memory}"
            )
        if property_memory > 0:
            tolerance = max(256 * 1024**2, property_memory // 100)
            if abs(property_memory - device_memory) > tolerance:
                raise RuntimeError(
                    "device property and live aperture totals disagree: "
                    f"property={property_memory}, live={device_memory}"
                )
    except Exception as exc:
        return _unavailable(
            f"ROCm live aperture discovery failed: {exc}",
            python_version=version_text,
            details=details,
        )
    allocated_memory = device_memory - free_memory
    physical_memory = _physical_memory_bytes() if _physical_bytes is None else _physical_bytes
    pool_candidates = [value for value in (device_memory, physical_memory) if value and value > 0]
    if not pool_candidates:
        return _unavailable(
            "ROCm shared-memory capacity could not be determined",
            python_version=version_text,
            details=details,
        )
    shared_pool = min(pool_candidates)
    usable = shared_pool - SHARED_MEMORY_RESERVE_BYTES
    details.update(
        device_index=device_index,
        device_name=device_name,
        gcn_arch_name=architecture,
        device_memory_bytes=device_memory or None,
        device_available_bytes=free_memory,
        device_allocated_bytes=allocated_memory,
        physical_memory_bytes=physical_memory,
        shared_pool_bytes=shared_pool,
    )
    if usable <= 0:
        return _unavailable(
            "ROCm shared pool cannot preserve the required 8 GiB host reserve",
            python_version=version_text,
            details=details,
        )
    if requested_limit is not None and requested_limit > usable:
        return _unavailable(
            "ALPACCAROO_ROCM_MEMORY_MB exceeds the shared pool after the 8 GiB reserve",
            python_version=version_text,
            details=details,
        )
    budget = usable if requested_limit is None else requested_limit
    details["memory_budget_bytes"] = budget

    return RocmCapability(
        available=True,
        reason=(
            f"pinned ROCm environment detected on {device_name} ({TARGET_GFX}); "
            "packed Qwen35 execution remains unqualified"
        ),
        python_version=version_text,
        **details,
    )


class Qwen35RocmChain:
    """Reserved production boundary for the dedicated Qwen35 ROCm chain."""

    backend_name = "qwen35-rocm"
    production_qualified = False

    @classmethod
    def preflight(
        cls,
        manifest: Mapping[str, Any] | None = None,
        *,
        capability: RocmCapability | None = None,
        memory_snapshot: Any | None = None,
        swap_activity: RocmSwapActivity | None = None,
        host_runtime_bytes: int = 0,
        meminfo_path: str | Path = "/proc/meminfo",
        vmstat_path: str | Path = "/proc/vmstat",
    ) -> RocmCapability:
        """Qualify the stack and live shared-memory budget before model load.

        A manifest is required for the production path.  The optional injected
        values are deterministic offline-test seams; production callers use
        live ``mem_get_info``, ``/proc/meminfo``, and ``/proc/vmstat`` data.
        Historical swap residency is diagnostic only.  New swap I/O observed
        across preflight rejects the run, and swap never adds capacity.
        """

        swap_before: tuple[int, int] | None = None
        if manifest is not None and swap_activity is None:
            try:
                swap_before = _read_linux_swap_counters(vmstat_path)
            except (OSError, ValueError) as exc:
                raise Qwen35RocmUnavailable(
                    f"cannot establish active-swap preflight: {exc}"
                ) from exc

        capability = (
            rocm_capability(probe_compute=True)
            if capability is None else capability
        )
        if not capability.available:
            raise Qwen35RocmUnavailable(capability.reason)
        if manifest is None:
            if not cls.production_qualified:
                raise Qwen35RocmUnavailable(
                    "gfx1151 ROCm environment passed, but native packed Qwen35 "
                    "kernels and the transactional chain are not production-qualified"
                )
            raise Qwen35RocmUnavailable(
                "ROCm production preflight requires the inspected model manifest"
            )

        if swap_activity is None:
            try:
                swap_after = _read_linux_swap_counters(vmstat_path)
            except (OSError, ValueError) as exc:
                raise Qwen35RocmUnavailable(
                    f"cannot complete active-swap preflight: {exc}"
                ) from exc
            assert swap_before is not None
            try:
                swap_activity = RocmSwapActivity(
                    pswpin_before=swap_before[0],
                    pswpin_after=swap_after[0],
                    pswpout_before=swap_before[1],
                    pswpout_after=swap_after[1],
                    source=str(Path(vmstat_path)),
                )
            except ValueError as exc:
                raise Qwen35RocmUnavailable(
                    f"active-swap counters are not monotonic: {exc}"
                ) from exc
        elif not isinstance(swap_activity, RocmSwapActivity):
            raise TypeError("swap_activity must be a RocmSwapActivity")

        from .packed_gpu import (
            AmdApuMemorySnapshot,
            build_amd_apu_admission_plan,
            build_qwen35_placement_plan,
            read_linux_amd_apu_memory_snapshot,
        )

        aperture = capability.device_memory_bytes
        accelerator_allocated = capability.device_allocated_bytes
        budget = capability.memory_budget_bytes
        if aperture is None or accelerator_allocated is None or budget is None:
            raise Qwen35RocmUnavailable(
                "ROCm capability lacks live aperture/allocation evidence",
                diagnostics=capability.descriptor(),
            )
        if memory_snapshot is None:
            try:
                memory_snapshot = read_linux_amd_apu_memory_snapshot(
                    accelerator_aperture_bytes=aperture,
                    accelerator_allocated_bytes=accelerator_allocated,
                    meminfo_path=meminfo_path,
                )
            except (OSError, ValueError) as exc:
                raise Qwen35RocmUnavailable(
                    f"cannot read live shared-memory state: {exc}",
                    diagnostics=capability.descriptor(),
                ) from exc
        elif not isinstance(memory_snapshot, AmdApuMemorySnapshot):
            raise TypeError("memory_snapshot must be an AmdApuMemorySnapshot")
        if (
            memory_snapshot.accelerator_aperture_bytes != aperture
            or memory_snapshot.accelerator_allocated_bytes != accelerator_allocated
        ):
            raise Qwen35RocmUnavailable(
                "injected memory snapshot does not match ROCm live aperture evidence"
            )

        required_dtypes = tuple(
            manifest.get("memory_plan", {}).get("required_weight_dtypes", ())
        )
        if not required_dtypes:
            raise Qwen35RocmUnavailable(
                "model manifest lacks required ROCm weight dtype inventory"
            )
        try:
            placement = build_qwen35_placement_plan(
                dict(manifest),
                budget_bytes=budget,
                qualified_dtypes=required_dtypes,
            )
            admission = build_amd_apu_admission_plan(
                placement,
                memory_snapshot,
                host_weight_mapping_bytes=int(
                    manifest["memory_plan"]["packed_weight_bytes"]
                ),
                host_runtime_bytes=host_runtime_bytes,
                os_reserve_bytes=SHARED_MEMORY_RESERVE_BYTES,
                # Historical swapped pages do not provide capacity, but do not
                # prove current thrashing. Live pswpin/pswpout deltas do.
                require_swap_clear=False,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise Qwen35RocmUnavailable(
                f"cannot build ROCm shared-memory admission plan: {exc}"
            ) from exc

        admission_descriptor = admission.descriptor()
        admission_descriptor["swap_activity"] = swap_activity.descriptor()
        admission_descriptor["swap_policy"] = (
            "swap contributes zero capacity; historical SwapUsed is allowed; "
            "observed pswpin/pswpout activity rejects production"
        )
        admission_descriptor["reserve_accounting"] = {
            "os_survival_reserve_bytes": admission.os_reserve_bytes,
            "device_driver_workspace_reserve_bytes": (
                admission.device_driver_reserve_bytes
            ),
            "relationship": (
                "distinct: OS reserve remains outside the admitted physical "
                "budget; device reserve is included in device_required_bytes"
            ),
        }
        rejection_reasons = list(admission.rejection_reasons)
        if swap_activity.active:
            rejection_reasons.append(
                "active swap I/O observed during preflight: "
                f"pswpin_delta={swap_activity.pswpin_delta}, "
                f"pswpout_delta={swap_activity.pswpout_delta} pages"
            )
        admission_descriptor["admitted"] = not rejection_reasons
        admission_descriptor["rejection_reasons"] = rejection_reasons
        qualified_capability = replace(
            capability,
            reason=(
                f"pinned ROCm environment and model-specific AMD APU admission "
                f"passed on {capability.device_name} ({capability.gcn_arch_name})"
            ),
            production_qualified=cls.production_qualified,
            admission=admission_descriptor,
        )
        if rejection_reasons:
            raise Qwen35RocmUnavailable(
                "AMD APU shared-memory preflight rejected before production "
                "allocation: " + "; ".join(rejection_reasons),
                diagnostics=qualified_capability.descriptor(),
            )
        if not cls.production_qualified:
            raise Qwen35RocmUnavailable(
                "gfx1151 ROCm environment passed, but native packed Qwen35 "
                "kernels and the transactional chain are not production-qualified",
                diagnostics=qualified_capability.descriptor(),
            )
        return qualified_capability

    @classmethod
    def create_production(cls, model: Any, **kwargs):
        capability = kwargs.get("capability")
        if (
            not isinstance(capability, RocmCapability)
            or capability.admission is None
            or not capability.admission.get("admitted", False)
        ):
            raise Qwen35RocmUnavailable(
                "ROCm production construction requires a successful model-specific "
                "shared-memory preflight"
            )
        raise Qwen35RocmUnavailable(
            "Qwen35 ROCm production construction is not implemented"
        )


__all__ = [
    "HIP_SERIES",
    "PYTHON_PIN",
    "PYTORCH_PIN",
    "Qwen35RocmChain",
    "Qwen35RocmUnavailable",
    "ROCM_RELEASE",
    "RocmCapability",
    "RocmSwapActivity",
    "SHARED_MEMORY_RESERVE_BYTES",
    "TARGET_DEVICE",
    "TARGET_GFX",
    "TRITON_PIN",
    "rocm_capability",
]
