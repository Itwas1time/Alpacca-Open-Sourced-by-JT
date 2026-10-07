# Alpaccaroo - the only module that branches on the host OS.
# MIT License. See LICENSE.
"""Host-OS shims for two things POSIX hands us for free.

The engine asks for a capability; this module supplies the best available
implementation and returns a neutral answer when the host has none. Every
OS-specific line in Alpaccaroo lives here, so `gguf.py` and `kernels.py` stay
platform-free and read the same everywhere.

In both functions the POSIX path comes first and returns immediately, so
Linux behaviour is exactly what it was before this module existed: the
fallbacks below it are only ever reached on hosts that lack the syscall.
"""

from __future__ import annotations

import ctypes
import mmap
import os

_CHUNK = 1 << 24  # 16 MiB: large enough that read() syscall overhead vanishes


def prefetch(mm: "mmap.mmap", fh) -> None:
    """Advise that the whole mapping is about to be read sequentially.

    Linux/BSD: madvise, which faults the mapping in asynchronously and
    costs nothing. Windows has no madvise (`mmap.mmap` has no such method),
    so we stream the file once instead: the loader's per-tensor touches
    then hit the page cache rather than taking a hard fault per 4 KiB page.
    Measured on a 4.58 GiB Q4_K_M model, cold: a 0.96 s read cuts
    Model.load from 14.8 s to 11.4 s.
    """
    try:
        mm.madvise(mmap.MADV_SEQUENTIAL)
        mm.madvise(mmap.MADV_WILLNEED)
        return
    except (AttributeError, ValueError, OSError):
        pass  # no madvise here; fall through to the read-based warm-up
    try:
        pos = fh.tell()
        fh.seek(0)
        while fh.read(_CHUNK):
            pass
        fh.seek(pos)
    except (OSError, ValueError):
        pass  # purely advisory: a failed warm-up only costs speed


def physical_cores() -> int:
    """Physical core count, or 0 when it cannot be determined.

    Decode is memory-bandwidth-bound and SMT siblings contend for the same
    load ports: measured on a 6C/12T Ryzen, 6 threads decode 9-16% faster
    than 12. Linux exposes the topology in sysfs; Windows answers the same
    question through GetLogicalProcessorInformation. Anywhere else we
    return 0 and leave the caller's default (all logical CPUs) alone
    rather than guess.
    """
    try:
        cores = set()
        base = "/sys/devices/system/cpu"
        for name in os.listdir(base):
            if not name.startswith("cpu") or not name[3:].isdigit():
                continue
            try:
                with open(f"{base}/{name}/topology/core_id") as f:
                    core = f.read().strip()
                with open(f"{base}/{name}/topology/physical_package_id") as f:
                    pkg = f.read().strip()
            except OSError:
                continue
            cores.add((pkg, core))
        if cores:
            return len(cores)
    except Exception:
        pass
    return _physical_cores_windows()


class _CacheDescriptor(ctypes.Structure):
    _fields_ = [("Level", ctypes.c_ubyte), ("Associativity", ctypes.c_ubyte),
                ("LineSize", ctypes.c_ushort), ("Size", ctypes.c_ulong),
                ("Type", ctypes.c_ulong)]


class _ProcessorInfoUnion(ctypes.Union):
    _fields_ = [("ProcessorCore", ctypes.c_ubyte), ("NumaNode", ctypes.c_ulong),
                ("Cache", _CacheDescriptor), ("Reserved", ctypes.c_ulonglong * 2)]


class _ProcessorInfo(ctypes.Structure):
    _fields_ = [("ProcessorMask", ctypes.c_size_t),
                ("Relationship", ctypes.c_ulong),
                ("u", _ProcessorInfoUnion)]


_RELATION_PROCESSOR_CORE = 0
_ERROR_INSUFFICIENT_BUFFER = 122


def _physical_cores_windows() -> int:
    """One entry per physical core is tagged RelationProcessorCore.

    The non-Ex call tops out at one 64-CPU processor group, which is why a
    zero result falls back to the caller's default rather than to a wrong
    number: on a >64-thread host we would rather not tune than mistune.
    """
    if not hasattr(ctypes, "WinDLL"):
        return 0
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        size = ctypes.c_ulong(0)
        kernel32.GetLogicalProcessorInformation(None, ctypes.byref(size))
        if ctypes.get_last_error() != _ERROR_INSUFFICIENT_BUFFER:
            return 0
        count = size.value // ctypes.sizeof(_ProcessorInfo)
        buf = (_ProcessorInfo * count)()
        if not kernel32.GetLogicalProcessorInformation(buf, ctypes.byref(size)):
            return 0
        return sum(1 for e in buf if e.Relationship == _RELATION_PROCESSOR_CORE)
    except Exception:
        return 0
