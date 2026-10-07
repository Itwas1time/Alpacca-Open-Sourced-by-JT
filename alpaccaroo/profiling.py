# Alpaccaroo - structured decode profiler and execution-path reporting.
# MIT License. See LICENSE.
"""Where does a token's time actually go, and which kernel computed it?

Two questions, two answers, both produced by this module:

**Timing.** :class:`Profiler` collects wall-clock seconds into named,
non-overlapping buckets (``attn_qkv``, ``attention``, ``ffn_down``,
``output_proj``, ``sampler`` ...) plus a per-token list for p50/p95. The
engine calls :func:`current` once per forward pass and skips every timing
site when it returns ``None``, so an un-profiled run pays one module
attribute lookup per token and nothing else. Timing never touches the
numbers a kernel computes: the profiled and un-profiled paths emit the
same token ids, which ``tests/smoke.py`` pins down.

**Paths.** ``backend numpy`` is too coarse to answer "are the native
quantized kernels actually running?". :func:`model_paths` walks a loaded
model's weights and reports, per role, which of these executes it:

===========================  ============================================
``pure-python-dense``        list-of-lists matvec, no NumPy
``pure-python-quant``        block decode per row, no NumPy
``numpy-dense``              float32 ndarray, BLAS GEMV
``numpy-dense-hotcache``     quantized weights expanded by the hot-cache
``numpy-quant-einsum``       NumPy quantized fallback, large matrices
``numpy-quant-batched``      NumPy quantized fallback, small matrices
``numba-codes-f32``          our fused f32-scale kernel over int8 codes
``numba-int-q4k``            our native integer-dot Q4_K kernel
``numba-int-q5k``            our native integer-dot Q5_K kernel
``numba-int-q6k``            our native integer-dot Q6_K kernel
``gpu-cuda``                 our CUDA kernels, weights resident in VRAM
===========================  ============================================

**Environment.** :func:`runtime_info` records the thread counts, BLAS
identity, CPU feature set, GPU status and package versions a measurement
is only comparable within. Everything is best-effort and standard-library
plus what is already installed; anything undetectable reports ``None``
rather than a guess.
"""

from __future__ import annotations

import os
import platform
import sys
from time import perf_counter as _pc

__all__ = ["Profiler", "current", "enable", "disable", "model_paths",
           "runtime_info", "cpu_features", "format_report"]


# The one hot-path global. `current()` is a function so callers can hoist
# it out of their loop; the engine reads it once per forward pass.
ACTIVE: "Profiler | None" = None


def current() -> "Profiler | None":
    """The running profiler, or None. Hot-path callers hoist this."""
    return ACTIVE


# Buckets in report order. Everything a token spends time in must land in
# exactly one of them, or `orchestration` (decode minus the rest) stops
# meaning "Python glue" and starts meaning "double counted".
DECODE_BUCKETS = (
    ("embedding", "token embedding row gather"),
    ("norm", "RMSNorm (attention + FFN + final)"),
    ("attn_qkv", "q/k/v projection"),
    ("rope", "rotary embedding"),
    ("attention", "scores, softmax, weighted V"),
    ("attn_out", "attention output projection"),
    ("ffn_gate_up", "FFN gate + up projection"),
    ("ffn_act", "SwiGLU / GELU activation"),
    ("ffn_down", "FFN down projection"),
    ("output_proj", "vocabulary output projection"),
)

# Buckets timed outside the decode loop.
OTHER_BUCKETS = (
    ("model_load", "GGUF open, unpack, JIT warmup"),
    ("prompt_render", "chat template render + tokenize"),
    ("prefill", "prompt forward (batched)"),
    ("sampler", "repeat penalty, top-k, top-p, choice"),
    ("tokenizer_stream", "id -> text streaming decode"),
)

# Matvec buckets, for the "quantized matvec time" roll-up the plan asks for.
_MATVEC_BUCKETS = ("attn_qkv", "attn_out", "ffn_gate_up", "ffn_down",
                   "output_proj")


class Profiler:
    """Accumulates timings for one run. Not thread-safe by design: the
    engine serializes generation, and a lock in the decode loop would cost
    more than the thing it protects."""

    __slots__ = ("timers", "counters", "token_seconds", "prefill_seconds",
                 "meta", "in_prefill", "_t_run", "_overhead_probe")

    def __init__(self) -> None:
        self.timers: dict[str, float] = {}
        self.counters: dict[str, int] = {}
        self.token_seconds: list[float] = []
        self.prefill_seconds: list[tuple[int, float]] = []  # (tokens, seconds)
        self.meta: dict = {}
        # The pure-Python tier prefills by looping forward(), so without this
        # every prompt token would also be counted as a decoded one and the
        # reported decode tok/s would be a blend of two different workloads.
        self.in_prefill = False
        self._t_run = _pc()
        self._overhead_probe = 0.0

    # ---- collection ------------------------------------------------------

    def add(self, name: str, dt: float) -> None:
        self.timers[name] = self.timers.get(name, 0.0) + dt
        self.counters[name] = self.counters.get(name, 0) + 1

    def bump(self, name: str, n: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + n

    def token(self, dt: float) -> None:
        self.token_seconds.append(dt)
        self.timers["decode"] = self.timers.get("decode", 0.0) + dt

    def prefill(self, n_tokens: int, dt: float) -> None:
        self.prefill_seconds.append((n_tokens, dt))
        self.add("prefill", dt)

    def set(self, key: str, value) -> None:
        self.meta[key] = value

    def measure_overhead(self, samples: int = 2000) -> float:
        """Cost of one add() plus its two perf_counter calls, so the report
        can say how much of `orchestration` is the profiler itself."""
        t0 = _pc()
        for _ in range(samples):
            t = _pc()
            self.add("_probe", _pc() - t)
        self._overhead_probe = (_pc() - t0) / samples
        self.timers.pop("_probe", None)
        self.counters.pop("_probe", None)
        return self._overhead_probe

    # ---- derived numbers -------------------------------------------------

    @property
    def n_tokens(self) -> int:
        return len(self.token_seconds)

    def percentiles(self) -> dict:
        """p50/p95/p99/min/max of per-token decode latency, in seconds."""
        ts = sorted(self.token_seconds)
        if not ts:
            return {}

        def pct(p: float) -> float:
            # nearest-rank: with 20 samples p95 is the 19th, not an
            # interpolation between two tokens that never happened
            i = max(0, min(len(ts) - 1, int(round(p * len(ts) + 0.5)) - 1))
            return ts[i]

        return {"min": ts[0], "p50": pct(0.50), "p95": pct(0.95),
                "p99": pct(0.99), "max": ts[-1],
                "mean": sum(ts) / len(ts)}

    def decode_breakdown(self) -> dict:
        """Per-bucket seconds inside the decode loop, plus the residue.

        `orchestration` is decode wall-clock minus every bucket measured
        inside it: Python dispatch, allocation, and the profiler's own
        instrumentation. It is a residue, not a measurement, and is
        reported as such.
        """
        total = self.timers.get("decode", 0.0)
        parts = {name: self.timers.get(name, 0.0)
                 for name, _ in DECODE_BUCKETS if name in self.timers}
        accounted = sum(parts.values())
        parts["orchestration"] = total - accounted
        return parts

    def matvec_seconds(self) -> float:
        return sum(self.timers.get(b, 0.0) for b in _MATVEC_BUCKETS)

    # ---- output ----------------------------------------------------------

    def snapshot(self) -> dict:
        """A JSON-serializable record of this run."""
        n = self.n_tokens
        decode = self.timers.get("decode", 0.0)
        prefill_tokens = sum(t for t, _ in self.prefill_seconds)
        prefill_s = self.timers.get("prefill", 0.0)
        out = {
            "schema": 1,
            "wall_seconds": _pc() - self._t_run,
            "tokens_decoded": n,
            "decode_seconds": decode,
            "decode_tok_per_s": (n / decode) if decode > 0 else None,
            "prefill_tokens": prefill_tokens,
            "prefill_seconds": prefill_s,
            "prefill_tok_per_s": ((prefill_tokens / prefill_s)
                                  if prefill_s > 0 else None),
            "token_latency_seconds": self.percentiles(),
            "decode_breakdown_seconds": self.decode_breakdown(),
            "matvec_seconds": self.matvec_seconds(),
            "grouped_matvec_seconds": self.timers.get("matvec_grouped", 0.0),
            "grouped_matvec_calls": self.counters.get("matvec_grouped", 0),
            "timers_seconds": dict(sorted(self.timers.items())),
            "call_counts": dict(sorted(self.counters.items())),
            "profiler_overhead_per_sample_seconds": self._overhead_probe or None,
        }
        out.update(self.meta)
        return out


# ---- lifecycle --------------------------------------------------------------

def enable(**meta) -> Profiler:
    """Start profiling. Returns the profiler; also reachable via current()."""
    global ACTIVE
    ACTIVE = Profiler()
    for k, v in meta.items():
        ACTIVE.set(k, v)
    return ACTIVE


def disable() -> "Profiler | None":
    """Stop profiling and return the finished profiler."""
    global ACTIVE
    p, ACTIVE = ACTIVE, None
    return p


def finish_snapshot(p: "Profiler | None", model=None) -> dict:
    """A profiler's snapshot with the environment (and, if the model is
    still around, a fresh path scan) merged in - the record a benchmark
    writes to JSON and :func:`format_report` renders."""
    snap = p.snapshot() if p is not None else {"schema": 1}
    snap["runtime"] = runtime_info()
    if model is not None:
        snap["paths"] = model_paths(model)
        snap.setdefault("model_describe", model.describe())
    return snap


# ---- execution-path reporting -----------------------------------------------

#: Every label :func:`matrix_path` can return, so a reader can tell a new
#: path from a typo and tests can assert the set is closed.
PATH_LABELS = (
    "pure-python-dense", "pure-python-quant",
    "numpy-dense", "numpy-dense-hotcache",
    "numpy-quant-einsum", "numpy-quant-batched",
    "numba-codes-f32",
    "numba-int-q4k", "numba-int-q5k", "numba-int-q6k",
    "gpu-cuda", "absent",
)


def matrix_path(W) -> str:
    """Which kernel a single weight matrix's matvec will run through."""
    if W is None:
        return "absent"
    from . import tensor as T
    if getattr(W, "is_gpu_matrix", False):
        return "gpu-cuda"
    if T.is_quantized_matrix(W):
        return W.path_label()
    return "numpy-dense" if T.HAS_NUMPY else "pure-python-dense"


def _matrix_weights(W) -> int:
    if W is None:
        return 0
    rows = getattr(W, "rows", None)
    cols = getattr(W, "cols", None)
    if rows is not None and cols is not None:
        return int(rows) * int(cols)
    shape = getattr(W, "shape", None)
    if shape and len(shape) == 2:
        return int(shape[0]) * int(shape[1])
    try:  # pure-python list of rows
        return len(W) * len(W[0])
    except Exception:
        return 0


#: (role, accessor) for every matrix the decode path touches. Roles that a
#: given architecture fuses (wqk, wgu) or omits report as absent.
_ROLE_ACCESSORS = (
    ("attn_q", lambda ly: ly.wq),
    ("attn_k", lambda ly: ly.wk),
    ("attn_qk_fused", lambda ly: ly.wqk),
    ("attn_v", lambda ly: ly.wv),
    ("attn_output", lambda ly: ly.wo),
    ("ffn_gate", lambda ly: ly.w_gate),
    ("ffn_up", lambda ly: ly.w_up),
    ("ffn_gate_up_fused", lambda ly: ly.wgu),
    ("ffn_down", lambda ly: ly.w_down),
)


def model_paths(model) -> dict:
    """Per-role execution paths and weight counts for a loaded model.

    Returns ``{"roles": {role: {"path": ..., "matrices": n, "weights": w,
    "shape": [rows, cols]}}, "by_path": {path: weights}, "summary": str}``.
    A role whose layers disagree (mixed GPU/host placement after a partial
    upload) reports ``path`` as a ``"+"``-joined set, which is the honest
    answer rather than the first layer's.
    """
    roles: dict[str, dict] = {}

    def note(role: str, W) -> None:
        if W is None:
            return
        e = roles.setdefault(role, {"paths": {}, "matrices": 0, "weights": 0,
                                    "shape": None})
        p = matrix_path(W)
        w = _matrix_weights(W)
        e["paths"][p] = e["paths"].get(p, 0) + 1
        e["matrices"] += 1
        e["weights"] += w
        if e["shape"] is None:
            rows = getattr(W, "rows", None)
            cols = getattr(W, "cols", None)
            if rows is None or cols is None:
                shape = getattr(W, "shape", None)
                if shape and len(shape) == 2:
                    rows, cols = int(shape[0]), int(shape[1])
            if rows is not None and cols is not None:
                e["shape"] = [int(rows), int(cols)]

    note("token_embd", getattr(model, "tok_embd", None))
    for ly in getattr(model, "layers", ()):
        for role, get in _ROLE_ACCESSORS:
            try:
                note(role, get(ly))
            except AttributeError:
                pass
    out = getattr(model, "output", None)
    if out is not None and out is not getattr(model, "tok_embd", None):
        note("output", out)
    elif out is not None:
        note("output(tied)", out)

    by_path: dict[str, int] = {}
    clean: dict[str, dict] = {}
    for role, e in roles.items():
        label = "+".join(sorted(e["paths"]))
        clean[role] = {"path": label, "matrices": e["matrices"],
                       "weights": e["weights"], "shape": e["shape"]}
        for p, count in e["paths"].items():
            share = e["weights"] * count // max(e["matrices"], 1)
            by_path[p] = by_path.get(p, 0) + share
    total = sum(by_path.values())
    summary = ", ".join(
        f"{p} {100.0 * w / total:.0f}%"
        for p, w in sorted(by_path.items(), key=lambda kv: -kv[1])) if total else ""
    return {"roles": clean, "by_path": by_path, "summary": summary,
            "total_weights": total}


# ---- environment ------------------------------------------------------------

#: CPU features worth naming in a performance report: the ones our kernels'
#: codegen can actually exploit, plus the baseline they assume.
_FEATURES_OF_INTEREST = (
    "sse4.2", "avx", "f16c", "fma", "avx2",
    "avx512f", "avx512bw", "avx512vl", "avx512dq",
    "avx512vnni", "avxvnni", "avxvnniint8", "avx512_bf16", "amx-int8",
)

# Windows PF_* constants for IsProcessorFeaturePresent, the only feature
# probe the platform offers without a compiler.
_PF_WINDOWS = {"sse4.2": 38, "avx": 39, "avx2": 40, "avx512f": 41}


def cpu_features() -> dict:
    """Detected CPU model and the subset of _FEATURES_OF_INTEREST present.

    llvmlite (already installed with the kernels) knows the exact host
    feature map, which is the same information the JIT compiles against.
    Without it, Linux reads /proc/cpuinfo and Windows asks the kernel;
    anything else reports an empty set with ``source: "unknown"``, never a
    guessed one.
    """
    info = {"model": platform.processor() or None,
            "machine": platform.machine() or None,
            "features": [], "source": "unknown"}
    try:
        from llvmlite import binding as _llvm
        try:
            _llvm.initialize()
            _llvm.initialize_native_target()
        except Exception:
            pass  # already initialized by numba, which is the common case
        feats = _llvm.get_host_cpu_features()
        info["model"] = _llvm.get_host_cpu_name() or info["model"]
        info["features"] = [f for f in _FEATURES_OF_INTEREST if feats.get(f)]
        info["source"] = "llvmlite"
        return info
    except Exception:
        pass
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo") as fh:
                flags: set[str] = set()
                for line in fh:
                    if line.startswith(("flags", "Features")):
                        flags = set(line.split(":", 1)[1].split())
                        break
                    if line.startswith("model name") and not info["model"]:
                        info["model"] = line.split(":", 1)[1].strip()
            if flags:
                # /proc/cpuinfo spells them without punctuation
                alias = {"sse4.2": "sse4_2", "amx-int8": "amx_int8",
                         "avx512_bf16": "avx512_bf16"}
                info["features"] = [f for f in _FEATURES_OF_INTEREST
                                    if alias.get(f, f.replace(".", "_")) in flags]
                info["source"] = "/proc/cpuinfo"
        except OSError:
            pass
        return info
    if sys.platform == "win32":
        try:
            import ctypes
            k32 = ctypes.WinDLL("kernel32")
            info["features"] = [
                name for name, pf in _PF_WINDOWS.items()
                if k32.IsProcessorFeaturePresent(pf)]
            info["source"] = "IsProcessorFeaturePresent"
        except Exception:
            pass
    return info


#: Entry points that answer "how many threads will this BLAS fan out to".
#: OpenBLAS ships several spellings (plain, ILP64 `64_` suffix, and the
#: `scipy_` prefix numpy's bundled wheels rename everything to) and none of
#: them is guaranteed, so we try the known ones and then look for any
#: `*get_num_threads*` export in the binary itself.
_BLAS_THREAD_SYMBOLS = (
    "openblas_get_num_threads",
    "openblas_get_num_threads64_",
    "scipy_openblas_get_num_threads64_",
    "scipy_openblas_get_num_threads",
    "bli_thread_get_num_threads",
    "MKL_Get_Max_Threads",
)

_BLAS_PARALLEL_MODES = {0: "sequential", 1: "threaded", 2: "openmp"}


def _blas_libraries() -> list:
    """Shared objects that could be the active BLAS, newest wheel layout
    first. numpy vendors its own copy under ``numpy.libs`` (``.dylibs`` on
    macOS); a system BLAS reached through a different path is not probed,
    and reports ``threads: None`` rather than a made-up number."""
    import glob
    from pathlib import Path
    try:
        import numpy as np
    except Exception:
        return []
    root = Path(np.__file__).resolve().parent.parent
    out = []
    for sub in ("numpy.libs", "numpy/.dylibs", "numpy/.libs"):
        d = root / sub
        for pat in ("*blas*.dll", "*blas*.so*", "*blas*.dylib", "*mkl*"):
            out.extend(sorted(glob.glob(str(d / pat))))
    return out


def _blas_thread_symbol(path: str) -> "str | None":
    """Any ``*get_num_threads*`` export named inside the binary. Export
    names live in the file as plain ASCII on both PE and ELF, so one bounded
    scan finds a renamed build we have no hard-coded spelling for."""
    import re
    try:
        with open(path, "rb") as fh:
            blob = fh.read(256 << 20)  # bounded: a BLAS is tens of MB
    except OSError:
        return None
    found = set(m.decode("ascii", "replace")
                for m in re.findall(rb"[A-Za-z_][A-Za-z0-9_]{3,60}", blob)
                if b"get_num_threads" in m)
    if not found:
        return None
    # prefer the shortest: the trailing-underscore aliases are Fortran
    # wrappers around the same counter
    return sorted(found, key=lambda s: (len(s), s))[0]


def _blas_info() -> dict:
    """BLAS identity, thread count and threading mode, best effort.

    NumPy's build metadata names the library; the thread count only comes
    from the library itself, so we call its ``get_num_threads`` entry point.
    This matters here: a BLAS fanning out to every logical CPU while the
    kernels' own pool holds the physical cores is the two-pool contention
    the attention kernel exists to avoid, and it is invisible without this
    number.
    """
    out = {"name": None, "threads": None, "parallel": None, "library": None,
           "env": {}}
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
                "BLIS_NUM_THREADS"):
        if var in os.environ:
            out["env"][var] = os.environ[var]
    try:
        import numpy as np
        cfg = np.__config__.show(mode="dicts")  # numpy >= 1.25
        blas = (cfg.get("Build Dependencies", {}) or {}).get("blas", {})
        name, version = blas.get("name"), blas.get("version")
        if name:
            out["name"] = f"{name} {version}" if version else name
    except Exception:
        pass
    try:
        import ctypes
        from pathlib import Path
        for cand in _blas_libraries():
            try:
                lib = ctypes.CDLL(cand)
            except OSError:
                continue
            names = list(_BLAS_THREAD_SYMBOLS)
            scanned = _blas_thread_symbol(cand)
            if scanned and scanned not in names:
                names.append(scanned)
            for sym in names:
                fn = getattr(lib, sym, None)
                if fn is None:
                    continue
                fn.restype = ctypes.c_int
                try:
                    out["threads"] = int(fn())
                except Exception:
                    continue
                out["library"] = Path(cand).name
                for psym in ("openblas_get_parallel",
                             "openblas_get_parallel64_",
                             "scipy_openblas_get_parallel64_"):
                    pfn = getattr(lib, psym, None)
                    if pfn is not None:
                        pfn.restype = ctypes.c_int
                        try:
                            out["parallel"] = _BLAS_PARALLEL_MODES.get(
                                int(pfn()), str(int(pfn())))
                        except Exception:
                            pass
                        break
                return out
    except Exception:
        pass
    return out


def runtime_info() -> dict:
    """Everything a measurement is only comparable within."""
    from . import _platform, kernels
    from . import tensor as T

    info: dict = {
        "os": f"{platform.system()} {platform.release()}",
        "platform": platform.platform(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "python_impl": platform.python_implementation(),
        "logical_cores": os.cpu_count(),
        "physical_cores": _platform.physical_cores() or None,
        "backend": T.backend_name(),
        "backend_detail": T.backend_detail(),
        "cpu": cpu_features(),
        "blas": _blas_info(),
        "env": {k: v for k, v in sorted(os.environ.items())
                if k.startswith("ALPACCAROO_")},
    }
    try:
        import numpy
        info["numpy"] = numpy.__version__
    except Exception:
        info["numpy"] = None
    try:
        import numba
        info["numba"] = numba.__version__
        info["numba_threads"] = int(numba.get_num_threads())
        info["numba_threading_layer"] = str(
            getattr(numba.config, "THREADING_LAYER", "") or "") or None
    except Exception:
        info["numba"] = None
        info["numba_threads"] = None
        info["numba_threading_layer"] = None
    try:
        import llvmlite
        info["llvmlite"] = llvmlite.__version__
    except Exception:
        info["llvmlite"] = None
    info["kernels"] = kernels.status()
    info["kernels_active"] = kernels.available()
    info["int_dot_enabled"] = kernels.int_dot_enabled()
    try:
        from . import cuda
        info["gpu"] = cuda.doctor_line()
        info["gpu_active"] = bool(cuda.available())
    except Exception:
        info["gpu"] = None
        info["gpu_active"] = False
    return info


# ---- human-readable report ---------------------------------------------------

def _fmt_s(seconds: float) -> str:
    if seconds >= 1.0:
        return f"{seconds:.3f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.3f} ms"
    return f"{seconds * 1e6:.1f} us"


def format_report(snap: dict, width: int = 78) -> str:
    """Render :meth:`Profiler.snapshot` output (plus any environment and
    path sections merged into it) as a terminal report."""
    L: list[str] = []
    rule = "-" * width

    def head(title: str) -> None:
        L.append(rule)
        L.append(title)
        L.append(rule)

    head("alpaccaroo profile")
    if snap.get("model_path"):
        L.append(f"model            {snap['model_path']}")
    if snap.get("model_describe"):
        L.append(f"                 {snap['model_describe']}")
    rt = snap.get("runtime") or {}
    if rt:
        L.append(f"backend          {rt.get('backend_detail') or rt.get('backend')}")
        L.append(f"kernels          {rt.get('kernels')}")
        cpu = rt.get("cpu") or {}
        feats = ",".join(cpu.get("features") or []) or "none detected"
        L.append(f"cpu              {cpu.get('model') or 'unknown'} "
                 f"[{feats}]")
        L.append(f"cores            {rt.get('physical_cores') or '?'} physical / "
                 f"{rt.get('logical_cores') or '?'} logical | "
                 f"numba threads {rt.get('numba_threads') or '?'}"
                 f" ({rt.get('numba_threading_layer') or 'n/a'})")
        blas = rt.get("blas") or {}
        L.append(f"blas             {blas.get('name') or 'unknown'} | "
                 f"threads {blas.get('threads') if blas.get('threads') is not None else '?'}"
                 + (f" ({blas['parallel']})" if blas.get("parallel") else "")
                 + (f" | env {blas['env']}" if blas.get("env") else ""))
        L.append(f"versions         python {rt.get('python')} | "
                 f"numpy {rt.get('numpy')} | numba {rt.get('numba')} | "
                 f"llvmlite {rt.get('llvmlite')}")
        if rt.get("gpu_active"):
            L.append(f"gpu              {rt.get('gpu')}")
        if rt.get("env"):
            L.append(f"env              {rt['env']}")

    paths = snap.get("paths") or {}
    if paths.get("roles"):
        head("execution paths (weights by kernel)")
        for role, e in sorted(paths["roles"].items(),
                              key=lambda kv: -kv[1]["weights"]):
            shape = e.get("shape")
            shape_s = f"{shape[0]}x{shape[1]}" if shape else "-"
            L.append(f"{role:<22} {e['path']:<22} {e['matrices']:>4} mat  "
                     f"{shape_s:>13}  {e['weights'] / 1e6:>8.1f} Mw")
        if paths.get("summary"):
            L.append(f"{'TOTAL':<22} {paths['summary']}")

    head("timing")
    if snap.get("timers_seconds", {}).get("model_load"):
        L.append(f"model load       {_fmt_s(snap['timers_seconds']['model_load'])}")
    if snap.get("timers_seconds", {}).get("prompt_render"):
        L.append(f"prompt render    {_fmt_s(snap['timers_seconds']['prompt_render'])}"
                 f"  ({snap.get('call_counts', {}).get('prompt_render', 0)} calls)")
    if snap.get("prefill_seconds"):
        tps = snap.get("prefill_tok_per_s")
        L.append(f"prefill          {_fmt_s(snap['prefill_seconds'])}"
                 f"  {snap.get('prefill_tokens', 0)} tokens"
                 + (f"  = {tps:.2f} tok/s" if tps else ""))
    dec = snap.get("decode_seconds") or 0.0
    n = snap.get("tokens_decoded") or 0
    if dec:
        tps = snap.get("decode_tok_per_s")
        L.append(f"decode           {_fmt_s(dec)}  {n} tokens"
                 + (f"  = {tps:.2f} tok/s" if tps else ""))
    # sampling and stream-decoding sit OUTSIDE model.forward, so they are
    # not part of the decode wall clock and would vanish from the report if
    # only the buckets inside it were printed
    timers = snap.get("timers_seconds") or {}
    counts = snap.get("call_counts") or {}
    for key, label in (("sampler", "sampler"),
                       ("tokenizer_stream", "tokenizer")):
        if timers.get(key):
            v = timers[key]
            L.append(f"{label:<17}{_fmt_s(v)}  {counts.get(key, 0)} calls"
                     f"  = {_fmt_s(v / max(counts.get(key, 1), 1))} each"
                     + (f"  ({100.0 * v / dec:.1f}% of decode)" if dec else ""))
    lat = snap.get("token_latency_seconds") or {}
    if lat:
        L.append("token latency    "
                 + "  ".join(f"{k}={_fmt_s(v)}" for k, v in
                             (("min", lat.get("min")), ("p50", lat.get("p50")),
                              ("p95", lat.get("p95")), ("p99", lat.get("p99")),
                              ("max", lat.get("max")))
                             if v is not None))

    bd = snap.get("decode_breakdown_seconds") or {}
    if bd and dec > 0:
        head(f"decode breakdown ({n} tokens)")
        L.append(f"{'bucket':<18}{'total':>12}{'per token':>13}{'share':>9}   "
                 f"description")
        order = [k for k, _ in DECODE_BUCKETS if k in bd] + ["orchestration"]
        desc = dict(DECODE_BUCKETS)
        desc["orchestration"] = "residue: python dispatch + profiler cost"
        for k in order:
            v = bd.get(k, 0.0)
            L.append(f"{k:<18}{_fmt_s(v):>12}{_fmt_s(v / max(n, 1)):>13}"
                     f"{100.0 * v / dec:>8.1f}%   {desc.get(k, '')}")
        mv = snap.get("matvec_seconds") or 0.0
        L.append(f"{'  = matvec':<18}{_fmt_s(mv):>12}"
                 f"{_fmt_s(mv / max(n, 1)):>13}{100.0 * mv / dec:>8.1f}%   "
                 f"all weight-matrix products")
        gm = snap.get("grouped_matvec_seconds") or 0.0
        if gm:
            L.append(f"{'  = grouped':<18}{_fmt_s(gm):>12}"
                     f"{_fmt_s(gm / max(n, 1)):>13}{100.0 * gm / dec:>8.1f}%   "
                     f"{snap.get('grouped_matvec_calls', 0)} shared-activation "
                     f"dispatches")
    if snap.get("profiler_overhead_per_sample_seconds"):
        ov = snap["profiler_overhead_per_sample_seconds"]
        calls = sum(snap.get("call_counts", {}).get(k, 0)
                    for k, _ in DECODE_BUCKETS)
        L.append(f"\nprofiler cost    ~{_fmt_s(ov)} per sample x {calls} samples "
                 f"= ~{_fmt_s(ov * calls)} of the total above")
    L.append(rule)
    return "\n".join(L)
