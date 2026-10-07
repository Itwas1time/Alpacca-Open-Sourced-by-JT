# Alpaccaroo - minimal tensor backend: NumPy when available, pure Python
# otherwise (set ALPACCAROO_PURE=1 to force the pure path).
# MIT License. See LICENSE.
"""The handful of dense operations the transformer needs.

Vectors are NumPy 1-D float32 arrays or Python lists of floats; matrices
are NumPy 2-D arrays, lists of row-lists, or :class:`alpaccaroo.qmatrix.
QuantMatrix` for weights kept in quantized form. The pure path is exact but
slow - it exists so the engine runs with zero dependencies, and so the two
implementations can verify each other in tests.
"""

from __future__ import annotations

import math
import os
from time import perf_counter as _perf

try:
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

if os.environ.get("ALPACCAROO_PURE"):
    _np = None

HAS_NUMPY = _np is not None

from .qmatrix import (  # noqa: E402  (re-exported quantized-matrix surface)
    QUANTIZED_MATVEC_DTYPES,
    _HOT_WEIGHT_ENV,
    QuantMatrix as QuantizedMatrix,
    _reset_hot_cache_state,
    can_quantized_matvec,
    hot_cache_stats,
)
from .quants import QK, QK_K  # noqa: E402,F401  (block sizes, for callers/tests)
from . import profiling as _profiling  # noqa: E402  (no alpaccaroo imports at its top)


# Flipped by alpaccaroo.cuda when the first weight matrix reaches VRAM, so
# backend_name() reflects what is actually running, not what is installed.
_GPU_ACTIVE = False


def backend_name() -> str:
    if _GPU_ACTIVE:
        return "gpu (cuda)"
    return "numpy" if HAS_NUMPY else "pure-python"


def backend_detail() -> str:
    """The coarse label plus what actually executes matrix products.

    ``backend numpy`` cannot distinguish a NumPy einsum fallback from our
    JIT'd integer-dot kernels, and those differ by more than 2x. This
    names the tier; :func:`alpaccaroo.profiling.model_paths` names the
    per-matrix path once a model is loaded.
    """
    if _GPU_ACTIVE:
        return "gpu-cuda (host fallback: " + _cpu_backend_detail() + ")"
    return _cpu_backend_detail()


def _cpu_backend_detail() -> str:
    if not HAS_NUMPY:
        return "pure-python"
    from . import kernels as _k
    if not _k.available():
        return "numpy (no kernels: einsum quantized matvec, BLAS dense)"
    if _k.int_dot_enabled():
        return f"numpy + alpaccaroo-kernels {_k.kernel_version()} (native integer-dot)"
    return (f"numpy + alpaccaroo-kernels {_k.kernel_version()} "
            f"(fused f32-scale, ALPACCAROO_INT_DOT off)")


def _is_gpu_matrix(W) -> bool:
    # duck-typed marker, not isinstance: importing alpaccaroo.cuda here would
    # drag the CUDA stack into every CPU-only import of this module
    return getattr(W, "is_gpu_matrix", False)


def quantized_matrix(data, dtype: str, rows: int, cols: int) -> QuantizedMatrix:
    return QuantizedMatrix(data, dtype, rows, cols)


def is_quantized_matrix(W) -> bool:
    return isinstance(W, QuantizedMatrix)


# ---- construction --------------------------------------------------------

def vector(values) -> "object":
    if HAS_NUMPY:
        return _np.asarray(values, dtype=_np.float32)
    return list(values)


def matrix(flat, rows: int, cols: int):
    """Build a matrix from flat data in row-major order (row = output)."""
    if HAS_NUMPY:
        return _np.asarray(flat, dtype=_np.float32).reshape(rows, cols)
    if isinstance(flat, list):
        return [flat[r * cols:(r + 1) * cols] for r in range(rows)]
    flat = list(flat)
    return [flat[r * cols:(r + 1) * cols] for r in range(rows)]


def zeros(n: int):
    if HAS_NUMPY:
        return _np.zeros(n, dtype=_np.float32)
    return [0.0] * n


def to_list(v) -> list:
    if HAS_NUMPY and isinstance(v, _np.ndarray):
        return v.tolist()
    return list(v)


# ---- core ops -------------------------------------------------------------

def stable_sigmoid(value: float) -> float:
    """Return sigmoid(value) without overflowing for large finite inputs."""
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def stable_softplus(value: float) -> float:
    """Return ``log(1 + exp(value))`` using a stable two-branch form."""
    if value > 0.0:
        return value + math.log1p(math.exp(-value))
    return math.log1p(math.exp(value))


def l2_normalize(values, epsilon: float):
    """L2-normalize with denominator ``max(sqrt(sum(x*x)), epsilon)``."""
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(f"epsilon must be positive and finite, got {epsilon!r}")
    vector_values = [float(value) for value in values]
    if not vector_values:
        raise ValueError("cannot L2-normalize an empty vector")
    if not all(math.isfinite(value) for value in vector_values):
        raise ValueError("L2 input must contain only finite values")
    den_scale = max(epsilon, max(abs(value) for value in vector_values))
    scaled_norm = math.sqrt(math.fsum(
        (value / den_scale) * (value / den_scale)
        for value in vector_values
    ))
    scaled_denominator = max(scaled_norm, epsilon / den_scale)
    normalized = [
        (value / den_scale) / scaled_denominator
        for value in vector_values
    ]
    return normalized

def matvec(W, x):
    """W (rows x cols) times x (cols) -> rows."""
    if isinstance(W, QuantizedMatrix) or _is_gpu_matrix(W):
        return W.matvec(x)
    if HAS_NUMPY:
        return W @ x
    return [sum(w * xv for w, xv in zip(row, x)) for row in W]


def _is_native_quant(W) -> bool:
    """True when W's matvec takes the integer-dot kernel, which is the one
    path that can share a pre-quantized activation vector."""
    return (isinstance(W, QuantizedMatrix)
            and W._mode in ("q4k_int", "q5k_int", "q6k_int")
            and W._dense_cache is None)


def _pair_indices(Ws, native):
    """(Q4_K index, partner index, partner mode) when the group holds
    exactly one native Q4_K beside exactly one native Q5_K or Q6_K and the
    two agree on width, else None.

    Two pairings are compiled, because two are what real files produce: a
    **Q4_K_M** file gives Q4_K attn_q/attn_k beside a Q6_K attn_v, and a
    **Q4_K_S** file gives the same group with a Q5_K attn_v. Shape- and
    dtype-gated on purpose - two partners, one of each kind, or a mode with
    no compiled pairing all fall through to the per-matrix dispatches
    unchanged, which is why adding a pairing is additive and cannot break
    the models that do not have it."""
    i4 = ip = None
    kind = ""
    for i, (W, f) in enumerate(zip(Ws, native)):
        if not f:
            continue
        if W._mode == "q4k_int":
            if i4 is not None:
                return None      # two Q4_Ks are already fused at load
            i4 = i
        elif W._mode in ("q5k_int", "q6k_int"):
            if ip is not None:
                return None      # no compiled three-way pairing
            ip, kind = i, W._mode
        else:
            return None          # a native mode with no compiled pairing
    if i4 is None or ip is None or Ws[i4].cols != Ws[ip].cols:
        return None
    return i4, ip, kind


def matvec_group(Ws, x):
    """Matvec several matrices against the SAME input vector.

    When two or more of them are native-mode quantized matrices, the int8
    activation quantization runs once instead of once per matrix - the
    result is bit-identical either way (quantize_acts is a deterministic
    function of x). Falls back to plain matvecs everywhere else.
    """
    if HAS_NUMPY:
        gpu = [W for W in Ws if _is_gpu_matrix(W)]
        if len(gpu) >= 2 and len({W.cols for W in gpu}) == 1:
            from . import cuda as _cuda
            dx = _cuda.upload_vector(x)
            if dx is not None:  # None degrades to per-matvec uploads
                # dispatch on the marker, never `W in gpu`: list membership
                # compares a dense ndarray against GpuMatrix with ==, and
                # NumPy broadcasts that into an ambiguous truth value
                return [W.matvec(x, dx) if _is_gpu_matrix(W)
                        else matvec(W, x) for W in Ws]
        # Precompute the flags rather than testing `W in native` per matrix:
        # a group can legitimately MIX a dense ndarray with native quantized
        # ones - ALPACCAROO_DENSE_WEIGHT_MB densifies attn_q one tier before
        # attn_k/attn_v - and `ndarray in [QuantMatrix, ...]` evaluates
        # `ndarray == QuantMatrix`, which NumPy broadcasts into "the truth
        # value of an array is ambiguous". Same failure the gpu branch above
        # documents; the marker-list form cannot hit it.
        native = [_is_native_quant(W) for W in Ws]
        if (sum(native) >= 2 and _HOT_WEIGHT_ENV not in os.environ
                and len({W.cols for W, f in zip(Ws, native) if f}) == 1):
            from . import kernels as _k
            p = _profiling.ACTIVE
            t0 = _perf() if p is not None else 0.0
            pre = _k.quantize_acts(x)
            pair = _pair_indices(Ws, native)
            if pair is not None and _k.group_kernel_enabled():
                # Packages D and K: the whole group in one parallel region.
                # Only the two compiled dtype pairings - Q4_K+Q6_K from a
                # Q4_K_M file and Q4_K+Q5_K from a Q4_K_S one; everything
                # else keeps the per-matrix dispatches below.
                i4, ip, kind = pair
                out = [None] * len(Ws)
                pair_fn = (_k.matvec_q4k_q6k_pair if kind == "q6k_int"
                           else _k.matvec_q4k_q5k_pair)
                out[i4], out[ip] = pair_fn(Ws[i4], Ws[ip], x, pre)
                for i, (W, f) in enumerate(zip(Ws, native)):
                    if out[i] is None:
                        out[i] = W.matvec(x, pre) if f else matvec(W, x)
                if p is not None:
                    p.bump("matvec_group_kernel")
            else:
                out = [W.matvec(x, pre) if f else matvec(W, x)
                       for W, f in zip(Ws, native)]
            if p is not None:
                p.add("matvec_grouped", _perf() - t0)
                p.bump("matvec_grouped_matrices", sum(native))
            return out
    return [matvec(W, x) for W in Ws]


def matmul_t(X, W):
    """X (batch x cols) times W.T (cols x rows) -> batch x rows."""
    if isinstance(W, QuantizedMatrix) or _is_gpu_matrix(W):
        return W.matmul_t(X)
    if not HAS_NUMPY:
        return [matvec(W, row) for row in X]
    return _np.asarray(X, dtype=_np.float32) @ W.T


def matrix_row(W, r: int):
    if isinstance(W, QuantizedMatrix) or _is_gpu_matrix(W):
        return W.row(r)
    if HAS_NUMPY:
        return W[r].copy()
    return list(W[r])


def matrix_rows(W, rows):
    if isinstance(W, QuantizedMatrix) or _is_gpu_matrix(W):
        return W.rows_at(rows)
    if HAS_NUMPY:
        return W[_np.asarray(rows, dtype=_np.int64)].copy()
    return [list(W[int(r)]) for r in rows]


def dot(a, b) -> float:
    if HAS_NUMPY:
        return float(a @ b)
    return sum(x * y for x, y in zip(a, b))


def add(a, b):
    if HAS_NUMPY:
        return a + b
    return [x + y for x, y in zip(a, b)]


def add_(a, b):
    """In-place a += b (returns a)."""
    if HAS_NUMPY:
        a += b
        return a
    for i, y in enumerate(b):
        a[i] += y
    return a


def mul(a, b):
    if HAS_NUMPY:
        return a * b
    return [x * y for x, y in zip(a, b)]


def scale(a, s: float):
    if HAS_NUMPY:
        return a * s
    return [x * s for x in a]


def silu(x):
    if HAS_NUMPY:
        return x / (1.0 + _np.exp(-x))
    out = [0.0] * len(x)
    for i, v in enumerate(x):
        if v >= 0:
            out[i] = v / (1.0 + math.exp(-v))
        else:  # avoid overflow in exp for very negative values
            e = math.exp(v)
            out[i] = v * e / (1.0 + e)
    return out


def gelu_pytorch_tanh(x):
    c = 0.7978845608028654
    if HAS_NUMPY:
        return 0.5 * x * (1.0 + _np.tanh(c * (x + 0.044715 * x * x * x)))
    return [0.5 * v * (1.0 + math.tanh(c * (v + 0.044715 * v * v * v)))
            for v in x]


def rmsnorm(x, weight, eps: float):
    if HAS_NUMPY:
        arr = x if getattr(x, "dtype", None) == _np.float32 else x.astype(_np.float32)
        if arr.ndim == 1:  # decode hot path: BLAS dot beats np.mean here
            inv = 1.0 / math.sqrt(float(arr @ arr) / arr.shape[0] + eps)
            return arr * inv * weight
        inv = 1.0 / _np.sqrt(_np.mean(arr * arr, axis=-1, keepdims=True) + eps)
        return arr * inv * weight
    ss = sum(v * v for v in x) / len(x)
    inv = 1.0 / math.sqrt(ss + eps)
    return [v * inv * w for v, w in zip(x, weight)]


def softmax(x):
    if HAS_NUMPY:
        m = _np.max(x)
        e = _np.exp(x - m)
        return e / _np.sum(e)
    m = max(x)
    e = [math.exp(v - m) for v in x]
    s = sum(e)
    return [v / s for v in e]


def argmax(x) -> int:
    if HAS_NUMPY:
        return int(_np.argmax(x))
    best, besti = x[0], 0
    for i, v in enumerate(x):
        if v > best:
            best, besti = v, i
    return besti
