# Alpaccaroo - quantized matrix storage and matvec/matmul kernels.
# MIT License. See LICENSE.
"""Row-major quantized GGUF matrix weights.

NumPy backend
    At construction the raw GGUF blocks are unpacked once (via
    :func:`alpaccaroo.quants.np_unpack`) into int8 quant codes plus per-sub-block
    effective float32 scales/offsets, all owned copies - the source mmap can
    close immediately and no raw block bytes are retained. This keeps RAM at
    roughly 1.1-1.3 bytes per weight (vs 4 for float32) while decode reads
    the codes directly; nothing is re-unpacked per token.

    matvec uses the fastest exact-scale kernels measured for this layout
    (see README "Honest performance expectations"): a batched-matmul
    block-dot for small matrices and an einsum block-dot for large ones.
    Both compute out = sum_s d_eff[:, s] * (codes[:, s] @ x_s) (+ offsets),
    which is bit-equivalent algebra to dequantize-then-GEMV.

    The optional hot-cache budget globals (``ALPACCAROO_HOT_WEIGHT_MB``) are not
    synchronized for concurrent matvec callers: alpaccaroo's own server
    serializes generation behind a lock, but embedders doing concurrent
    inference with the budget set should serialize calls or leave it unset.

Pure backend
    Owns a copy of the raw block bytes and decodes rows on the fly with the
    pure decoders in :mod:`alpaccaroo.quants`. Slow but dependency-free; the
    model loader never wraps matrices in pure mode, this path exists so both
    backends can verify each other in tests.
"""

from __future__ import annotations

import math
import os
import weakref

from . import kernels as _kernels
from .quants import QUANT_GEOMETRY, dequantize

try:
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

if os.environ.get("ALPACCAROO_PURE"):
    _np = None

HAS_NUMPY = _np is not None

QUANTIZED_MATVEC_DTYPES = frozenset(QUANT_GEOMETRY)

# Below this many elements the batched-matmul kernel is meant to beat the
# einsum one. Measured on two different boxes it never does: einsum wins at
# every shape a llama- or Gemma-class model uses, by 1.07x to 4.3x. With the
# old 1M threshold the only shape that took the batched path was Gemma 3's
# attn_k/v (294912 elements), and that cost 1.2 ms per token across its 52
# matvecs. Default it off, and keep the knob so it can be re-measured rather
# than re-guessed on hardware with a different BLAS.
def _small_matvec_elems() -> int:
    raw = os.environ.get("ALPACCAROO_SMALL_MATVEC_ELEMS", "")
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


_SMALL_MATVEC_ELEMS = _small_matvec_elems()


# Batch size at which the tiled dequantize-then-GEMM path overtakes the fused
# kernels. The tiled path costs O(weights) to dequantize before it can call
# BLAS, so it charges a full-model dequantize even for a one-row batch; the
# fused kernels stream the codes instead and pay only for the work asked for.
# Measured on a 14336x4096 Q4_K matrix, milliseconds:
#
#     batch     1     8    16    32    64    96   128   192   256
#     fused   1.5  11.3  19.4  23.5  41.6  52.6  76.4 113.1 147.6
#     tiled  42.7  54.9  54.5  57.4  63.5  71.2  82.7  89.4 131.7
#
# They cross between 128 and 192. The default stops at 96, the last size where
# the fused path still wins clearly (1.35x), so a machine with faster BLAS
# than this one cannot be pushed into a regression. Re-measure, do not
# re-guess: it is a knob.
def _fused_matmul_max_batch() -> int:
    raw = os.environ.get("ALPACCAROO_FUSED_MATMUL_MAX_BATCH", "")
    try:
        return max(0, int(raw))
    except ValueError:
        return 96


# Native-mode (integer-dot) batched matmul: below this batch, loop the
# integer matvec (weights re-stream per row, but at 2.4x the f32 rate);
# above it, dequantize tiles and call BLAS. The f32 tiled path costs a
# full-model dequantize regardless of batch, so the crossover sits where
# batch * matvec_ms ~= tiled_ms. Measured on the 14336x4096 Q4_K shape:
# int matvec 0.53 ms, tiled ~43 ms -> ~80; default below it for safety.
def _int_matmul_max_batch() -> int:
    raw = os.environ.get("ALPACCAROO_INT_MATMUL_MAX_BATCH", "")
    try:
        return max(0, int(raw))
    except ValueError:
        return 64

_HOT_WEIGHT_ENV = "ALPACCAROO_HOT_WEIGHT_MB"
_HOT_CACHE_LIMIT_BYTES = None
_HOT_CACHE_USED_BYTES = 0
_HOT_CACHE_MATRICES = 0
_HOT_CACHE_OWNERS = weakref.WeakSet()


class QuantMatrix:
    """Quantized (rows x cols) matrix; quant blocks run along columns.

    A contiguous row range therefore maps to a contiguous range of the
    underlying blocks, which is what row slicing and the pure backend rely
    on.
    """

    __slots__ = ("dtype", "rows", "cols", "block_elements", "block_bytes",
                 "blocks_per_row", "sub_len", "n_sub", "data",
                 "_q", "_q3", "_small", "_d", "_m", "_mode",
                 "_qp", "_qh", "_sci", "_mni", "_dh", "_dmh",
                 "_dense_cache", "_dense_cache_bytes", "__weakref__")

    def __init__(
        self, data, dtype: str, rows: int, cols: int, *,
        allow_int_dot: bool = True,
    ):
        geom = QUANT_GEOMETRY.get(dtype)
        if geom is None:
            raise ValueError(f"quantized matvec does not support {dtype}")
        block_elements, block_bytes, sub_len, _affine = geom
        if cols % block_elements:
            raise ValueError(
                f"{dtype} matrix columns must be a multiple of {block_elements}")
        self.dtype = dtype
        self.rows = rows
        self.cols = cols
        self.block_elements = block_elements
        self.block_bytes = block_bytes
        self.blocks_per_row = cols // block_elements
        self.sub_len = sub_len
        self.n_sub = cols // sub_len
        expected = rows * self.blocks_per_row * block_bytes
        if len(data) != expected:
            raise ValueError(
                f"{dtype} matrix has {len(data)} bytes, expected {expected}")
        self._dense_cache = None
        self._dense_cache_bytes = 0
        self._mode = "codes"
        self._qp = self._qh = self._sci = self._mni = None
        self._dh = self._dmh = None
        if HAS_NUMPY:
            self._small = rows * cols < _SMALL_MATVEC_ELEMS
            self.data = None  # unpacked copies own everything; mmap may close
            if (allow_int_dot and not self._small
                    and dtype in ("Q4_K", "Q5_K", "Q6_K")
                    and _kernels.int_dot_enabled()):
                # native block fields for the integer-dot kernels: the
                # file's own 4-bit codes / 6-bit sub-scales / f16 supers,
                # 0.58 (Q4_K) and 1.07 (Q6_K) bytes per weight instead of
                # the 1.25 of int8 codes + f32 scales
                from .quants import (np_unpack_q4k_native,
                                     np_unpack_q5k_native,
                                     np_unpack_q6k_native)
                bpr = self.blocks_per_row
                if dtype == "Q4_K":
                    qp, sc, mn, dh, dmh = np_unpack_q4k_native(
                        data, rows * cols)
                    self._mode = "q4k_int"
                    self._qp = _np.ascontiguousarray(
                        qp.reshape(rows, cols // 2))
                    self._sci = _np.ascontiguousarray(
                        sc.reshape(rows, bpr, 8))
                    self._mni = _np.ascontiguousarray(
                        mn.reshape(rows, bpr, 8))
                    self._dh = _np.ascontiguousarray(dh.reshape(rows, bpr))
                    self._dmh = _np.ascontiguousarray(dmh.reshape(rows, bpr))
                elif dtype == "Q5_K":
                    qs5, qh5, sc, mn, dh, dmh = np_unpack_q5k_native(
                        data, rows * cols)
                    self._mode = "q5k_int"
                    self._qp = _np.ascontiguousarray(
                        qs5.reshape(rows, cols // 2))
                    self._qh = _np.ascontiguousarray(
                        qh5.reshape(rows, cols // 8))
                    self._sci = _np.ascontiguousarray(
                        sc.reshape(rows, bpr, 8))
                    self._mni = _np.ascontiguousarray(
                        mn.reshape(rows, bpr, 8))
                    self._dh = _np.ascontiguousarray(dh.reshape(rows, bpr))
                    self._dmh = _np.ascontiguousarray(dmh.reshape(rows, bpr))
                else:
                    q6, sc6, dh = np_unpack_q6k_native(data, rows * cols)
                    self._mode = "q6k_int"
                    self._q3 = _np.ascontiguousarray(
                        q6.reshape(rows, self.n_sub, sub_len))
                    self._sci = _np.ascontiguousarray(
                        sc6.reshape(rows, self.n_sub))
                    self._dh = _np.ascontiguousarray(dh.reshape(rows, bpr))
                self._q = self._d = self._m = None
                if self._mode in ("q4k_int", "q5k_int"):
                    self._q3 = None
                return
            from .quants import np_unpack
            q, d_eff, m_eff = np_unpack(data, rows * cols, dtype)
            q3 = q.reshape(rows, self.n_sub, sub_len)
            if self._small:
                # store small matrices in the batched-matmul kernel layout so
                # the per-token astype reads contiguously; _q3 stays the
                # logical element-order (rows, n_sub, sub_len) view of it
                self._q = _np.ascontiguousarray(q3.transpose(1, 0, 2))
                self._q3 = self._q.transpose(1, 0, 2)
            else:
                self._q = q3
                self._q3 = q3
            self._d = d_eff.reshape(rows, self.n_sub)
            self._m = None if m_eff is None else m_eff.reshape(rows, self.n_sub)
        else:
            self.data = bytes(data)
            self._small = False
            self._q = self._q3 = self._d = self._m = None

    def __del__(self):  # pragma: no cover - depends on interpreter shutdown
        try:
            if self._dense_cache is not None:
                _release_hot_cache_bytes(self._dense_cache_bytes)
        except Exception:
            pass

    # ---- introspection ---------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        return self.rows, self.cols

    def path_label(self) -> str:
        """Which kernel this matrix's matvec will actually run through.

        One of alpaccaroo.profiling.PATH_LABELS. Reports what a call made
        right now would do, so a hot-cached matrix says so rather than
        naming the kernel its codes would otherwise have taken.
        """
        if not HAS_NUMPY:
            return "pure-python-quant"
        if self._dense_cache is not None:
            return "numpy-dense-hotcache"
        if self._mode == "q4k_int":
            return "numba-int-q4k"
        if self._mode == "q5k_int":
            return "numba-int-q5k"
        if self._mode == "q6k_int":
            return "numba-int-q6k"
        if not self._small and _kernels.available():
            return "numba-codes-f32"
        return "numpy-quant-batched" if self._small else "numpy-quant-einsum"

    def storage_nbytes(self) -> int:
        """Bytes held by the quantized representation (excl. hot cache)."""
        if HAS_NUMPY:
            if self._mode == "q4k_int":
                return (self._qp.nbytes + self._sci.nbytes + self._mni.nbytes
                        + self._dh.nbytes + self._dmh.nbytes)
            if self._mode == "q5k_int":
                return (self._qp.nbytes + self._qh.nbytes + self._sci.nbytes
                        + self._mni.nbytes + self._dh.nbytes
                        + self._dmh.nbytes)
            if self._mode == "q6k_int":
                return self._q3.nbytes + self._sci.nbytes + self._dh.nbytes
            n = self._q.nbytes + self._d.nbytes
            if self._m is not None:
                n += self._m.nbytes
            return n
        return len(self.data)

    # ---- matvec (decode path) ---------------------------------------------

    def __matmul__(self, x):
        return self.matvec(x)

    def matvec(self, x, pre=None):
        """self (rows x cols) @ x (cols) -> (rows).

        `pre` optionally carries kernels.quantize_acts(x) so several
        native-mode matrices reading the same vector quantize it once;
        non-native modes ignore it.
        """
        if not HAS_NUMPY:
            return self._matvec_pure(x)
        if self._dense_cache is not None or _HOT_WEIGHT_ENV in os.environ:
            dense = self._dense_hot_cache()
            if dense is not None:
                return dense @ _np.asarray(x, dtype=_np.float32)
        if self._mode == "q4k_int":
            # integer-dot kernel on the native block fields: activations are
            # quantized to int8 per 256 block (see kernels.quantize_acts),
            # weights stream at ~0.58 B/weight instead of 1.25
            return _kernels.matvec_q4k_int(self._qp, self._sci, self._mni,
                                           self._dh, self._dmh, x, pre)
        if self._mode == "q5k_int":
            return _kernels.matvec_q5k_int(self._qp, self._qh, self._sci,
                                           self._mni, self._dh, self._dmh,
                                           x, pre)
        if self._mode == "q6k_int":
            return _kernels.matvec_q6k_int(self._q3, self._sci, self._dh, x,
                                           pre)
        if not self._small and _kernels.available():
            # our fused Python-source kernel, JIT-compiled by pinned numba:
            # reads the int8 codes once, ~10x the einsum path (measured)
            return _kernels.matvec_codes(self._q3, self._d, self._m, x)
        xs = _np.asarray(x, dtype=_np.float32).reshape(self.n_sub, self.sub_len)
        if self._small:
            # batched (n_sub) BLAS matmuls of (rows, sub_len) @ (sub_len, 1)
            blockdot = _np.matmul(self._q.astype(_np.float32),
                                  xs[:, :, None])[:, :, 0]      # (n_sub, rows)
            out = _np.einsum("sr,rs->r", blockdot, self._d)
        else:
            blockdot = _np.einsum("rsl,sl->rs", self._q3, xs)   # (rows, n_sub)
            out = _np.einsum("rs,rs->r", blockdot, self._d)
        if self._m is not None:
            out += self._m @ xs.sum(axis=1)
        return out

    def _matvec_pure(self, x):
        out = [0.0] * self.rows
        row_bytes = self.blocks_per_row * self.block_bytes
        for r in range(self.rows):
            wrow = dequantize(self.data[r * row_bytes:(r + 1) * row_bytes],
                              self.cols, self.dtype)
            out[r] = sum(w * xv for w, xv in zip(wrow, x))
        return out

    # ---- batched matmul (prefill path) -------------------------------------

    def matmul_t(self, X):
        """X (batch x cols) @ self.T -> (batch x rows)."""
        if not HAS_NUMPY:
            return [self.matvec(row) for row in X]
        X = _np.asarray(X, dtype=_np.float32)
        if X.ndim != 2 or X.shape[1] != self.cols:
            raise ValueError(
                f"expected ({X.shape[0]}, {self.cols}) input, got {X.shape}")
        if self._dense_cache is not None or _HOT_WEIGHT_ENV in os.environ:
            dense = self._dense_hot_cache()
            if dense is not None:
                return X @ dense.T
        if self._mode in ("q4k_int", "q5k_int", "q6k_int"):
            # the integer matvec streams weights at 2.4x the f32 kernel's
            # rate, so re-streaming them per batch row beats a full-model
            # dequantize up to a large batch; past that, tile + BLAS
            if X.shape[0] <= _int_matmul_max_batch():
                out = _np.empty((X.shape[0], self.rows), dtype=_np.float32)
                for i in range(X.shape[0]):
                    out[i] = self.matvec(X[i])
                return out
            out = _np.empty((X.shape[0], self.rows), dtype=_np.float32)
            # 32 MB tiles, not the f32 path's 4 MB: each iteration hands off
            # between the numba pool (dequant) and the BLAS pool (GEMM), and
            # small tiles pay that ping-pong per tile - measured 1077 ms vs
            # 271 ms for the same 28672x4096 matmul at 4 MB vs 32 MB tiles
            tile_rows = max(256, (32 << 20) // max(self.cols * 4, 1))
            buf = _np.empty((min(tile_rows, self.rows), self.cols),
                            dtype=_np.float32)
            for r0 in range(0, self.rows, tile_rows):
                r1 = min(self.rows, r0 + tile_rows)
                # the JIT dequant kernel, not _tile_f32: one parallel pass
                # instead of NumPy's multi-pass unpack (~2x, measured); the
                # buffer is reused so its pages fault in exactly once
                if self._mode == "q4k_int":
                    tile = _kernels.dequant_q4k_tile(
                        self._qp[r0:r1], self._sci[r0:r1], self._mni[r0:r1],
                        self._dh[r0:r1], self._dmh[r0:r1], buf[:r1 - r0])
                elif self._mode == "q5k_int":
                    tile = _kernels.dequant_q5k_tile(
                        self._qp[r0:r1], self._qh[r0:r1], self._sci[r0:r1],
                        self._mni[r0:r1], self._dh[r0:r1], self._dmh[r0:r1],
                        buf[:r1 - r0])
                else:
                    tile = _kernels.dequant_q6k_tile(
                        self._q3[r0:r1], self._sci[r0:r1], self._dh[r0:r1],
                        buf[:r1 - r0])
                out[:, r0:r1] = X @ tile.T
            return out
        # Below the crossover, stream the codes once with the fused kernel.
        # The tiled path underneath costs O(weights) no matter how small the
        # batch is, so a short prefill used to cost a full-model dequantize.
        if (not self._small and X.shape[0] <= _fused_matmul_max_batch()
                and self._q3.flags["C_CONTIGUOUS"] and _kernels.available()):
            return _kernels.matmul_codes(self._q3, self._d, self._m, X)
        out = _np.empty((X.shape[0], self.rows), dtype=_np.float32)
        x_sub_sums = None
        if self._m is not None:
            x_sub_sums = X.reshape(X.shape[0], self.n_sub, self.sub_len).sum(axis=2)
        # dequantize ~4 MB row tiles; GGUF rows are contiguous so each tile
        # is a plain slice of the code/scale arrays
        tile_rows = max(16, (4 << 20) // max(self.cols * 4, 1))
        q3 = self._q3
        for r0 in range(0, self.rows, tile_rows):
            r1 = min(self.rows, r0 + tile_rows)
            tile = _np.ascontiguousarray(q3[r0:r1], dtype=_np.float32)
            tile *= self._d[r0:r1, :, None]
            out[:, r0:r1] = X @ tile.reshape(r1 - r0, self.cols).T
            if self._m is not None:
                out[:, r0:r1] += x_sub_sums @ self._m[r0:r1].T
        return out

    # ---- native-mode f32 expansion (batched matmul tiles, row access) ------

    def _tile_f32(self, r0: int, r1: int):
        """Dequantize rows [r0, r1) of a native-mode matrix to float32.

        Same math as the per-token kernels' f32 side: d_eff = f16(d) * sc,
        m_eff = -f16(dmin) * mn, value = d_eff * code + m_eff.
        """
        if self._mode == "q4k_int":
            return _expand_q4k_f32(self._qp[r0:r1], self._sci[r0:r1],
                                   self._mni[r0:r1], self._dh[r0:r1],
                                   self._dmh[r0:r1], self.n_sub, self.sub_len)
        if self._mode == "q5k_int":
            return _expand_q5k_f32(self._qp[r0:r1], self._qh[r0:r1],
                                   self._sci[r0:r1], self._mni[r0:r1],
                                   self._dh[r0:r1], self._dmh[r0:r1],
                                   self.n_sub, self.sub_len)
        if self._mode == "q6k_int":
            return _expand_q6k_f32(self._q3[r0:r1], self._sci[r0:r1],
                                   self._dh[r0:r1])
        raise RuntimeError(f"_tile_f32 called in mode {self._mode}")

    # ---- row access (embedding lookups) ------------------------------------

    def row(self, i: int):
        if not 0 <= i < self.rows:
            raise IndexError(i)
        if not HAS_NUMPY:
            row_bytes = self.blocks_per_row * self.block_bytes
            return dequantize(self.data[i * row_bytes:(i + 1) * row_bytes],
                              self.cols, self.dtype)
        if self._dense_cache is not None or _HOT_WEIGHT_ENV in os.environ:
            _sync_hot_cache_budget(_hot_cache_limit_bytes())
            if self._dense_cache is not None:
                return self._dense_cache[i].copy()
        if self._mode != "codes":
            return self._tile_f32(i, i + 1).reshape(-1)
        v = self._q3[i].astype(_np.float32)
        v *= self._d[i, :, None]
        if self._m is not None:
            v += self._m[i, :, None]
        return v.reshape(-1)

    def rows_at(self, rows):
        if not HAS_NUMPY:
            return [self.row(int(r)) for r in rows]
        idx = _np.asarray(rows, dtype=_np.int64)
        if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= self.rows):
            raise IndexError("row index out of range")
        if not idx.size:
            # the native expanders reshape by block width, which numpy
            # rejects for zero rows; every mode agrees an empty gather is
            # an empty (0, cols) result
            return _np.empty((0, self.cols), dtype=_np.float32)
        if self._dense_cache is not None or _HOT_WEIGHT_ENV in os.environ:
            _sync_hot_cache_budget(_hot_cache_limit_bytes())
            if self._dense_cache is not None:
                return self._dense_cache[idx].copy()
        if self._mode == "q4k_int":
            # fancy-index gathers, then the same expansion the tile path
            # uses: one vectorized pass instead of a Python loop per row
            # (a 900-token prompt gathers 900 embedding rows at once)
            return _expand_q4k_f32(self._qp[idx], self._sci[idx],
                                   self._mni[idx], self._dh[idx],
                                   self._dmh[idx], self.n_sub, self.sub_len)
        if self._mode == "q5k_int":
            return _expand_q5k_f32(self._qp[idx], self._qh[idx],
                                   self._sci[idx], self._mni[idx],
                                   self._dh[idx], self._dmh[idx],
                                   self.n_sub, self.sub_len)
        if self._mode == "q6k_int":
            return _expand_q6k_f32(self._q3[idx], self._sci[idx],
                                   self._dh[idx])
        v = self._q3[idx].astype(_np.float32)
        v *= self._d[idx][:, :, None]
        if self._m is not None:
            v += self._m[idx][:, :, None]
        return v.reshape(idx.size, self.cols)

    # ---- optional dense f32 cache (ALPACCAROO_HOT_WEIGHT_MB) -------------------

    def _dense_from_storage(self):
        if self._mode != "codes":
            return self._tile_f32(0, self.rows)
        v = self._q3.astype(_np.float32)
        v *= self._d[:, :, None]
        if self._m is not None:
            v += self._m[:, :, None]
        return v.reshape(self.rows, self.cols)

    def _dense_hot_cache(self):
        if not HAS_NUMPY:
            return None
        _sync_hot_cache_budget(_hot_cache_limit_bytes())
        if self._dense_cache is not None:
            return self._dense_cache
        nbytes = self.rows * self.cols * 4
        if not _reserve_hot_cache_bytes(nbytes):
            return None
        try:
            dense = self._dense_from_storage()
        except Exception:
            _release_hot_cache_bytes(nbytes)
            raise
        self._dense_cache = dense
        self._dense_cache_bytes = nbytes
        _HOT_CACHE_OWNERS.add(self)
        return dense


# ---- native-mode f32 expansion helpers -------------------------------------
# One implementation for both the contiguous tile slices (prefill, hot cache)
# and the fancy-indexed row gathers (embedding lookups): the math must be
# bit-identical everywhere or the row/tile parity tests would drift apart.

def _expand_q4k_f32(qp, sci, mni, dh, dmh, n_sub, sub_len):
    nr = qp.shape[0]
    qp3 = qp.reshape(nr, -1, 32)
    codes = _np.empty((nr, qp3.shape[1], 64), dtype=_np.uint8)
    codes[:, :, :32] = qp3 & 0x0F
    codes[:, :, 32:] = qp3 >> 4
    d_eff = (dh.view(_np.float16).astype(_np.float32)[:, :, None]
             * sci.astype(_np.float32))
    m_eff = (dmh.view(_np.float16).astype(_np.float32)[:, :, None]
             * mni.astype(_np.float32))
    v = codes.reshape(nr, n_sub, sub_len).astype(_np.float32)
    v *= d_eff.reshape(nr, n_sub, 1)
    v -= m_eff.reshape(nr, n_sub, 1)
    return v.reshape(nr, n_sub * sub_len)


def _expand_q5k_f32(qp, qh, sci, mni, dh, dmh, n_sub, sub_len):
    nr = qp.shape[0]
    qp3 = qp.reshape(nr, -1, 32)                 # 32-byte chunks of 64 elems
    nch = qp3.shape[1]
    qh3 = qh.reshape(nr, -1, 32)                 # 32 fifth-bit bytes / block
    codes = _np.empty((nr, nch, 64), dtype=_np.uint8)
    for c4 in range(4):                          # chunk index within a block
        lo = ((qh3 >> (2 * c4)) & 1) << 4
        hi = ((qh3 >> (2 * c4 + 1)) & 1) << 4
        codes[:, c4::4, :32] = (qp3[:, c4::4] & 0x0F) | lo
        codes[:, c4::4, 32:] = (qp3[:, c4::4] >> 4) | hi
    d_eff = (dh.view(_np.float16).astype(_np.float32)[:, :, None]
             * sci.astype(_np.float32))
    m_eff = (dmh.view(_np.float16).astype(_np.float32)[:, :, None]
             * mni.astype(_np.float32))
    v = codes.reshape(nr, n_sub, sub_len).astype(_np.float32)
    v *= d_eff.reshape(nr, n_sub, 1)
    v -= m_eff.reshape(nr, n_sub, 1)
    return v.reshape(nr, n_sub * sub_len)


def _expand_q6k_f32(q3, sci, dh):
    nr = q3.shape[0]
    d_eff = (dh.view(_np.float16).astype(_np.float32).repeat(16, axis=1)
             * sci.astype(_np.float32))
    v = q3.astype(_np.float32)
    v *= d_eff[:, :, None]
    return v.reshape(nr, -1)


# ---- optional hot-cache budget (module state) ------------------------------

def _hot_cache_limit_bytes() -> int:
    if not HAS_NUMPY:
        return 0
    raw = os.environ.get(_HOT_WEIGHT_ENV)
    if raw is None or not raw.strip():
        return 0
    try:
        mb = float(raw.strip())
    except ValueError:
        return 0
    if not math.isfinite(mb) or mb <= 0.0:
        return 0
    try:
        return int(mb * 1024 * 1024)
    except (OverflowError, ValueError):
        return 0


def _clear_hot_caches() -> None:
    global _HOT_CACHE_USED_BYTES, _HOT_CACHE_MATRICES
    for owner in list(_HOT_CACHE_OWNERS):
        owner._dense_cache = None
        owner._dense_cache_bytes = 0
    try:
        _HOT_CACHE_OWNERS.clear()
    except Exception:
        pass
    _HOT_CACHE_USED_BYTES = 0
    _HOT_CACHE_MATRICES = 0


def _sync_hot_cache_budget(limit: int) -> None:
    global _HOT_CACHE_LIMIT_BYTES
    if _HOT_CACHE_LIMIT_BYTES != limit:
        _clear_hot_caches()
        _HOT_CACHE_LIMIT_BYTES = limit


def _reserve_hot_cache_bytes(nbytes: int) -> bool:
    global _HOT_CACHE_USED_BYTES, _HOT_CACHE_MATRICES
    limit = _hot_cache_limit_bytes()
    _sync_hot_cache_budget(limit)
    if limit <= 0 or nbytes > limit - _HOT_CACHE_USED_BYTES:
        return False
    _HOT_CACHE_USED_BYTES += nbytes
    _HOT_CACHE_MATRICES += 1
    return True


def _release_hot_cache_bytes(nbytes: int) -> None:
    global _HOT_CACHE_USED_BYTES, _HOT_CACHE_MATRICES
    _HOT_CACHE_USED_BYTES = max(0, _HOT_CACHE_USED_BYTES - nbytes)
    _HOT_CACHE_MATRICES = max(0, _HOT_CACHE_MATRICES - 1)


def _reset_hot_cache_state() -> None:
    global _HOT_CACHE_LIMIT_BYTES
    _clear_hot_caches()
    _HOT_CACHE_LIMIT_BYTES = None


def hot_cache_stats() -> dict[str, int]:
    limit = _hot_cache_limit_bytes()
    _sync_hot_cache_budget(limit)
    return {
        "limit_bytes": limit,
        "used_bytes": _HOT_CACHE_USED_BYTES,
        "matrices": _HOT_CACHE_MATRICES,
    }


def can_quantized_matvec(dtype: str, cols: int) -> bool:
    geom = QUANT_GEOMETRY.get(dtype)
    return geom is not None and cols % geom[0] == 0


__all__ = ["QuantMatrix", "QUANTIZED_MATVEC_DTYPES", "can_quantized_matvec",
           "hot_cache_stats"]
