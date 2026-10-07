# Alpaccaroo - our own GPU tier: quantized matvec/matmul CUDA kernels,
# written in Python. MIT License. See LICENSE.
"""Alpaccaroo's GPU tier: Python-source CUDA kernels, optionally JIT-compiled.

Weight matrices stay quantized in VRAM in the same universal codes layout
the NumPy backend unpacks (int8 codes + per-sub-block effective float32
scales/offsets, :func:`alpaccaroo.quants.np_unpack`). Every kernel here is
Alpaccaroo's own algorithm, authored in this file as ordinary Python and
compiled at runtime by the OPTIONAL, PINNED numba-cuda JIT
(``pip install alpaccaroo[gpu]``). Without a CUDA device or the pinned JIT -
or with ``ALPACCAROO_GPU=0`` - nothing changes: the CPU tiers (kernels/NumPy/
pure) remain the reference implementations and the fallback, and any error
in this tier degrades to them rather than failing the engine.

Placement: weight matrices move to VRAM at load, and only whole matrices -
per-matrix dispatch makes mixed GPU/CPU placement exact, so running out of
VRAM mid-load just leaves the rest on the CPU tiers. The host KV cache
stays authoritative (prefill's prefix reuse and truncation are untouched).
On top of the v1 per-matvec/GEMM dispatch this file adds two stage-2
paths; both degrade to the exact v1/CPU behavior on any failure:

- attention_batch: prefill's batched causal GQA - the O(n^2) share that
  made long prompts degrade with length - runs on the device with an
  online softmax. The per-layer K/V slice re-uploads from the host cache
  every chunk (measured tax in the wrapper docstring), far cheaper than
  the einsum it replaces.
- chain_forward: when EVERY chain matrix of a llama-class model is
  resident, whole-token decode runs device-side (rmsnorm, rope, GQA over
  a device K/V mirror, silu*up, residual adds) with ONE synchronization -
  the logits download - per token, instead of a sync D2H per matvec. The
  mirror uploads once after prefill; each decoded token's new K/V row is
  written on the device and copied back inside the same sync, so the host
  cache remains the source of truth, and Model._chain_invalidate re-syncs
  exactly the rows any host path rewrites.
- chain_prefill: whole prefill chunks on the same chain. The per-chunk
  host glue the stage-2 paths still paid - numpy batch rope, rmsnorm,
  residual adds, and an H2D/D2H ping-pong around every GEMM and the
  attention (measured 10.6 s of the 8B's 17.6 s 4096-token prefill) -
  moves into batch kernels chained device-side: one embedding upload per
  chunk, K/V rows written straight into the decode mirror (which the
  batched attention then READS, retiring the re-upload tax above), and
  one synchronous copy-back of the chunk's K/V rows to the authoritative
  host cache. Same gate, same fallback: any failure recomputes the chunk
  on the exact existing path.

The token embedding never uploads: it is a row-gather workload, the one
shape these kernels are wrong for, and it must stay gatherable on the host.

Pin policy: numba-cuda is locked to ``NUMBA_CUDA_PIN`` below, mirroring
the kernels tier. The pin is never updated implicitly; a different
installed version deactivates the tier (``ALPACCAROO_GPU=force`` overrides at
your own risk).

Known trap, recorded because the failure is silent and misattributed: with
``nvidia-nvjitlink`` missing, cuda.core's DLL search can find a torch
wheel's bundled nvJitLink_120_0.dll instead, and every kernel then dies
with nvJitLinkError ERROR_OUTDATED_LIBRARY(14). The availability probe
below therefore compiles and RUNS a real kernel, catches Exception
broadly, and reports the cause through status() instead of activating a
tier that cannot launch.

Buffer reuse contract: the per-size pinned/device activation buffers (and
the grow-only batched-matmul staging pair) are not synchronized for
concurrent callers - alpaccaroo's own server serializes generation behind a
lock, same caveat as the hot-weight cache in :mod:`alpaccaroo.qmatrix`.
Every matvec and every batched matmul ends with a synchronous D2H on the
legacy default stream, which drains the queued async H2D and kernel, so
rewriting the staging buffers on the next call is safe.
"""

from __future__ import annotations

import math
import os
import sys

from .quants import QUANT_GEOMETRY, np_unpack

NUMBA_CUDA_PIN = "0.30.4"

# VRAM headroom never spent on weights: the WDDM display path, the CUDA
# context and the JIT each claim device memory after load, and exhausting
# the last MiB turns into unrelated launch failures much later.
GPU_RESERVE_MB = 512

# Threads per block for the row-per-block matvec; also the shared-memory
# reduction width, so it must stay a power of two.
_TPB = 256

# Tiled-GEMM block geometry for the batched matmul (prefill path). One
# thread block computes a (_GEMM_ROWS x _GEMM_BT) output tile: threadIdx.x
# walks rows (coalesced code loads and stores), threadIdx.y walks batch
# lanes, and each thread keeps a 4-row x 2-batch-lane micro-tile in
# registers. Both block shapes measured on the 8B layer set (RTX 4090,
# batch 256, kernel time summed over wgu+wdown+wo+wqk+wv, best of 5):
# 16x16 threads (64 rows x 32 batch tile) 12.00 ms, 32x8 threads (128
# rows x 16 batch tile) 14.60 ms - the codes stream re-reads VRAM once
# per batch tile, so the shape with the wider batch tile reads the
# dominant stream half as often. 16x16 kept (10.0 TFLOP/s on the fused
# ffn matrix).
# This one kernel replaced the old wide-kernel/looped-matvec pair and the
# ALPACCAROO_GPU_WIDE_MATMUL_ELEMS crossover knob that picked between them:
# decoding each code tile into shared memory once removes the redundant
# per-batch-lane unpack that made both old paths lose, at every size, so
# the knob (and the env var, now silently ignored) had nothing left to
# choose. Column tiles stage through shared memory padded by one float so
# neither the batch-lane reads nor the row reads bank-conflict.
_GEMM_TX = 16                     # row lanes per block (threadIdx.x)
_GEMM_TY = 16                     # batch lanes per block (threadIdx.y)
_GEMM_ROWS = 4 * _GEMM_TX         # output rows per block (4 row regs/thread)
_GEMM_BT = 2 * _GEMM_TY           # output batch per block (2 lanes/thread)
_GEMM_COLS = 64                   # staged column tile, float32 elements
_GEMM_PAD = _GEMM_COLS + 1        # +1: shared.array shapes must be plain
#                                   module constants (inline arithmetic in
#                                   the shape tuple fails numba's typing)
_GEMM_WORDS = _GEMM_COLS // 4     # same tile in packed int32 code words
_GEMM_TPB = _GEMM_TX * _GEMM_TY

# Batched-attention block geometry (prefill path). One block owns _ATT_R
# (token, q-head) rows that all attend through one kv head and streams K/V
# tiles of _ATT_TS positions through shared memory with an online
# (rescaling) softmax, so no (t_q, t_kv) score matrix ever exists and any
# context length fits. The layout requires _ATT_TPB == _ATT_R * _ATT_TS/2:
# the score phase gives each thread two tile columns of one row (3 shared
# loads feed 2 FMAs), the accumulate phase two rows of one dim column
# (same ratio). Head widths above _ATT_HD (the Gemma family's 256) stay on
# the NumPy path: the shared tiles are sized at compile time and four
# 16x128-float tiles already fill ~35 KiB of the 48 KiB block budget.
_ATT_HD = 128                 # widest head these tiles serve
_ATT_HDP = _ATT_HD + 4        # padded stride keeps row and column reads
#                               off the same shared-memory banks
_ATT_R = 16                   # (token, q-head) rows per block
_ATT_RH = _ATT_R // 2         # dual-row micro-tile, accumulate phase
_ATT_TS = 16                  # K/V tile length along the sequence
_ATT_TPB = 128                # _ATT_R * _ATT_TS // 2, see above

# Decode-attention geometry (device chain): one block per (q head,
# _DEC_SPLIT-position chunk of the context), each producing an online-
# softmax partial (m, l, unnormalized acc) that a tiny combine kernel
# merges per head. The split exists because one-block-per-head leaves
# the card idle at depth: 32 blocks reading the 8B's K/V rows measured
# 26.1 tok/s at a 4400-token context, the split form 53.0 - a bandwidth-
# bound scan needs blocks proportional to the context, not the head
# count. Only the score tile lives in shared memory; K/V rows are read
# straight from the mirror, contiguous in both phases (per-thread in the
# score phase, per-position across threads in the accumulate phase).
_DEC_TS = 512
_DEC_SPLIT = 512              # context rows per partial block
_DEC_TPB = 128

_state: dict | None = None  # lazy: {"ok": True, ...} or {} / {"error"} off


def _vram_cap_bytes() -> int:
    """ALPACCAROO_GPU_VRAM_MB caps total uploaded weight bytes - the test
    hook for the mixed-placement fallback; unset/0 lets free VRAM minus
    GPU_RESERVE_MB decide."""
    raw = os.environ.get("ALPACCAROO_GPU_VRAM_MB", "")
    if not raw.strip():
        return 0
    try:
        mb = float(raw)
    except ValueError:
        return 0
    if not math.isfinite(mb) or mb <= 0.0:
        return 0
    try:
        return int(mb * 1024 * 1024)
    except (OverflowError, ValueError):
        return 0


def _init() -> dict:
    global _state
    if _state is not None:
        return _state
    _state = {}
    mode = os.environ.get("ALPACCAROO_GPU", "").strip().lower()
    if mode in ("0", "off", "no") or os.environ.get("ALPACCAROO_PURE"):
        return _state
    try:
        import numpy as np
        import numba_cuda
        from numba import cuda, float32, int32
        from cuda.bindings import runtime as rt
    except Exception:
        return _state
    try:
        # a 3-row fixture matrix legitimately launches a 3-block grid; the
        # per-launch "Grid size N will likely result in GPU
        # under-utilization" advice is noise at that scale, and only
        # there. numba-cuda VENDORS its warning class (numba.cuda.core.
        # errors, not numba.core.errors - verified distinct classes), so
        # the filter must name the vendored one to have any effect.
        import warnings
        from numba.cuda.core.errors import NumbaPerformanceWarning
        warnings.filterwarnings("ignore", message=".*Grid size.*",
                                category=NumbaPerformanceWarning)
    except Exception:
        pass
    if numba_cuda.__version__ != NUMBA_CUDA_PIN and mode != "force":
        print(f"alpaccaroo: numba-cuda {numba_cuda.__version__} != pinned "
              f"{NUMBA_CUDA_PIN}; gpu tier disabled (ALPACCAROO_GPU=force to "
              f"override)", file=sys.stderr)
        return _state

    try:
        if not cuda.is_available():
            return _state
        dev = cuda.get_current_device()
        name = (dev.name.decode() if isinstance(dev.name, bytes)
                else str(dev.name))
        err, free_b, total_b = rt.cudaMemGetInfo()
        if int(err):
            raise RuntimeError(f"cudaMemGetInfo: {err}")

        # The probe compiles AND runs before the tier may activate: import
        # succeeds even when the JIT-link stack is broken (see the
        # nvJitLink trap in the module docstring), so only a kernel that
        # produced correct output proves the tier can work.
        @cuda.jit(cache=True)
        def _probe(a, b):
            i = cuda.grid(1)
            if i < a.size:
                b[i] = a[i] * float32(2.0) + b[i]

        pa = np.arange(32, dtype=np.float32)
        pb = np.ones(32, dtype=np.float32)
        d_a, d_b = cuda.to_device(pa), cuda.to_device(pb)
        _probe[1, 32](d_a, d_b)
        got = d_b.copy_to_host()
        if float(abs(got[31] - 63.0)) != 0.0:
            raise RuntimeError("probe kernel returned wrong values")
    except Exception as e:
        # broad on purpose: report the real cause (missing driver, stale
        # nvJitLink, exhausted device...) through status() instead of
        # crashing whoever asked whether a GPU exists
        _state = {"error": f"{type(e).__name__}: {e}"}
        return _state

    @cuda.jit(fastmath=True, cache=True)
    def _matvec_codes(qw, d, m, x, out, shift, affine):
        # out[r] = sum_c (d[r, c>>shift] * code + m[...]) * x[c]: the same
        # per-element algebra as the effective scales np_unpack feeds every
        # CPU path. One block per row; qw is the int8 codes viewed as
        # int32, so each thread pulls four codes per load and a warp moves
        # 128 contiguous bytes - 734 GB/s measured vs 500 for byte loads.
        sh = cuda.shared.array(_TPB, float32)
        r = cuda.blockIdx.x
        t = cuda.threadIdx.x
        cw = qw.shape[1]
        acc = float32(0.0)
        c = t
        while c < cw:
            v = qw[r, c]
            c0 = c << 2
            s = c0 >> shift  # 4-byte groups never straddle a sub-block
            b0 = float32(((v & int32(0xFF)) ^ int32(0x80)) - int32(0x80))
            b1 = float32((((v >> 8) & int32(0xFF)) ^ int32(0x80))
                         - int32(0x80))
            b2 = float32((((v >> 16) & int32(0xFF)) ^ int32(0x80))
                         - int32(0x80))
            b3 = float32((((v >> 24) & int32(0xFF)) ^ int32(0x80))
                         - int32(0x80))
            ds = d[r, s]
            if affine:
                ms = m[r, s]
                acc += (ds * b0 + ms) * x[c0]
                acc += (ds * b1 + ms) * x[c0 + 1]
                acc += (ds * b2 + ms) * x[c0 + 2]
                acc += (ds * b3 + ms) * x[c0 + 3]
            else:
                acc += ds * (b0 * x[c0] + b1 * x[c0 + 1]
                             + b2 * x[c0 + 2] + b3 * x[c0 + 3])
            c += _TPB
        sh[t] = acc
        cuda.syncthreads()
        i = _TPB // 2
        while i > 0:
            if t < i:
                sh[t] += sh[t + i]
            cuda.syncthreads()
            i >>= 1
        if t == 0:
            out[r] = sh[0]

    @cuda.jit(fastmath=True, cache=True)
    def _matmul_codes_gemm(qw, d, m, X, out, shift, affine):
        # out[b, r] = sum_c (d[r, c>>shift] * code + m[...]) * X[b, c]:
        # one launch for ANY matrix and batch size. Classic shared-memory
        # tiling, with the twist that the staged weight tile is DECODED
        # once per block - each int32 word unpacks its four codes and
        # applies the effective scale (and offset: folding m per element
        # is the exact affine algebra, so no separate x-sub-sums term)
        # straight into shared memory. The old wide kernel re-decoded
        # every code once per batch lane; here the unpack cost is divided
        # by the batch tile and the inner loop is pure f32 FMA on shared
        # memory. Ragged edges (rows, batch, cols not multiples of the
        # tile) zero-fill the staging tiles, which contribute nothing.
        Xs = cuda.shared.array((_GEMM_BT, _GEMM_PAD), float32)
        Ws = cuda.shared.array((_GEMM_ROWS, _GEMM_PAD), float32)
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        tid = ty * _GEMM_TX + tx
        batch = out.shape[0]
        rows = out.shape[1]
        cols = X.shape[1]
        cw = qw.shape[1]
        r0 = cuda.blockIdx.x * _GEMM_ROWS
        b0 = cuda.blockIdx.y * _GEMM_BT
        acc0a = float32(0.0)
        acc1a = float32(0.0)
        acc2a = float32(0.0)
        acc3a = float32(0.0)
        acc0b = float32(0.0)
        acc1b = float32(0.0)
        acc2b = float32(0.0)
        acc3b = float32(0.0)
        w0 = 0
        while w0 < cw:
            # stage the activation tile: consecutive threads read
            # consecutive columns of one X row - coalesced
            i = tid
            while i < _GEMM_BT * _GEMM_COLS:
                bi = i // _GEMM_COLS
                ci = i - bi * _GEMM_COLS
                b = b0 + bi
                c = (w0 << 2) + ci
                v = float32(0.0)
                if b < batch and c < cols:
                    v = X[b, c]
                Xs[bi, ci] = v
                i += _GEMM_TPB
            # decode the weight tile: consecutive threads read
            # consecutive int32 words of one code row - coalesced, four
            # codes per load (the matvec kernel's trick), decoded once
            # per block instead of once per batch lane
            i = tid
            while i < _GEMM_ROWS * _GEMM_WORDS:
                ri = i // _GEMM_WORDS
                wi = i - ri * _GEMM_WORDS
                r = r0 + ri
                w = w0 + wi
                ci = wi << 2
                if r < rows and w < cw:
                    v = qw[r, w]
                    s = (w << 2) >> shift  # words never straddle a sub-block
                    ds = d[r, s]
                    e0 = float32(((v & int32(0xFF)) ^ int32(0x80))
                                 - int32(0x80))
                    e1 = float32((((v >> 8) & int32(0xFF)) ^ int32(0x80))
                                 - int32(0x80))
                    e2 = float32((((v >> 16) & int32(0xFF)) ^ int32(0x80))
                                 - int32(0x80))
                    e3 = float32((((v >> 24) & int32(0xFF)) ^ int32(0x80))
                                 - int32(0x80))
                    if affine:
                        ms = m[r, s]
                        Ws[ri, ci] = ds * e0 + ms
                        Ws[ri, ci + 1] = ds * e1 + ms
                        Ws[ri, ci + 2] = ds * e2 + ms
                        Ws[ri, ci + 3] = ds * e3 + ms
                    else:
                        Ws[ri, ci] = ds * e0
                        Ws[ri, ci + 1] = ds * e1
                        Ws[ri, ci + 2] = ds * e2
                        Ws[ri, ci + 3] = ds * e3
                else:
                    Ws[ri, ci] = float32(0.0)
                    Ws[ri, ci + 1] = float32(0.0)
                    Ws[ri, ci + 2] = float32(0.0)
                    Ws[ri, ci + 3] = float32(0.0)
                i += _GEMM_TPB
            cuda.syncthreads()
            # 4 rows x 2 batch lanes per thread: 6 shared loads feed 8
            # FMAs, vs 5-for-4 with a single lane - the inner loop is
            # shared-bandwidth-bound, so fewer loads per FMA is the win
            for ci in range(_GEMM_COLS):
                xa = Xs[ty, ci]
                xb = Xs[ty + _GEMM_TY, ci]
                wv = Ws[tx, ci]
                acc0a += wv * xa
                acc0b += wv * xb
                wv = Ws[tx + _GEMM_TX, ci]
                acc1a += wv * xa
                acc1b += wv * xb
                wv = Ws[tx + 2 * _GEMM_TX, ci]
                acc2a += wv * xa
                acc2b += wv * xb
                wv = Ws[tx + 3 * _GEMM_TX, ci]
                acc3a += wv * xa
                acc3b += wv * xb
            cuda.syncthreads()
            w0 += _GEMM_WORDS
        b = b0 + ty
        if b < batch:
            r = r0 + tx
            if r < rows:
                out[b, r] = acc0a
            r += _GEMM_TX
            if r < rows:
                out[b, r] = acc1a
            r += _GEMM_TX
            if r < rows:
                out[b, r] = acc2a
            r += _GEMM_TX
            if r < rows:
                out[b, r] = acc3a
        b += _GEMM_TY
        if b < batch:
            r = r0 + tx
            if r < rows:
                out[b, r] = acc0b
            r += _GEMM_TX
            if r < rows:
                out[b, r] = acc1b
            r += _GEMM_TX
            if r < rows:
                out[b, r] = acc2b
            r += _GEMM_TX
            if r < rows:
                out[b, r] = acc3b

    @cuda.jit(fastmath=True, cache=True)
    def _attention_batch_gqa(Q, K, V, positions, out, group, inv_sqrt,
                             window, kv_start):
        # Batched causal grouped-query attention: the exact math of
        # model._attention_batch_np / _batch_window_np, with an online
        # softmax so no (t_q, t_kv) score matrix is ever materialized.
        # Per K/V tile each row keeps a running max m and weight sum l,
        # rescales its partial output by exp(m_old - m_new), and masked
        # entries carry weight exactly 0.0 - the same weight the
        # reference's -1e30 fill produces once its max-subtracted exp
        # underflows. -3.4e38 is the mask sentinel; real scores sit tens
        # of orders of magnitude above it. blockIdx.y picks the kv head,
        # blockIdx.x a tile of _ATT_R (token, q-head) rows attending
        # through it, so every staged K/V tile is reused _ATT_R times.
        Qs = cuda.shared.array((_ATT_R, _ATT_HDP), float32)
        Ks = cuda.shared.array((_ATT_TS, _ATT_HDP), float32)
        Vs = cuda.shared.array((_ATT_TS, _ATT_HDP), float32)
        Os = cuda.shared.array((_ATT_R, _ATT_HDP), float32)
        Ss = cuda.shared.array((_ATT_R, _ATT_TS), float32)
        mrow = cuda.shared.array(_ATT_R, float32)
        lrow = cuda.shared.array(_ATT_R, float32)
        arow = cuda.shared.array(_ATT_R, float32)
        prow = cuda.shared.array(_ATT_R, int32)
        pmax = cuda.shared.array(1, int32)
        tid = cuda.threadIdx.x
        kh = cuda.blockIdx.y
        r0 = cuda.blockIdx.x * _ATT_R
        t_q = Q.shape[0]
        hd = Q.shape[2]
        t_kv = K.shape[0]
        nrows = t_q * group
        scale = float32(inv_sqrt)

        i = tid
        while i < _ATT_R * hd:
            j = i // hd
            dd = i - j * hd
            gr = r0 + j
            qv = float32(0.0)
            if gr < nrows:
                ti = gr // group
                qv = Q[ti, kh * group + (gr - ti * group), dd]
            Qs[j, dd] = qv
            Os[j, dd] = float32(0.0)
            i += _ATT_TPB
        if tid < _ATT_R:
            gr = r0 + tid
            p = -1
            if gr < nrows:
                p = positions[gr // group]
            prow[tid] = p
            mrow[tid] = float32(-3.4e38)
            lrow[tid] = float32(0.0)
        cuda.syncthreads()
        if tid == 0:
            pm = -1
            for j in range(_ATT_R):
                if prow[j] > pm:
                    pm = prow[j]
            pmax[0] = pm
        cuda.syncthreads()

        s0 = 0
        while s0 < t_kv:
            if kv_start + s0 > pmax[0]:
                break  # causal: nothing right of the newest row's position
            i = tid
            while i < _ATT_TS * hd:
                ss = i // hd
                dd = i - ss * hd
                s = s0 + ss
                kv = float32(0.0)
                vv = float32(0.0)
                if s < t_kv:
                    kv = K[s, kh, dd]
                    vv = V[s, kh, dd]
                Ks[ss, dd] = kv
                Vs[ss, dd] = vv
                i += _ATT_TPB
            cuda.syncthreads()
            # score phase: two tile columns per thread, one shared q-row
            # load feeding both dots
            j = tid >> 3
            c0 = tid & 7
            acc0 = float32(0.0)
            acc1 = float32(0.0)
            for dd in range(hd):
                qv = Qs[j, dd]
                acc0 += qv * Ks[c0, dd]
                acc1 += qv * Ks[c0 + 8, dd]
            pj = prow[j]
            sc0 = float32(-3.4e38)
            sc1 = float32(-3.4e38)
            sa = kv_start + s0 + c0
            if pj >= 0 and s0 + c0 < t_kv and sa <= pj and \
                    (window <= 0 or sa > pj - window):
                sc0 = acc0 * scale
            sa = kv_start + s0 + c0 + 8
            if pj >= 0 and s0 + c0 + 8 < t_kv and sa <= pj and \
                    (window <= 0 or sa > pj - window):
                sc1 = acc1 * scale
            Ss[j, c0] = sc0
            Ss[j, c0 + 8] = sc1
            cuda.syncthreads()
            if tid < _ATT_R:
                m_old = mrow[tid]
                mt = m_old
                for ss in range(_ATT_TS):
                    if Ss[tid, ss] > mt:
                        mt = Ss[tid, ss]
                # all-masked tile: mt stays m_old, alpha is exp(0) = 1
                # and every weight below is 0 - an exact no-op
                alpha = math.exp(m_old - mt)
                psum = float32(0.0)
                for ss in range(_ATT_TS):
                    p2 = float32(0.0)
                    if Ss[tid, ss] > float32(-1.0e37):
                        p2 = math.exp(Ss[tid, ss] - mt)
                    Ss[tid, ss] = p2
                    psum += p2
                mrow[tid] = mt
                lrow[tid] = lrow[tid] * alpha + psum
                arow[tid] = alpha
            cuda.syncthreads()
            # accumulate phase: two rows per thread, one shared V load
            # feeding both FMAs
            i = tid
            while i < _ATT_RH * hd:
                j0 = i // hd
                dd = i - j0 * hd
                j1 = j0 + _ATT_RH
                a0 = Os[j0, dd] * arow[j0]
                a1 = Os[j1, dd] * arow[j1]
                for ss in range(_ATT_TS):
                    vv = Vs[ss, dd]
                    a0 += Ss[j0, ss] * vv
                    a1 += Ss[j1, ss] * vv
                Os[j0, dd] = a0
                Os[j1, dd] = a1
                i += _ATT_TPB
            cuda.syncthreads()
            s0 += _ATT_TS

        i = tid
        while i < _ATT_R * hd:
            j = i // hd
            dd = i - j * hd
            gr = r0 + j
            if gr < nrows:
                ov = float32(0.0)
                if lrow[j] > float32(0.0):
                    ov = Os[j, dd] / lrow[j]
                ti = gr // group
                out[ti, kh * group + (gr - ti * group), dd] = ov
            i += _ATT_TPB

    @cuda.jit(fastmath=True, cache=True)
    def _attention_decode_part(q, Kl, Vl, pm, pl, pacc, t, group, inv_sqrt):
        # Single-token GQA over the device K/V mirror, one online-softmax
        # PARTIAL per block: blockIdx.x is the q head, blockIdx.y a
        # _DEC_SPLIT-row chunk of the context (see the geometry comment).
        # Score phase: each thread dots q (shared) against whole K rows -
        # per-thread contiguous reads. Accumulate phase: threads own
        # output dims, so each tile position is one coalesced V-row read.
        # No mask: decode attends to every cached position, exactly like
        # the CPU decode attention. Emits (m, l, unnormalized acc).
        Ss = cuda.shared.array(_DEC_TS, float32)
        red = cuda.shared.array(_DEC_TPB, float32)
        qs = cuda.shared.array(_ATT_HD, float32)
        tid = cuda.threadIdx.x
        hh = cuda.blockIdx.x
        hd = Kl.shape[2]
        kvh = hh // group
        scale = float32(inv_sqrt)
        c0 = cuda.blockIdx.y * _DEC_SPLIT
        c1 = t
        if c1 > c0 + _DEC_SPLIT:
            c1 = c0 + _DEC_SPLIT
        i = tid
        while i < hd:
            qs[i] = q[hh * hd + i]
            i += _DEC_TPB
        cuda.syncthreads()
        m = float32(-3.4e38)
        l = float32(0.0)
        acc = float32(0.0)
        t0 = c0
        while t0 < c1:
            ts_len = c1 - t0
            if ts_len > _DEC_TS:
                ts_len = _DEC_TS
            tt = tid
            while tt < ts_len:
                dot = float32(0.0)
                for dd in range(hd):
                    dot += qs[dd] * Kl[t0 + tt, kvh, dd]
                Ss[tt] = dot * scale
                tt += _DEC_TPB
            cuda.syncthreads()
            mt = float32(-3.4e38)
            tt = tid
            while tt < ts_len:
                if Ss[tt] > mt:
                    mt = Ss[tt]
                tt += _DEC_TPB
            red[tid] = mt
            cuda.syncthreads()
            i = _DEC_TPB // 2
            while i > 0:
                if tid < i and red[tid + i] > red[tid]:
                    red[tid] = red[tid + i]
                cuda.syncthreads()
                i >>= 1
            mt = red[0]
            m_new = m
            if mt > m_new:
                m_new = mt
            alpha = math.exp(m - m_new)
            cuda.syncthreads()  # everyone has read red[0]; reuse for sums
            psum = float32(0.0)
            tt = tid
            while tt < ts_len:
                p = math.exp(Ss[tt] - m_new)
                Ss[tt] = p
                psum += p
                tt += _DEC_TPB
            red[tid] = psum
            cuda.syncthreads()
            i = _DEC_TPB // 2
            while i > 0:
                if tid < i:
                    red[tid] += red[tid + i]
                cuda.syncthreads()
                i >>= 1
            l = l * alpha + red[0]
            m = m_new
            if tid < hd:
                a = acc * alpha
                for tt in range(ts_len):
                    a += Ss[tt] * Vl[t0 + tt, kvh, tid]
                acc = a
            t0 += _DEC_TS
            cuda.syncthreads()  # Ss and red are rewritten by the next tile
        ck = cuda.blockIdx.y
        if tid == 0:
            pm[hh, ck] = m
            pl[hh, ck] = l
        if tid < hd:
            pacc[hh, ck, tid] = acc

    @cuda.jit(fastmath=True, cache=True)
    def _attention_decode_combine(pm, pl, pacc, out, n_chunks):
        # merge one head's chunk partials: the launch grid guarantees
        # every chunk saw real rows, so every pm entry is a real max
        hh = cuda.blockIdx.x
        tid = cuda.threadIdx.x
        hd = pacc.shape[2]
        M = float32(-3.4e38)
        for c in range(n_chunks):
            if pm[hh, c] > M:
                M = pm[hh, c]
        L = float32(0.0)
        for c in range(n_chunks):
            L += pl[hh, c] * math.exp(pm[hh, c] - M)
        d = tid
        while d < hd:
            a = float32(0.0)
            for c in range(n_chunks):
                a += pacc[hh, c, d] * math.exp(pm[hh, c] - M)
            out[hh * hd + d] = a / L
            d += _DEC_TPB

    @cuda.jit(fastmath=True, cache=True)
    def _rmsnorm_dev(x, w, out, eps):
        # tensor.rmsnorm's decode algebra: out = x / sqrt(mean(x^2) +
        # eps) * w. One block; the tree reduction is deterministic.
        sh = cuda.shared.array(_TPB, float32)
        t = cuda.threadIdx.x
        n = x.shape[0]
        acc = float32(0.0)
        i = t
        while i < n:
            v = x[i]
            acc += v * v
            i += _TPB
        sh[t] = acc
        cuda.syncthreads()
        i = _TPB // 2
        while i > 0:
            if t < i:
                sh[t] += sh[t + i]
            cuda.syncthreads()
            i >>= 1
        inv = float32(1.0) / math.sqrt(sh[0] / float32(n) + float32(eps))
        i = t
        while i < n:
            out[i] = x[i] * inv * w[i]
            i += _TPB

    @cuda.jit(fastmath=True, cache=True)
    def _rope_store_decode(qk, v, cos_tab, sin_tab, bqk, bv, has_bqk,
                           has_bv, n_head, n_rot, neox, Kl, Vl, pos):
        # One launch finishes the whole qkv stage of a decoded token:
        # bias adds, both rope styles on q and k, and the new K/V rows
        # written straight into the device mirror at their absolute
        # position. Thread jobs: rotation pairs for the n_head q heads
        # then the n_kv k heads, pass-through dims (n_rot < hd), then the
        # v row. q rotates in place in the qk buffer; the callers pass qk
        # and v themselves as the bias placeholders when a model has no
        # biases, so the flags alone decide.
        i = cuda.grid(1)
        hd = Kl.shape[2]
        n_kv = Kl.shape[1]
        half = n_rot >> 1
        nheads = n_head + n_kv
        npair = nheads * half
        ntail = nheads * (hd - n_rot)
        if i < npair:
            hh = i // half
            j = i - hh * half
            base = hh * hd
            if neox:
                a = base + j
                b = base + half + j
            else:
                a = base + 2 * j
                b = base + 2 * j + 1
            x0 = qk[a]
            x1 = qk[b]
            if has_bqk:
                x0 += bqk[a]
                x1 += bqk[b]
            c = cos_tab[pos, j]
            s = sin_tab[pos, j]
            y0 = x0 * c - x1 * s
            y1 = x0 * s + x1 * c
            if hh < n_head:
                qk[a] = y0
                qk[b] = y1
            else:
                Kl[pos, hh - n_head, a - base] = y0
                Kl[pos, hh - n_head, b - base] = y1
        elif i < npair + ntail:
            k = i - npair
            w2 = hd - n_rot
            hh = k // w2
            dd = n_rot + (k - hh * w2)
            idx = hh * hd + dd
            x0 = qk[idx]
            if has_bqk:
                x0 += bqk[idx]
            if hh < n_head:
                qk[idx] = x0
            else:
                Kl[pos, hh - n_head, dd] = x0
        elif i < npair + ntail + n_kv * hd:
            k = i - npair - ntail
            kvh = k // hd
            dd = k - kvh * hd
            x0 = v[k]
            if has_bv:
                x0 += bv[k]
            Vl[pos, kvh, dd] = x0

    @cuda.jit(fastmath=True, cache=True)
    def _silu_mul_dev(gu, out):
        # act = gate / (1 + exp(-gate)) * up over the fused gate|up
        # buffer - the llama-class MLP's exact CPU algebra
        n_ff = out.shape[0]
        i = cuda.grid(1)
        if i < n_ff:
            g = gu[i]
            out[i] = g / (float32(1.0) + math.exp(-g)) * gu[n_ff + i]

    @cuda.jit(fastmath=True, cache=True)
    def _silu_mul_batch(gu, out):
        # the same algebra over a (batch, 2*n_ff) fused-GEMM result;
        # sits between the two prefill GEMMs of ffn_swiglu_batch so the
        # 3x-larger gate|up block never travels to the host
        n_ff = out.shape[1]
        n = out.shape[0] * n_ff
        i = cuda.grid(1)
        if i < n:
            b = i // n_ff
            j = i - b * n_ff
            g = gu[b, j]
            out[b, j] = g / (float32(1.0) + math.exp(-g)) * gu[b, n_ff + j]

    @cuda.jit(fastmath=True, cache=True)
    def _matvec_codes_res(qw, d, m, x, res, out, shift, affine):
        # _matvec_codes with the residual add fused: out[r] = res[r] +
        # row dot. res and out may alias (the chain adds in place); each
        # row's slot is read once, by the thread that then writes it.
        sh = cuda.shared.array(_TPB, float32)
        r = cuda.blockIdx.x
        t = cuda.threadIdx.x
        cw = qw.shape[1]
        acc = float32(0.0)
        c = t
        while c < cw:
            v = qw[r, c]
            c0 = c << 2
            s = c0 >> shift
            b0 = float32(((v & int32(0xFF)) ^ int32(0x80)) - int32(0x80))
            b1 = float32((((v >> 8) & int32(0xFF)) ^ int32(0x80))
                         - int32(0x80))
            b2 = float32((((v >> 16) & int32(0xFF)) ^ int32(0x80))
                         - int32(0x80))
            b3 = float32((((v >> 24) & int32(0xFF)) ^ int32(0x80))
                         - int32(0x80))
            ds = d[r, s]
            if affine:
                ms = m[r, s]
                acc += (ds * b0 + ms) * x[c0]
                acc += (ds * b1 + ms) * x[c0 + 1]
                acc += (ds * b2 + ms) * x[c0 + 2]
                acc += (ds * b3 + ms) * x[c0 + 3]
            else:
                acc += ds * (b0 * x[c0] + b1 * x[c0 + 1]
                             + b2 * x[c0 + 2] + b3 * x[c0 + 3])
            c += _TPB
        sh[t] = acc
        cuda.syncthreads()
        i = _TPB // 2
        while i > 0:
            if t < i:
                sh[t] += sh[t + i]
            cuda.syncthreads()
            i >>= 1
        if t == 0:
            out[r] = res[r] + sh[0]

    @cuda.jit(fastmath=True, cache=True)
    def _rmsnorm_batch(X, w, out, eps):
        # _rmsnorm_dev over a (batch, n) block, one block per row - the
        # prefill chunk's tensor.rmsnorm. Same deterministic tree
        # reduction per row as the decode kernel.
        sh = cuda.shared.array(_TPB, float32)
        r = cuda.blockIdx.x
        t = cuda.threadIdx.x
        n = X.shape[1]
        acc = float32(0.0)
        i = t
        while i < n:
            v = X[r, i]
            acc += v * v
            i += _TPB
        sh[t] = acc
        cuda.syncthreads()
        i = _TPB // 2
        while i > 0:
            if t < i:
                sh[t] += sh[t + i]
            cuda.syncthreads()
            i >>= 1
        inv = float32(1.0) / math.sqrt(sh[0] / float32(n) + float32(eps))
        i = t
        while i < n:
            out[r, i] = X[r, i] * inv * w[i]
            i += _TPB

    @cuda.jit(fastmath=True, cache=True)
    def _rope_store_batch(qk, v, cos_tab, sin_tab, bqk, bv, has_bqk,
                          has_bv, n_head, n_rot, neox, Q, Kl, Vl, pos0):
        # _rope_store_decode over a whole prefill chunk: the same thread
        # jobs (rotation pairs for the q then k heads, pass-through dims,
        # the v row) crossed with the chunk's rows; row b sits at
        # absolute position pos0 + b, contiguous by construction
        # (forward_batch's arange). Rotated q lands in the contiguous Q
        # block the batched attention kernel reads; k and v go straight
        # into the mirror rows at their absolute positions.
        i = cuda.grid(1)
        hd = Kl.shape[2]
        n_kv = Kl.shape[1]
        half = n_rot >> 1
        nheads = n_head + n_kv
        npair = nheads * half
        ntail = nheads * (hd - n_rot)
        jobs = npair + ntail + n_kv * hd
        b = i // jobs
        if b >= qk.shape[0]:
            return
        j = i - b * jobs
        pos = pos0 + b
        if j < npair:
            hh = j // half
            jj = j - hh * half
            base = hh * hd
            if neox:
                a = base + jj
                a2 = base + half + jj
            else:
                a = base + 2 * jj
                a2 = base + 2 * jj + 1
            x0 = qk[b, a]
            x1 = qk[b, a2]
            if has_bqk:
                x0 += bqk[a]
                x1 += bqk[a2]
            c = cos_tab[pos, jj]
            s = sin_tab[pos, jj]
            y0 = x0 * c - x1 * s
            y1 = x0 * s + x1 * c
            if hh < n_head:
                Q[b, hh, a - base] = y0
                Q[b, hh, a2 - base] = y1
            else:
                Kl[pos, hh - n_head, a - base] = y0
                Kl[pos, hh - n_head, a2 - base] = y1
        elif j < npair + ntail:
            k = j - npair
            w2 = hd - n_rot
            hh = k // w2
            dd = n_rot + (k - hh * w2)
            idx = hh * hd + dd
            x0 = qk[b, idx]
            if has_bqk:
                x0 += bqk[idx]
            if hh < n_head:
                Q[b, hh, dd] = x0
            else:
                Kl[pos, hh - n_head, dd] = x0
        else:
            k = j - npair - ntail
            kvh = k // hd
            dd = k - kvh * hd
            x0 = v[b, k]
            if has_bv:
                x0 += bv[k]
            Vl[pos, kvh, dd] = x0

    @cuda.jit(fastmath=True, cache=True)
    def _add_batch(res, src):
        # res += src over a (batch, n) block: the prefill chunk's
        # residual adds, which the CPU path does with one numpy `+`
        n = res.shape[1]
        total = res.shape[0] * n
        i = cuda.grid(1)
        if i < total:
            b = i // n
            j = i - b * n
            res[b, j] += src[b, j]

    _state = {
        "ok": True, "np": np, "cuda": cuda, "rt": rt,
        "H2D": rt.cudaMemcpyKind.cudaMemcpyHostToDevice,
        "D2H": rt.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        "version": numba_cuda.__version__,
        "name": name, "free_mb": free_b >> 20, "total_mb": total_b >> 20,
        "matvec": _matvec_codes, "gemm": _matmul_codes_gemm,
        "att_batch": _attention_batch_gqa,
        "att_part": _attention_decode_part,
        "att_combine": _attention_decode_combine,
        "rmsnorm": _rmsnorm_dev, "rope_store": _rope_store_decode,
        "silu_mul": _silu_mul_dev, "silu_mul_batch": _silu_mul_batch,
        "matvec_res": _matvec_codes_res,
        "rmsnorm_batch": _rmsnorm_batch,
        "rope_store_batch": _rope_store_batch, "add_batch": _add_batch,
        # per-size activation buffers, reused across calls (a cudaMalloc
        # per matvec costs more than the transfer it serves), plus the
        # grow-only pinned/device staging pairs for batched matmul
        "dx": {}, "hx": {}, "hout": {}, "dout": {},
        "hX2": None, "dX2": None, "hO2": None, "dO2": None,
        "matrices": 0, "uploaded_bytes": 0, "skipped": 0,
    }
    return _state


def available() -> bool:
    """True when the pinned JIT probe compiled, launched and verified."""
    return bool(_init().get("ok"))


def status() -> str:
    st = _init()
    if st.get("ok"):
        return (f"alpaccaroo-gpu active (numba-cuda=={st['version']}, "
                f"pin {NUMBA_CUDA_PIN}, our Python source)")
    if st.get("error"):
        return f"alpaccaroo-gpu inactive ({st['error']})"
    return "alpaccaroo-gpu inactive (CPU tiers in use)"


def doctor_line() -> str:
    """One line for `alpaccaroo doctor`: device + VRAM, or why there is none."""
    st = _init()
    if st.get("ok"):
        return (f"{st['name']} ({st['free_mb']} MiB free of "
                f"{st['total_mb']} MiB) - {status()}")
    return f"none detected - {status()}"


def vram_stats() -> dict[str, int]:
    st = _init()
    return {"matrices": st.get("matrices", 0),
            "uploaded_bytes": st.get("uploaded_bytes", 0),
            # chain K/V mirrors + prefill scratch: device-resident but
            # not weight uploads, so reported apart from the capped figure
            "chain_bytes": st.get("chain_bytes", 0),
            "skipped": st.get("skipped", 0)}


# ---- transfers -------------------------------------------------------------
# Raw runtime-API memcpys against pinned staging, not the numba copy
# helpers: per 16 KiB activation upload the numba path measured 47 us of
# Python argument handling, the raw call 31; the result download 17 vs 10.
# At ~160 matvec calls per decoded token that difference is milliseconds.

def _memcpy_h2d_async(st, dev, pinned, nbytes: int) -> None:
    err, = st["rt"].cudaMemcpyAsync(
        dev.__cuda_array_interface__["data"][0], pinned.ctypes.data,
        nbytes, st["H2D"], 0)
    if int(err):
        raise RuntimeError(f"cudaMemcpyAsync H2D: {err}")


def _memcpy_d2h(st, pinned, dev, nbytes: int) -> None:
    # synchronous: also the point where the legacy default stream drains,
    # which is what makes reusing the staging buffers safe (see docstring)
    err, = st["rt"].cudaMemcpy(
        pinned.ctypes.data, dev.__cuda_array_interface__["data"][0],
        nbytes, st["D2H"])
    if int(err):
        raise RuntimeError(f"cudaMemcpy D2H: {err}")


# Raw-pointer variants for the decode chain: the K/V mirror rows live at
# computed offsets inside whole-context device arrays, and building a
# DeviceNDArray view per row per layer per token costs more Python time
# than the 4 KiB copy it would describe. Plain integer pointer arithmetic
# instead; the arrays the pointers came from are held alive by the chain.

def _memcpy_h2d_ptr(st, dst_ptr: int, src_ptr: int, nbytes: int) -> None:
    """Synchronous H2D (mirror re-sync: once per prefill, not per token)."""
    err, = st["rt"].cudaMemcpy(dst_ptr, src_ptr, nbytes, st["H2D"])
    if int(err):
        raise RuntimeError(f"cudaMemcpy H2D: {err}")


def _memcpy_d2h_ptr_async(st, dst_ptr: int, src_ptr: int, nbytes: int) -> None:
    """Async D2H into pinned memory (per-token K/V row copy-back; the
    token's one sync drains it before the host reads)."""
    err, = st["rt"].cudaMemcpyAsync(dst_ptr, src_ptr, nbytes, st["D2H"], 0)
    if int(err):
        raise RuntimeError(f"cudaMemcpyAsync D2H: {err}")


def _memcpy_d2h_ptr(st, dst_ptr: int, src_ptr: int, nbytes: int) -> None:
    """Synchronous D2H to a raw host pointer (prefill's per-chunk K/V
    copy-back lands straight in the host cache rows - pageable memory,
    which the runtime synchronizes on anyway, so staging through a
    pinned block would only add a host-side memcpy per layer). The
    first such copy of a chunk is also the drain that makes restaging
    the chunk's pinned upload buffer safe."""
    err, = st["rt"].cudaMemcpy(dst_ptr, src_ptr, nbytes, st["D2H"])
    if int(err):
        raise RuntimeError(f"cudaMemcpy D2H: {err}")


def _upload_x(st, x):
    """Stage one float32 activation vector into the per-size device buffer."""
    np_ = st["np"]
    cols = x.shape[0]
    hx = st["hx"].get(cols)
    if hx is None:
        hx = st["cuda"].pinned_array(cols, np_.float32)
        st["hx"][cols] = hx
    hx[:] = x
    dx = st["dx"].get(cols)
    if dx is None:
        dx = st["cuda"].device_array(cols, np_.float32)
        st["dx"][cols] = dx
    _memcpy_h2d_async(st, dx, hx, cols * 4)
    return dx


def upload_vector(x):
    """Upload one activation vector so several GpuMatrix.matvec calls can
    share a single H2D (tensor.matvec_group's wqk+wv chain). Returns the
    device buffer, or None when the tier is inactive or the upload fails -
    callers then degrade to per-matvec uploads."""
    st = _init()
    if not st.get("ok"):
        return None
    try:
        x = st["np"].ascontiguousarray(x, dtype=st["np"].float32)
        return _upload_x(st, x)
    except Exception:
        return None


def _download_out(st, dout, rows: int):
    np_ = st["np"]
    hout = st["hout"].get(rows)
    if hout is None:
        hout = st["cuda"].pinned_array(rows, np_.float32)
        st["hout"][rows] = hout
    _memcpy_d2h(st, hout, dout, rows * 4)
    return np_.array(hout)  # fresh array: callers keep results across calls


# Batched staging (the GEMM path): grow-only 1-D pinned/device pairs,
# viewed to each call's (batch, cols)/(batch, rows) shape, instead of the
# matvec path's per-size dicts - prefill tail chunks make batch a free
# variable, and keying buffers on every (batch, cols) pair seen would
# accumulate hundreds of MiB of dead pinned pages over a long run. One
# async H2D per matmul_t call, one sync D2H; the D2H drains the legacy
# default stream, so the reuse contract is the same as the matvec buffers.

def _grown(st, key, n):
    """The 1-D staging buffer for `key`, grown to at least n float32s."""
    buf = st.get(key)
    if buf is None or buf.size < n:
        mk = (st["cuda"].pinned_array if key.startswith("h")
              else st["cuda"].device_array)
        buf = mk(n, st["np"].float32)
        st[key] = buf
    return buf


def _upload_batch(st, X):
    """Stage one (batch, cols) activation block; one async H2D."""
    n = X.shape[0] * X.shape[1]
    hX = _grown(st, "hX2", n)[:n].reshape(X.shape)
    hX[...] = X
    dX = _grown(st, "dX2", n)[:n].reshape(X.shape)
    _memcpy_h2d_async(st, dX, hX, n * 4)
    return dX


def _download_batch(st, dout, batch: int, rows: int):
    """Bring one (batch, rows) GEMM result home; one sync D2H."""
    np_ = st["np"]
    n = batch * rows
    hO = _grown(st, "hO2", n)[:n].reshape(batch, rows)
    _memcpy_d2h(st, hO, dout, n * 4)
    return np_.array(hO)  # fresh array, same contract as _download_out


# ---- batched attention (prefill path) --------------------------------------

def _att_stage(st, key, arr):
    """Grow-only pinned/device pair for one attention operand, keyed by
    name (each key keeps one dtype); one async H2D. Same drain-based
    reuse contract as the GEMM staging."""
    n = arr.size
    hk, dk = "hAtt" + key, "dAtt" + key
    h = st.get(hk)
    if h is None or h.size < n:
        h = st["cuda"].pinned_array(n, arr.dtype)
        st[hk] = h
    hv = h[:n].reshape(arr.shape)
    hv[...] = arr
    d = st.get(dk)
    if d is None or d.size < n:
        d = st["cuda"].device_array(n, arr.dtype)
        st[dk] = d
    dv = d[:n].reshape(arr.shape)
    _memcpy_h2d_async(st, dv, hv, n * arr.dtype.itemsize)
    return dv


def attention_batch(q, K, V, positions, group: int, inv_sqrt: float,
                    window: int = 0, kv_start: int = 0):
    """Batched causal GQA for the prefill path.

    q (t_q, n_head, hd), K/V (t_kv, n_kv, hd) host float32 (the model's
    KV-cache slices), positions absolute int32; optional sliding window
    and kv_start carry the Gemma 3 variant's semantics. Returns
    (t_q, n_head*hd) float32, or None when the tier is off, the geometry
    is out of range (head_dim > _ATT_HD), or a previous failure parked
    the path - callers keep their exact NumPy einsum either way, and it
    remains the reference. The K/V slice re-uploads every call because
    the cache is host-authoritative between chunks: on the 8B's full
    4096-token prefill that is 10.5 GiB of H2D over the run, measured
    (sync-instrumented) at 1.2 s / ~39% of the 3.1 s GPU attention stage
    - a bandwidth tax worth paying against the 147 s the NumPy einsum
    was heading for at that length.
    """
    st = _init()
    if not st.get("ok") or st.get("att_dead"):
        return None
    np_ = st["np"]
    try:
        q = np_.ascontiguousarray(q, dtype=np_.float32)
        K = np_.ascontiguousarray(K, dtype=np_.float32)
        V = np_.ascontiguousarray(V, dtype=np_.float32)
        if q.ndim != 3 or K.ndim != 3 or K.shape != V.shape:
            return None
        t_q, n_head, hd = q.shape
        t_kv, n_kv, hd_k = K.shape
        pos = np_.ascontiguousarray(positions, dtype=np_.int32)
        if (hd != hd_k or hd > _ATT_HD or t_q == 0 or t_kv == 0
                or group < 1 or n_kv * group != n_head
                or pos.shape != (t_q,)):
            return None
    except Exception:
        return None
    try:
        dq = _att_stage(st, "q", q)
        dK = _att_stage(st, "k", K)
        dV = _att_stage(st, "v", V)
        dp = _att_stage(st, "p", pos)
        n_out = t_q * n_head * hd
        dO = _grown(st, "dAtto", n_out)[:n_out].reshape(t_q, n_head, hd)
        gx = (t_q * group + _ATT_R - 1) // _ATT_R
        st["att_batch"][(gx, n_kv), _ATT_TPB](
            dq, dK, dV, dp, dO, group, float(inv_sqrt),
            int(window), int(kv_start))
        hO = _grown(st, "hAtto", n_out)[:n_out]
        _memcpy_d2h(st, hO, dO, n_out * 4)
        return np_.array(hO).reshape(t_q, n_head * hd)
    except Exception as e:
        # parked, not degraded per-matrix: attention holds no weights, so
        # the NumPy path recomputes losslessly from the host cache
        st["att_dead"] = True
        print(f"alpaccaroo-gpu: batch attention degraded to NumPy "
              f"({type(e).__name__}: {e})", file=sys.stderr)
        return None


# ---- fused prefill FFN -----------------------------------------------------

def _launch_gemm(st, gmat, dX, dout, batch: int) -> None:
    """Queue the tiled GEMM for a resident matrix, device in/out."""
    gx = (gmat.rows + _GEMM_ROWS - 1) // _GEMM_ROWS
    gy = (batch + _GEMM_BT - 1) // _GEMM_BT
    st["gemm"][(gx, gy), (_GEMM_TX, _GEMM_TY)](
        gmat._dq, gmat._dd, gmat._dd if gmat._dm is None else gmat._dm,
        dX, dout, gmat._shift, gmat._dm is not None)


def ffn_swiglu_batch(wgu, wdown, X):
    """The llama-class prefill FFN fused on the device: X @ wgu.T, then
    silu(gate) * up, then @ wdown.T - one activation upload, one result
    download. `wgu` is the fused gate|up matrix (2*n_ff rows), `wdown`
    the down projection. Returns (batch, n_embd) float32, or None when
    either matrix is not device-resident (mixed placement, degraded) or
    batch < 2 - callers keep the exact three-step host path, and batch 1
    keeps the matvec kernel's bit-identity with decode. Measured
    motivation: at the 8B's chunk shapes the host silu line alone costs
    13.2 ms per layer-chunk (np.exp over 256x14336) and the gate|up
    download plus act re-upload another ~42 MiB of PCIe; this path
    removes all three.
    """
    st = _init()
    if not st.get("ok"):
        return None
    if not (getattr(wgu, "is_gpu_matrix", False)
            and getattr(wdown, "is_gpu_matrix", False)):
        return None
    if wgu._dead or wdown._dead:
        return None
    n_ff = wdown.cols
    if wgu.rows != 2 * n_ff:
        return None
    np_ = st["np"]
    X = np_.ascontiguousarray(X, dtype=np_.float32)
    if X.ndim != 2 or X.shape[1] != wgu.cols:
        return None
    batch = X.shape[0]
    if batch < 2:
        return None
    try:
        dX = _upload_batch(st, X)
        n_gu = batch * wgu.rows
        dgu = _grown(st, "dFgu", n_gu)[:n_gu].reshape(batch, wgu.rows)
        _launch_gemm(st, wgu, dX, dgu, batch)
        n_act = batch * n_ff
        dact = _grown(st, "dFact", n_act)[:n_act].reshape(batch, n_ff)
        blocks = (n_act + _TPB - 1) // _TPB
        st["silu_mul_batch"][blocks, _TPB](dgu, dact)
        n_out = batch * wdown.rows
        dout = _grown(st, "dO2", n_out)[:n_out].reshape(batch, wdown.rows)
        _launch_gemm(st, wdown, dact, dout, batch)
        return _download_batch(st, dout, batch, wdown.rows)
    except Exception as e:
        # degrade the pair like any other runtime failure; the caller
        # recomputes on the host from the downloaded weights
        wgu._degrade(e)
        wdown._degrade(e)
        return None


class _VramBudget(Exception):
    """Upload refused by the free-VRAM reserve or ALPACCAROO_GPU_VRAM_MB."""


class GpuMatrix:
    """Quantized (rows x cols) matrix resident in VRAM, codes layout.

    Built from the raw GGUF block bytes at load time (the host QuantMatrix
    frees its unpack source, so wrapping one after construction is not
    possible) and holding ONLY device arrays - a matrix that fails to
    upload never constructs, and the loader falls back to the exact
    QuantMatrix it would have built anyway. A matrix that fails at runtime
    downloads its arrays once and computes on the host from then on.
    """

    # duck-typed marker checked by tensor.py: an isinstance test there
    # would drag the CUDA stack into every CPU-only import
    is_gpu_matrix = True

    __slots__ = ("dtype", "rows", "cols", "sub_len", "n_sub", "vram_nbytes",
                 "_shift", "_dq", "_dd", "_dm", "_dead",
                 "_hq3", "_hd", "_hm")

    def __init__(self, data, dtype: str, rows: int, cols: int):
        st = _init()
        if not st.get("ok"):
            raise RuntimeError("gpu tier inactive")
        geom = QUANT_GEOMETRY.get(dtype)
        if geom is None:
            raise ValueError(f"gpu matvec does not support {dtype}")
        block_elements, _bb, sub_len, _aff = geom
        if cols % block_elements:
            raise ValueError(
                f"{dtype} matrix columns must be a multiple of "
                f"{block_elements}")
        if sub_len & (sub_len - 1) or cols % 4:
            # the kernels index scales by c >> log2(sub_len) and load
            # codes four at a time; every GGUF geometry satisfies both
            raise ValueError(f"unsupported geometry {dtype}/{cols}")
        np_ = st["np"]
        q, d_eff, m_eff = np_unpack(data, rows * cols, dtype)
        qw = np_.ascontiguousarray(q.reshape(rows, cols)).view(np_.int32)
        d = np_.ascontiguousarray(d_eff.reshape(rows, -1))
        m = (None if m_eff is None
             else np_.ascontiguousarray(m_eff.reshape(rows, -1)))
        need = qw.nbytes + d.nbytes + (0 if m is None else m.nbytes)
        cap = _vram_cap_bytes()
        if cap and st["uploaded_bytes"] + need > cap:
            raise _VramBudget(f"cap {cap >> 20} MiB")
        err, free_b, _total = st["rt"].cudaMemGetInfo()
        if int(err):
            raise RuntimeError(f"cudaMemGetInfo: {err}")
        if need + (GPU_RESERVE_MB << 20) > free_b:
            raise _VramBudget(f"{free_b >> 20} MiB free")
        self.dtype = dtype
        self.rows = rows
        self.cols = cols
        self.sub_len = sub_len
        self.n_sub = cols // sub_len
        self._shift = sub_len.bit_length() - 1
        cuda = st["cuda"]
        self._dq = cuda.to_device(qw)
        self._dd = cuda.to_device(d)
        self._dm = None if m is None else cuda.to_device(m)
        self.vram_nbytes = need
        self._dead = False
        self._hq3 = self._hd = self._hm = None
        st["matrices"] += 1
        st["uploaded_bytes"] += need
        if st["matrices"] == 1:
            from . import tensor as _t
            _t._GPU_ACTIVE = True  # backend_name() flips to "gpu (cuda)"

    # ---- introspection ---------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        return self.rows, self.cols

    def storage_nbytes(self) -> int:
        """Bytes resident on the device for this matrix."""
        return self.vram_nbytes

    def _release_vram(self) -> None:
        """Return this matrix's bytes to the budget. Idempotent; called on
        degrade (device arrays dropped) and on garbage collection, so a
        freed model no longer counts against ALPACCAROO_GPU_VRAM_MB and a
        later budget exhaustion warns again."""
        st = _state
        if st and self.vram_nbytes:
            st["uploaded_bytes"] = max(
                0, st["uploaded_bytes"] - self.vram_nbytes)
            st["matrices"] = max(0, st["matrices"] - 1)
            st["skip_warned"] = False
            if st["matrices"] == 0:
                from . import tensor as _t
                _t._GPU_ACTIVE = False  # nothing device-resident any more
        self.vram_nbytes = 0

    def __del__(self):
        try:
            self._release_vram()
        except Exception:
            pass  # interpreter teardown: module globals may already be gone

    # ---- degraded host path ----------------------------------------------

    def _host_arrays(self):
        """Download codes+scales once; afterwards this matrix computes on
        the CPU with the same effective-scale algebra as QuantMatrix."""
        st = _init()
        np_ = st["np"]
        if self._hq3 is None:
            q = self._dq.copy_to_host().view(np_.int8)
            self._hq3 = q.reshape(self.rows, self.n_sub, self.sub_len)
            self._hd = self._dd.copy_to_host()
            self._hm = None if self._dm is None else self._dm.copy_to_host()
        return self._hq3, self._hd, self._hm

    def _degrade(self, exc) -> None:
        if not self._dead:
            # download while the device still answers. If even the download
            # fails the weights are unrecoverable (there is deliberately no
            # host copy) and the original error propagates - the one case
            # this tier cannot degrade from is losing the device itself.
            self._host_arrays()
            self._dead = True
            # a degraded matrix computes on the host from here on: drop the
            # device arrays (numba frees them on collection) and give the
            # bytes back to the budget instead of holding VRAM that will
            # never be read again
            self._dq = self._dd = self._dm = None
            self._release_vram()
            print(f"alpaccaroo-gpu: matrix {self.rows}x{self.cols} {self.dtype} "
                  f"degraded to CPU ({type(exc).__name__}: {exc})",
                  file=sys.stderr)

    def _matvec_host(self, x):
        st = _init()
        np_ = st["np"]
        q3, d, m = self._host_arrays()
        xs = np_.asarray(x, dtype=np_.float32).reshape(self.n_sub,
                                                       self.sub_len)
        blockdot = np_.einsum("rsl,sl->rs", q3, xs)
        out = np_.einsum("rs,rs->r", blockdot, d)
        if m is not None:
            out += m @ xs.sum(axis=1)
        return out

    def _matmul_host(self, X):
        st = _init()
        np_ = st["np"]
        q3, d, m = self._host_arrays()
        out = np_.empty((X.shape[0], self.rows), dtype=np_.float32)
        x_sub_sums = (X.reshape(X.shape[0], self.n_sub,
                                self.sub_len).sum(axis=2)
                      if m is not None else None)
        tile_rows = max(16, (4 << 20) // max(self.cols * 4, 1))
        for r0 in range(0, self.rows, tile_rows):
            r1 = min(self.rows, r0 + tile_rows)
            tile = np_.ascontiguousarray(q3[r0:r1], dtype=np_.float32)
            tile *= d[r0:r1, :, None]
            out[:, r0:r1] = X @ tile.reshape(r1 - r0, self.cols).T
            if m is not None:
                out[:, r0:r1] += x_sub_sums @ m[r0:r1].T
        return out

    # ---- matvec (decode path) ---------------------------------------------

    def __matmul__(self, x):
        return self.matvec(x)

    def matvec(self, x, dx=None):
        """self (rows x cols) @ x (cols) -> (rows) float32.

        `dx` optionally carries upload_vector(x) so matrices sharing an
        input vector upload it once (tensor.matvec_group).
        """
        st = _init()
        np_ = st["np"]
        x = np_.ascontiguousarray(x, dtype=np_.float32)
        if x.shape[0] != self.cols:
            raise ValueError(
                f"expected ({self.cols},) input, got {x.shape}")
        if self._dead:
            return self._matvec_host(x)
        try:
            if dx is None:
                dx = _upload_x(st, x)
            dout = st["dout"].get(self.rows)
            if dout is None:
                dout = st["cuda"].device_array(self.rows, np_.float32)
                st["dout"][self.rows] = dout
            st["matvec"][self.rows, _TPB](
                self._dq, self._dd,
                self._dd if self._dm is None else self._dm,
                dx, dout, self._shift, self._dm is not None)
            return _download_out(st, dout, self.rows)
        except Exception as e:
            self._degrade(e)
            return self._matvec_host(x)

    # ---- batched matmul (prefill path) -------------------------------------

    def matmul_t(self, X):
        """X (batch x cols) @ self.T -> (batch x rows) float32.

        Batch 1 is the decode path and rides the matvec kernel unchanged
        (bit-identical either way - same kernel, same buffers); every
        larger batch takes the tiled GEMM, one launch at any size.
        """
        st = _init()
        np_ = st["np"]
        X = np_.ascontiguousarray(X, dtype=np_.float32)
        if X.ndim != 2 or X.shape[1] != self.cols:
            raise ValueError(
                f"expected ({X.shape[0]}, {self.cols}) input, got {X.shape}")
        if self._dead:
            return self._matmul_host(X)
        batch = X.shape[0]
        if batch == 0:
            return np_.empty((0, self.rows), dtype=np_.float32)
        if batch == 1:
            return self.matvec(X[0]).reshape(1, self.rows)
        try:
            dX = _upload_batch(st, X)
            dout = _grown(st, "dO2",
                          batch * self.rows)[:batch * self.rows]
            dout = dout.reshape(batch, self.rows)
            gx = (self.rows + _GEMM_ROWS - 1) // _GEMM_ROWS
            gy = (batch + _GEMM_BT - 1) // _GEMM_BT
            st["gemm"][(gx, gy), (_GEMM_TX, _GEMM_TY)](
                self._dq, self._dd,
                self._dd if self._dm is None else self._dm,
                dX, dout, self._shift, self._dm is not None)
            return _download_batch(st, dout, batch, self.rows)
        except Exception as e:
            self._degrade(e)
            return self._matmul_host(X)

    # ---- row access (rare: only reached if a GpuMatrix is row-gathered) ----

    def row(self, i: int):
        if not 0 <= i < self.rows:
            raise IndexError(i)
        st = _init()
        np_ = st["np"]
        q3, d, m = ((self._hq3, self._hd, self._hm) if self._dead
                    else (None, None, None))
        if q3 is None:
            try:
                q = self._dq[i].copy_to_host().view(np_.int8)
                drow = self._dd[i].copy_to_host()
                mrow = None if self._dm is None \
                    else self._dm[i].copy_to_host()
            except Exception as e:
                self._degrade(e)
                q3, d, m = self._host_arrays()
        if q3 is not None:
            q = q3[i].reshape(-1)
            drow = d[i]
            mrow = None if m is None else m[i]
        v = q.reshape(self.n_sub, self.sub_len).astype(np_.float32)
        v *= drow[:, None]
        if mrow is not None:
            v += mrow[:, None]
        return v.reshape(-1)

    def rows_at(self, rows):
        st = _init()
        np_ = st["np"]
        idx = np_.asarray(rows, dtype=np_.int64)
        if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= self.rows):
            raise IndexError("row index out of range")
        if not idx.size:
            return np_.empty((0, self.cols), dtype=np_.float32)
        return np_.stack([self.row(int(i)) for i in idx])


def gpu_matrix(data, dtype: str, rows: int, cols: int):
    """Build a VRAM-resident matrix from raw GGUF block bytes.

    Returns None - and the caller keeps its exact CPU QuantMatrix - when
    the tier is inactive, the budget is spent, or anything at all fails;
    this function never raises.
    """
    st = _init()
    if not st.get("ok"):
        return None
    try:
        return GpuMatrix(data, dtype, rows, cols)
    except _VramBudget as e:
        st["skipped"] += 1
        if not st.get("skip_warned"):
            st["skip_warned"] = True
            print(f"alpaccaroo-gpu: VRAM budget reached ({e}) after "
                  f"{st['matrices']} matrices "
                  f"({st['uploaded_bytes'] >> 20} MiB); remaining matrices "
                  f"stay on the CPU tiers", file=sys.stderr)
        return None
    except Exception as e:
        st["skipped"] += 1
        if not st.get("fail_warned"):
            st["fail_warned"] = True
            print(f"alpaccaroo-gpu: upload failed ({type(e).__name__}: {e}); "
                  f"matrix stays on the CPU tiers", file=sys.stderr)
        return None


# ---- device-resident decode chain ------------------------------------------

class _ChainUnsupported(Exception):
    """This model or geometry cannot ride the decode chain. Not an error:
    the existing per-matvec path simply keeps decoding."""


def _chain_env_off() -> bool:
    return os.environ.get("ALPACCAROO_GPU_CHAIN", "").strip().lower() in (
        "0", "off", "no")


def _launch_mv(st, gmat, dx, dout) -> None:
    """Queue the matvec kernel on a resident matrix, device in/out."""
    if gmat._dead:
        raise RuntimeError("matrix degraded mid-chain")
    st["matvec"][gmat.rows, _TPB](
        gmat._dq, gmat._dd, gmat._dd if gmat._dm is None else gmat._dm,
        dx, dout, gmat._shift, gmat._dm is not None)


def _launch_mv_res(st, gmat, dx, dres) -> None:
    """Queue the fused matvec+residual kernel: dres += gmat @ dx."""
    if gmat._dead:
        raise RuntimeError("matrix degraded mid-chain")
    st["matvec_res"][gmat.rows, _TPB](
        gmat._dq, gmat._dd, gmat._dd if gmat._dm is None else gmat._dm,
        dx, dres, dres, gmat._shift, gmat._dm is not None)


class DecodeChain:
    """Device-resident single-token decode for one llama-class model.

    Everything between the token-embedding gather (host, ~16 KiB upload)
    and the logits download runs on the device: rmsnorm, the fused qkv
    matvecs, rope + K/V-mirror row store, GQA attention over the mirror,
    the residual-fused output/down projections and silu*up - one queued
    launch each, ONE synchronization per token. The host KV cache stays
    authoritative: the token's new K/V rows are async-copied back inside
    that same sync, and every host mutation path (prefill's batch writes,
    truncation, reset, CPU decode) reports through Model._chain_invalidate
    so the mirror re-uploads exactly the rows the host changed before the
    next chained token.

    Built lazily on the first device-resident forward - the first
    PREFILL chunk when chain_prefill runs, the first decoded token
    otherwise (mirror allocation + upload); construction raises
    _ChainUnsupported for anything it cannot serve bit-for-bit honestly
    - gemma-family forward passes (extra norms, GELU, softcap, scaled
    embeddings), heads wider than _ATT_HD, degraded or CPU-resident
    chain matrices - and chain_forward then parks the chain permanently
    for that model instance. A TIED output head is not a refusal: the
    final hidden vector downloads (that D2H is the one sync) and the
    host computes the logits, since the tied embedding never uploads.
    ANY runtime failure also parks the chain; the token that failed is
    recomputed by the unchanged per-matvec path.

    The same instance serves prefill_chunk, the device-resident prefill
    path: whole chunks run through the batch kernels with the chunk's
    K/V rows written straight into this mirror (and copied back to the
    authoritative host cache), so the batched attention reads the
    mirror instead of re-uploading host K/V every chunk and the first
    decoded token finds valid_upto already at n_past.
    """

    def __init__(self, model):
        st = _init()
        if not st.get("ok"):
            raise _ChainUnsupported("gpu tier inactive")
        hp = model.hp
        if hp.arch in ("gemma", "gemma3"):
            raise _ChainUnsupported("gemma-family forward pass")
        if hp.embed_scale != 1.0 or hp.final_logit_softcap > 0.0:
            raise _ChainUnsupported("scaled embedding / softcap")
        if hp.rope_style not in ("norm", "neox"):
            raise _ChainUnsupported(f"rope style {hp.rope_style}")
        hd = hp.head_dim
        if hd > _ATT_HD or hp.n_rot % 2 or not 0 < hp.n_rot <= hd:
            raise _ChainUnsupported("head geometry")
        if hp.n_kv <= 0 or hp.n_head % hp.n_kv:
            raise _ChainUnsupported("head grouping")
        if getattr(model, "_rope_cos", None) is None:
            raise _ChainUnsupported("no rope tables")
        np_ = st["np"]
        cuda = st["cuda"]

        def gm(W, role):
            if W is None or not getattr(W, "is_gpu_matrix", False) or W._dead:
                raise _ChainUnsupported(f"{role} not device-resident")
            return W

        qd = hp.n_head * hd
        kvd = hp.n_kv * hd
        picked = []
        for ly in model.layers:
            if ly.q_norm is not None or ly.post_attn_norm is not None:
                raise _ChainUnsupported("extra per-layer norms")
            wqk = None if ly.wqk is None else gm(ly.wqk, "wqk")
            wq = wk = None
            if wqk is None:
                wq = gm(ly.wq, "attn_q")
                wk = gm(ly.wk, "attn_k")
            wgu = None if ly.wgu is None else gm(ly.wgu, "wgu")
            wg = wu = None
            if wgu is None:
                wg = gm(ly.w_gate, "ffn_gate")
                wu = gm(ly.w_up, "ffn_up")
            picked.append((ly, wqk, wq, wk, gm(ly.wv, "attn_v"),
                           gm(ly.wo, "attn_output"), wgu, wg, wu,
                           gm(ly.w_down, "ffn_down")))

        n_ctx = model.n_ctx
        # ALPACCAROO_KV_F16=1: opt-in half-precision K/V mirror - stores cast
        # f32 -> f16 on the device, kernels read f16 back up to f32.
        # Halves the mirror's VRAM and the per-token K/V bandwidth decode
        # pays at depth, but the numbers legitimately shift (~1e-3), so
        # the default stays f32: every parity claim, and client integrations, run the
        # default. Read once at chain build; the host cache stays f32
        # and authoritative either way (rows come home widened).
        self.kv_f16 = os.environ.get("ALPACCAROO_KV_F16", "").strip() == "1"
        kv_dtype = np_.float16 if self.kv_f16 else np_.float32
        kv_itemsize = 2 if self.kv_f16 else 4
        mirror = 2 * hp.n_layer * n_ctx * kvd * kv_itemsize
        err, free_b, _total = st["rt"].cudaMemGetInfo()
        if int(err):
            raise RuntimeError(f"cudaMemGetInfo: {err}")
        if mirror + (GPU_RESERVE_MB << 20) > free_b:
            raise _ChainUnsupported(
                f"K/V mirror needs {mirror >> 20} MiB, {free_b >> 20} free")
        # visibility, not budget: the mirror is chain scratch, deliberately
        # outside the ALPACCAROO_GPU_VRAM_MB weight cap (which the docs define
        # as uploaded weight bytes), but vram_stats/doctor must not
        # under-report device residency by a gigabyte (review finding)
        st["chain_bytes"] = st.get("chain_bytes", 0) + mirror
        self._mirror_bytes = mirror

        self.n_layer = hp.n_layer
        self.n_ctx = n_ctx
        self.n_head = hp.n_head
        self.n_kv = hp.n_kv
        self.hd = hd
        self.qd = qd
        self.kvd = kvd
        self.n_embd = hp.n_embd
        self.n_ff = hp.n_ff
        self.n_rot = hp.n_rot
        self.neox = hp.rope_style == "neox"
        self.group = hp.n_head // hp.n_kv
        self.eps = float(hp.rms_eps)
        self.inv_sqrt = 1.0 / math.sqrt(hd)
        self.rowb = kvd * kv_itemsize

        self.dK = [cuda.device_array((n_ctx, hp.n_kv, hd), kv_dtype)
                   for _ in range(hp.n_layer)]
        self.dV = [cuda.device_array((n_ctx, hp.n_kv, hd), kv_dtype)
                   for _ in range(hp.n_layer)]
        self.kdev = [a.__cuda_array_interface__["data"][0] for a in self.dK]
        self.vdev = [a.__cuda_array_interface__["data"][0] for a in self.dV]
        self.dcos = cuda.to_device(
            np_.ascontiguousarray(model._rope_cos, dtype=np_.float32))
        self.dsin = cuda.to_device(
            np_.ascontiguousarray(model._rope_sin, dtype=np_.float32))
        # absolute positions 0..n_ctx-1: prefill chunks pass the
        # [pos0, pos0+batch) slice to the batched attention kernel, so
        # no per-chunk positions upload ever happens
        self.dpos_all = cuda.to_device(np_.arange(n_ctx, dtype=np_.int32))

        def dev_vec(v, size, role):
            if v is None:
                raise _ChainUnsupported(f"missing {role}")
            arr = np_.ascontiguousarray(v, dtype=np_.float32)
            if arr.shape != (size,):
                raise _ChainUnsupported(f"{role} shape {arr.shape}")
            return cuda.to_device(arr)

        self.layers = []
        for ly, wqk, wq, wk, wv, wo, wgu, wg, wu, wdown in picked:
            has_bqk = ly.bq is not None or ly.bk is not None
            dbqk = None
            if has_bqk:
                bq = ly.bq if ly.bq is not None else np_.zeros(qd, np_.float32)
                bk = ly.bk if ly.bk is not None else np_.zeros(kvd, np_.float32)
                dbqk = cuda.to_device(np_.ascontiguousarray(
                    np_.concatenate([bq, bk]), dtype=np_.float32))
            has_bv = ly.bv is not None
            dbv = None
            if has_bv:
                dbv = dev_vec(ly.bv, kvd, "attn_v bias")
            self.layers.append((
                dev_vec(ly.attn_norm, hp.n_embd, "attn_norm"),
                wqk, wq, wk, wv, wo,
                dev_vec(ly.ffn_norm, hp.n_embd, "ffn_norm"),
                wgu, wg, wu, wdown, dbqk, has_bqk, dbv, has_bv))
        self.dnorm_out = dev_vec(model.out_norm, hp.n_embd, "output_norm")

        self.dx = cuda.device_array(hp.n_embd, np_.float32)
        self.dh = cuda.device_array(hp.n_embd, np_.float32)
        self.dqk = cuda.device_array(qd + kvd, np_.float32)
        self.dvt = cuda.device_array(kvd, np_.float32)
        self.datt = cuda.device_array(qd, np_.float32)
        max_ck = (n_ctx + _DEC_SPLIT - 1) // _DEC_SPLIT
        self.dpm = cuda.device_array((hp.n_head, max_ck), np_.float32)
        self.dpl = cuda.device_array((hp.n_head, max_ck), np_.float32)
        self.dpacc = cuda.device_array((hp.n_head, max_ck, hd), np_.float32)
        self.dgu = cuda.device_array(2 * hp.n_ff, np_.float32)
        self.dact = cuda.device_array(hp.n_ff, np_.float32)
        self.dq_view = self.dqk[:qd]
        self.dk_view = self.dqk[qd:]
        self.dg_view = self.dgu[:hp.n_ff]
        self.du_view = self.dgu[hp.n_ff:]

        self.out_gm = (model.output if getattr(model.output, "is_gpu_matrix",
                                               False) else None)
        if self.out_gm is not None and self.out_gm._dead:
            self.out_gm = None  # degraded head computes host-side anyway
        self.n_vocab = (self.out_gm.rows if self.out_gm is not None else 0)
        self.hemb = cuda.pinned_array(hp.n_embd, np_.float32)
        self.hkv = cuda.pinned_array((hp.n_layer, 2, kvd), kv_dtype)
        hkv_base = self.hkv.ctypes.data
        self.hk_ptr = [hkv_base + (li * 2) * self.rowb
                       for li in range(hp.n_layer)]
        self.hv_ptr = [hkv_base + (li * 2 + 1) * self.rowb
                       for li in range(hp.n_layer)]
        # hxout is allocated unconditionally: a VRAM-resident head can
        # degrade AFTER the chain builds (any launch failure in its own
        # matvec), and step() then takes the hidden-download branch - which
        # would AttributeError on a build-time-conditional buffer and park
        # the chain forever (review finding, stage 2)
        self.hxout = cuda.pinned_array(hp.n_embd, np_.float32)
        if self.out_gm is not None:
            self.dlogits = cuda.device_array(self.n_vocab, np_.float32)
            self.hlog = cuda.pinned_array(self.n_vocab, np_.float32)
        else:
            self.dlogits = None

        half = hp.n_rot // 2
        rope_jobs = ((hp.n_head + hp.n_kv) * half
                     + (hp.n_head + hp.n_kv) * (hd - hp.n_rot) + kvd)
        self.rope_jobs = rope_jobs  # per row; prefill_chunk scales by batch
        self.rope_blocks = (rope_jobs + _TPB - 1) // _TPB
        self.silu_blocks = (hp.n_ff + _TPB - 1) // _TPB
        self.valid_upto = 0  # device-mirror rows in sync with the host
        # prefill scratch: grow-only device buffers (see _pbuf) plus the
        # pinned staging for the chunk's one embedding upload, allocated
        # on first use so a decode-only run pays nothing
        self._pdev = {}
        self._hpx = None

    def __del__(self):
        try:
            st = _state
            if st and getattr(self, "_mirror_bytes", 0):
                st["chain_bytes"] = max(
                    0, st.get("chain_bytes", 0) - self._mirror_bytes)
                self._mirror_bytes = 0
        except Exception:
            pass  # interpreter teardown: module globals may already be gone

    def invalidate(self, pos: int) -> None:
        """A host-cache mutation touched rows from `pos` on: forget them.
        Reported by Model._chain_invalidate from every mutation path."""
        if pos < self.valid_upto:
            self.valid_upto = max(0, int(pos))

    def _sync_mirror(self, st, model, a: int, b: int) -> None:
        """Upload host K/V rows [a, b) of every layer to the mirror.

        Synchronous copies through one grow-only pinned staging block
        (reuse is safe because each copy completes before the restage);
        this runs once after each prefill or truncation, not per token."""
        rows = b - a
        n = rows * self.kvd
        np_ = st["np"]
        stage = _grown(st, "hMirror", n)
        flat = stage.view(np_.float16)[:n] if self.kv_f16 else stage[:n]
        hv = flat.reshape(rows, self.n_kv, self.hd)
        nb = rows * self.rowb
        for li in range(self.n_layer):
            hv[...] = model.cache_k[li][a:b]
            _memcpy_h2d_ptr(st, self.kdev[li] + a * self.rowb,
                            stage.ctypes.data, nb)
            hv[...] = model.cache_v[li][a:b]
            _memcpy_h2d_ptr(st, self.vdev[li] + a * self.rowb,
                            stage.ctypes.data, nb)

    def _pbuf(self, st, key: str, n: int):
        """Grow-only device scratch for the prefill path (one chunk's
        activations). Chain-owned rather than module-level pools so a
        freed model releases the VRAM with its mirror; growth is charged
        to chain_bytes alongside the mirror (visibility, not budget)."""
        buf = self._pdev.get(key)
        if buf is None or buf.size < n:
            grown = (n - buf.size if buf is not None else n) * 4
            buf = st["cuda"].device_array(n, st["np"].float32)
            self._pdev[key] = buf
            st["chain_bytes"] = st.get("chain_bytes", 0) + grown
            self._mirror_bytes += grown
        return buf[:n]

    def _pinned_x(self, st, n: int):
        """Grow-only pinned staging for the chunk's embedding upload."""
        buf = self._hpx
        if buf is None or buf.size < n:
            buf = st["cuda"].pinned_array(n, st["np"].float32)
            self._hpx = buf
        return buf

    def _prefill_scratch_growth(self, batch: int) -> int:
        """Bytes of NEW device scratch a `batch`-token chunk would
        allocate - the grow-only buffers' existing capacity is free. The
        key/size table mirrors prefill_chunk's _pbuf calls exactly."""
        need = 0
        for key, n in (("x", batch * self.n_embd),
                       ("h", batch * self.n_embd),
                       ("o", batch * self.n_embd),
                       ("qk", batch * (self.qd + self.kvd)),
                       ("v", batch * self.kvd),
                       ("q3", batch * self.qd),
                       ("att", batch * self.qd),
                       ("gu", batch * 2 * self.n_ff),
                       ("act", batch * self.n_ff)):
            buf = self._pdev.get(key)
            if buf is None or buf.size < n:
                need += (n - (buf.size if buf is not None else 0)) * 4
        return need

    def prefill_chunk(self, model, tokens: list[int], want_logits: bool):
        """One prefill chunk fully on the device; returns the last row's
        logits, or None when want_logits is False. The caller
        (Model.forward_batch via chain_prefill) advances n_past on
        success, exactly as its own body would have.

        The chunk's embedding rows gather on the host (token_embd never
        uploads) and ride ONE H2D; everything between - batch rmsnorm,
        the qkv GEMMs, batch rope with the K/V rows stored straight into
        the decode mirror, batched GQA attention reading the mirror (no
        per-chunk K/V re-upload, the tax the attention_batch wrapper
        documents), the output/down GEMMs, silu*up and the residual adds
        - stays device-resident. Downloads: the chunk's K/V rows into
        the authoritative host cache (one synchronous copy-back stage
        per chunk, two copies per layer because the mirror is per-layer
        arrays; the first also drains the queued kernels, making the
        pinned staging reusable - the same contract every sync D2H in
        this file relies on), plus the logits or hidden row when the
        caller wants them. valid_upto then covers the chunk, so the
        first decoded token skips its bulk mirror upload."""
        st = _init()
        np_ = st["np"]
        batch = len(tokens)
        pos0 = model.n_past
        if pos0 + batch > self.n_ctx:
            raise RuntimeError("device K/V mirror full")
        if self.valid_upto < pos0:
            # prefix reuse restarted above the synced watermark: fill
            # the gap from the host cache, the decode step's own copy
            self._sync_mirror(st, model, self.valid_upto, pos0)
        self.valid_upto = pos0  # rows from here on are rewritten below
        from . import tensor as T
        x = np_.ascontiguousarray(T.matrix_rows(model.tok_embd, tokens),
                                  dtype=np_.float32)
        n_x = batch * self.n_embd
        hX = self._pinned_x(st, n_x)
        hX[:n_x] = x.reshape(-1)
        dX = self._pbuf(st, "x", n_x).reshape(batch, self.n_embd)
        _memcpy_h2d_async(st, dX, hX, n_x * 4)
        dH = self._pbuf(st, "h", n_x).reshape(batch, self.n_embd)
        dO = self._pbuf(st, "o", n_x).reshape(batch, self.n_embd)
        dqk = self._pbuf(st, "qk", batch * (self.qd + self.kvd)).reshape(
            batch, self.qd + self.kvd)
        dv = self._pbuf(st, "v", batch * self.kvd).reshape(batch, self.kvd)
        dQ = self._pbuf(st, "q3", batch * self.qd).reshape(
            batch, self.n_head, self.hd)
        datt = self._pbuf(st, "att", batch * self.qd).reshape(
            batch, self.n_head, self.hd)
        dgu = self._pbuf(st, "gu", batch * 2 * self.n_ff).reshape(
            batch, 2 * self.n_ff)
        dact = self._pbuf(st, "act", batch * self.n_ff).reshape(
            batch, self.n_ff)
        dpos = self.dpos_all[pos0:pos0 + batch]

        rms = st["rmsnorm_batch"]
        rope = st["rope_store_batch"]
        attb = st["att_batch"]
        smul = st["silu_mul_batch"]
        addb = st["add_batch"]
        eps = self.eps
        t_kv = pos0 + batch
        rope_blocks = (batch * self.rope_jobs + _TPB - 1) // _TPB
        add_blocks = (n_x + _TPB - 1) // _TPB
        silu_blocks = (batch * self.n_ff + _TPB - 1) // _TPB
        att_gx = (batch * self.group + _ATT_R - 1) // _ATT_R

        def gemm(gmat, din, dout):
            if gmat._dead:
                raise RuntimeError("matrix degraded mid-chain")
            _launch_gemm(st, gmat, din, dout, batch)

        for li, (dn1, wqk, wq, wk, wv, wo, dn2, wgu, wg, wu, wdown,
                 dbqk, has_bqk, dbv, has_bv) in enumerate(self.layers):
            rms[batch, _TPB](dX, dn1, dH, eps)
            if wqk is not None:
                gemm(wqk, dH, dqk)
            else:
                gemm(wq, dH, dqk[:, :self.qd])
                gemm(wk, dH, dqk[:, self.qd:])
            gemm(wv, dH, dv)
            rope[rope_blocks, _TPB](
                dqk, dv, self.dcos, self.dsin,
                self.dqk if dbqk is None else dbqk,
                self.dvt if dbv is None else dbv,
                has_bqk, has_bv, self.n_head, self.n_rot, self.neox,
                dQ, self.dK[li], self.dV[li], pos0)
            attb[(att_gx, self.n_kv), _ATT_TPB](
                dQ, self.dK[li][:t_kv], self.dV[li][:t_kv], dpos,
                datt, self.group, self.inv_sqrt, 0, 0)
            gemm(wo, datt.reshape(batch, self.qd), dO)
            addb[add_blocks, _TPB](dX, dO)
            rms[batch, _TPB](dX, dn2, dH, eps)
            if wgu is not None:
                gemm(wgu, dH, dgu)
            else:
                gemm(wg, dH, dgu[:, :self.n_ff])
                gemm(wu, dH, dgu[:, self.n_ff:])
            smul[silu_blocks, _TPB](dgu, dact)
            gemm(wdown, dact, dO)
            addb[add_blocks, _TPB](dX, dO)

        # host cache stays authoritative: the chunk's mirror rows come
        # home now (and drain everything queued above)
        nb = batch * self.rowb
        off = pos0 * self.rowb
        if self.kv_f16:
            # the f32 host cache cannot take f16 rows by raw memcpy:
            # stage them pinned and let the slice assignment widen
            n = batch * self.kvd
            stage = _grown(st, "hMirror", n)
            hrows = stage.view(np_.float16)[:n].reshape(
                batch, self.n_kv, self.hd)
            for li in range(self.n_layer):
                _memcpy_d2h_ptr(st, stage.ctypes.data,
                                self.kdev[li] + off, nb)
                model.cache_k[li][pos0:pos0 + batch] = hrows
                _memcpy_d2h_ptr(st, stage.ctypes.data,
                                self.vdev[li] + off, nb)
                model.cache_v[li][pos0:pos0 + batch] = hrows
        else:
            for li in range(self.n_layer):
                _memcpy_d2h_ptr(st, model.cache_k[li].ctypes.data + off,
                                self.kdev[li] + off, nb)
                _memcpy_d2h_ptr(st, model.cache_v[li].ctypes.data + off,
                                self.vdev[li] + off, nb)
        self.valid_upto = t_kv
        if not want_logits:
            return None
        rms_row = st["rmsnorm"]
        rms_row[1, _TPB](dX[batch - 1], self.dnorm_out, self.dh, eps)
        if self.out_gm is not None and not self.out_gm._dead:
            _launch_mv(st, self.out_gm, self.dh, self.dlogits)
            _memcpy_d2h(st, self.hlog, self.dlogits, self.n_vocab * 4)
            return np_.array(self.hlog)
        # tied (or degraded) head: hidden row home, host projects - the
        # same split step() uses
        _memcpy_d2h(st, self.hxout, self.dh, self.n_embd * 4)
        return T.matvec(model.output, np_.array(self.hxout))

    def step(self, model, token: int):
        """Decode one token fully on the device; returns logits. The
        caller (Model._forward_np) advances n_past on success, exactly as
        the CPU body would have."""
        st = _init()
        np_ = st["np"]
        pos = model.n_past
        if pos >= self.n_ctx:
            raise RuntimeError("device K/V mirror full")
        if self.valid_upto < pos:
            self._sync_mirror(st, model, self.valid_upto, pos)
            self.valid_upto = pos
        from . import tensor as T
        self.hemb[:] = T.matrix_row(model.tok_embd, token)
        _memcpy_h2d_async(st, self.dx, self.hemb, self.n_embd * 4)
        rms = st["rmsnorm"]
        rope = st["rope_store"]
        attp = st["att_part"]
        attc = st["att_combine"]
        smul = st["silu_mul"]
        dx, dh, dqk, dvt = self.dx, self.dh, self.dqk, self.dvt
        datt, dgu, dact = self.datt, self.dgu, self.dact
        dpm, dpl, dpacc = self.dpm, self.dpl, self.dpacc
        eps = self.eps
        t = pos + 1
        nck = (t + _DEC_SPLIT - 1) // _DEC_SPLIT
        off = pos * self.rowb
        for li, (dn1, wqk, wq, wk, wv, wo, dn2, wgu, wg, wu, wdown,
                 dbqk, has_bqk, dbv, has_bv) in enumerate(self.layers):
            rms[1, _TPB](dx, dn1, dh, eps)
            if wqk is not None:
                _launch_mv(st, wqk, dh, dqk)
            else:
                _launch_mv(st, wq, dh, self.dq_view)
                _launch_mv(st, wk, dh, self.dk_view)
            _launch_mv(st, wv, dh, dvt)
            rope[self.rope_blocks, _TPB](
                dqk, dvt, self.dcos, self.dsin,
                dqk if dbqk is None else dbqk,
                dvt if dbv is None else dbv,
                has_bqk, has_bv, self.n_head, self.n_rot, self.neox,
                self.dK[li], self.dV[li], pos)
            attp[(self.n_head, nck), _DEC_TPB](
                dqk, self.dK[li], self.dV[li], dpm, dpl, dpacc, t,
                self.group, self.inv_sqrt)
            attc[self.n_head, _DEC_TPB](dpm, dpl, dpacc, datt, nck)
            _launch_mv_res(st, wo, datt, dx)
            _memcpy_d2h_ptr_async(st, self.hk_ptr[li], self.kdev[li] + off,
                                  self.rowb)
            _memcpy_d2h_ptr_async(st, self.hv_ptr[li], self.vdev[li] + off,
                                  self.rowb)
            rms[1, _TPB](dx, dn2, dh, eps)
            if wgu is not None:
                _launch_mv(st, wgu, dh, dgu)
            else:
                _launch_mv(st, wg, dh, self.dg_view)
                _launch_mv(st, wu, dh, self.du_view)
            smul[self.silu_blocks, _TPB](dgu, dact)
            _launch_mv_res(st, wdown, dact, dx)
        rms[1, _TPB](dx, self.dnorm_out, dh, eps)
        if self.out_gm is not None and not self.out_gm._dead:
            _launch_mv(st, self.out_gm, dh, self.dlogits)
            _memcpy_d2h(st, self.hlog, self.dlogits, self.n_vocab * 4)
            logits = np_.array(self.hlog)
        else:
            # tied (or degraded) head: the hidden vector comes home in
            # the one sync and the host projects the logits, exactly as
            # the existing path does for a never-uploaded embedding
            _memcpy_d2h(st, self.hxout, dh, self.n_embd * 4)
            logits = T.matvec(model.output, np_.array(self.hxout))
        # that sync drained the queued row copies: scatter them into the
        # authoritative host cache before anyone can observe this token
        hkv = self.hkv
        for li in range(self.n_layer):
            model.cache_k[li][pos] = hkv[li, 0].reshape(self.n_kv, self.hd)
            model.cache_v[li][pos] = hkv[li, 1].reshape(self.n_kv, self.hd)
        self.valid_upto = pos + 1
        return logits


def chain_forward(model, token: int):
    """One device-resident decode step for `model`, or None when the
    chain is unavailable - the caller then runs the existing path, which
    is always correct. The chain builds lazily on the first decoded token
    (mirror allocation and upload); ANY failure, build or runtime, parks
    it permanently for this model instance (never the tier), and
    ALPACCAROO_GPU_CHAIN=0 refuses it outright."""
    st = _init()
    if not st.get("ok"):
        return None
    ch = model._gpu_chain
    if ch is None:
        if model._gpu_chain_dead or _chain_env_off():
            return None
        try:
            ch = DecodeChain(model)
        except _ChainUnsupported:
            model._gpu_chain_dead = True
            return None
        except Exception as e:
            model._gpu_chain_dead = True
            print(f"alpaccaroo-gpu: decode chain unavailable "
                  f"({type(e).__name__}: {e})", file=sys.stderr)
            return None
        model._gpu_chain = ch
    try:
        return ch.step(model, token)
    except Exception as e:
        model._gpu_chain = None
        model._gpu_chain_dead = True
        print(f"alpaccaroo-gpu: decode chain disabled ({type(e).__name__}: "
              f"{e}); decoding continues on the per-matvec path",
              file=sys.stderr)
        return None


# consecutive prefill-chunk failures before the device prefill path is
# parked for the model instance; a success resets the streak
_PREFILL_PARK_AFTER = 3


def chain_prefill(model, tokens, want_logits: bool):
    """One device-resident prefill chunk for `model`, or None when the
    path is unavailable - the caller (Model.forward_batch) then runs its
    existing body, which is always correct. Success is a 1-tuple
    carrying the logits (None inside it when want_logits is False, so a
    legitimately logits-free chunk is distinguishable from a refusal).

    Shares the decode chain's gate and instance: llama-class geometry,
    every chain matrix resident, and ALPACCAROO_GPU_CHAIN=0 refuses both
    paths - no knob of its own. The chain now builds at the first
    prefill chunk rather than the first decoded token, so the chunk's
    K/V rows land in the mirror as they are computed and the decode
    chain's bulk mirror upload becomes a no-op. Batch 1 stays on the
    existing path (GEMM batch 1 rides the matvec kernel there,
    bit-identical with decode - see GpuMatrix.matmul_t).

    Failure policy: a chunk whose scratch growth would not fit in free
    VRAM (minus the reserve) is refused BEFORE any allocation, and a
    runtime failure only counts toward a consecutive-failure streak -
    the path parks for good at _PREFILL_PARK_AFTER in a row, so one
    transient VRAM spike from another process costs one chunk, not the
    device prefill path for the life of the serve. Either way the
    caller recomputes the refused chunk; the decode chain keeps its own
    verdict (though a chain that failed on ITS side also takes this
    path down, since the mirror dies with the instance)."""
    st = _init()
    if not st.get("ok") or len(tokens) < 2:
        return None
    if model._gpu_prefill_dead:
        return None
    ch = model._gpu_chain
    if ch is None:
        if model._gpu_chain_dead or _chain_env_off():
            return None
        try:
            ch = DecodeChain(model)
        except _ChainUnsupported:
            model._gpu_chain_dead = True
            return None
        except Exception as e:
            model._gpu_chain_dead = True
            print(f"alpaccaroo-gpu: decode chain unavailable "
                  f"({type(e).__name__}: {e})", file=sys.stderr)
            return None
        model._gpu_chain = ch
    growth = ch._prefill_scratch_growth(len(tokens))
    if growth > 0:
        err, free_b, _total = st["rt"].cudaMemGetInfo()
        if err == 0 and growth + (GPU_RESERVE_MB << 20) > free_b:
            return None  # no room right now: not a failure, no streak
    try:
        out = (ch.prefill_chunk(model, tokens, want_logits),)
    except Exception as e:
        model._gpu_prefill_fails += 1
        if model._gpu_prefill_fails >= _PREFILL_PARK_AFTER:
            model._gpu_prefill_dead = True
            print(f"alpaccaroo-gpu: device prefill disabled "
                  f"({type(e).__name__}: {e}); prefill continues on the "
                  f"existing path", file=sys.stderr)
        else:
            print(f"alpaccaroo-gpu: device prefill chunk failed "
                  f"({type(e).__name__}: {e}); the caller recomputes it "
                  f"(streak {model._gpu_prefill_fails}/{_PREFILL_PARK_AFTER})",
                  file=sys.stderr)
        return None
    model._gpu_prefill_fails = 0
    return out


def warmup() -> None:
    """Trigger JIT compilation once (cached on disk afterwards)."""
    st = _init()
    if not st.get("ok"):
        return
    try:
        np_ = st["np"]
        cuda = st["cuda"]
        qw = cuda.to_device(np_.zeros((2, 8), np_.int32))
        d = cuda.to_device(np_.zeros((2, 1), np_.float32))
        x = cuda.to_device(np_.zeros(32, np_.float32))
        out = cuda.device_array(2, np_.float32)
        st["matvec"][2, _TPB](qw, d, d, x, out, 5, False)
        st["matvec"][2, _TPB](qw, d, d, x, out, 5, True)
        dX = cuda.to_device(np_.zeros((2, 32), np_.float32))
        o2 = cuda.device_array((2, 2), np_.float32)
        st["gemm"][(1, 1), (_GEMM_TX, _GEMM_TY)](qw, d, d, dX, o2, 5, False)
        st["gemm"][(1, 1), (_GEMM_TX, _GEMM_TY)](qw, d, d, dX, o2, 5, True)
        # stage-2 kernels: batch attention plus the decode-chain set, so
        # neither the first prefill chunk nor the first decoded token
        # pays a JIT compile
        dq3 = cuda.to_device(np_.zeros((2, 2, 8), np_.float32))
        dkv3 = cuda.to_device(np_.zeros((2, 1, 8), np_.float32))
        dpos = cuda.to_device(np_.zeros(2, np_.int32))
        do3 = cuda.device_array((2, 2, 8), np_.float32)
        st["att_batch"][(1, 1), _ATT_TPB](dq3, dkv3, dkv3, dpos, do3, 2,
                                          1.0, 0, 0)
        vec = cuda.to_device(np_.zeros(32, np_.float32))
        vout = cuda.device_array(32, np_.float32)
        st["rmsnorm"][1, _TPB](vec, vec, vout, 1e-5)
        st["matvec_res"][2, _TPB](qw, d, d, x, out, out, 5, False)
        st["matvec_res"][2, _TPB](qw, d, d, x, out, out, 5, True)
        gu = cuda.to_device(np_.zeros(16, np_.float32))
        act = cuda.device_array(8, np_.float32)
        st["silu_mul"][1, _TPB](gu, act)
        gu2 = cuda.to_device(np_.zeros((2, 16), np_.float32))
        act2 = cuda.device_array((2, 8), np_.float32)
        st["silu_mul_batch"][1, _TPB](gu2, act2)
        ctab = cuda.to_device(np_.zeros((4, 2), np_.float32))
        Kl = cuda.device_array((4, 1, 4), np_.float32)
        Vl = cuda.device_array((4, 1, 4), np_.float32)
        qk8 = cuda.to_device(np_.zeros(8, np_.float32))
        v4 = cuda.to_device(np_.zeros(4, np_.float32))
        for neox in (False, True):
            st["rope_store"][1, _TPB](qk8, v4, ctab, ctab, qk8, v4,
                                      False, False, 1, 4, neox, Kl, Vl, 0)
        # prefill-chain batch kernels, so the first device chunk pays no
        # JIT compile either
        ob = cuda.device_array((2, 32), np_.float32)
        st["rmsnorm_batch"][2, _TPB](dX, x, ob, 1e-5)
        st["add_batch"][1, _TPB](ob, dX)
        qkb = cuda.to_device(np_.zeros((2, 8), np_.float32))
        vb = cuda.to_device(np_.zeros((2, 4), np_.float32))
        Qb = cuda.device_array((2, 1, 4), np_.float32)
        for neox in (False, True):
            st["rope_store_batch"][1, _TPB](qkb, vb, ctab, ctab, qk8, v4,
                                            False, False, 1, 4, neox,
                                            Qb, Kl, Vl, 0)
        adec = cuda.device_array(4, np_.float32)
        pm = cuda.device_array((1, 1), np_.float32)
        pl = cuda.device_array((1, 1), np_.float32)
        pacc = cuda.device_array((1, 1, 4), np_.float32)
        st["att_part"][(1, 1), _DEC_TPB](v4, Kl, Vl, pm, pl, pacc, 2, 1, 1.0)
        st["att_combine"][1, _DEC_TPB](pm, pl, pacc, adec, 1)
        cuda.synchronize()
    except Exception as e:
        # warn, but do NOT clear _state: matrices already resident hold
        # references into it, and their per-call degrade path (download,
        # then compute on the host) needs the state alive to run at all
        print(f"alpaccaroo: gpu warmup failed ({type(e).__name__}: {e}); "
              f"kernel calls will degrade per-matrix to the CPU tiers",
              file=sys.stderr)


__all__ = ["DecodeChain", "GpuMatrix", "NUMBA_CUDA_PIN", "attention_batch",
           "available", "chain_forward", "chain_prefill", "doctor_line",
           "ffn_swiglu_batch", "gpu_matrix", "status", "upload_vector",
           "vram_stats", "warmup"]
