# Alpaccaroo - our own fused quantized-matvec kernels, written in Python.
# MIT License. See LICENSE.
"""Alpaccaroo's native-speed kernels: Python source, optionally JIT-compiled.

Every kernel here is Alpaccaroo's own algorithm, authored and maintained in
this file as ordinary Python. When the OPTIONAL, PINNED Numba JIT is
installed (``pip install alpaccaroo[kernels]``), these functions are compiled
at runtime to native SIMD machine code and the quantized decode path runs
at memory-bandwidth speed while weights stay quantized in RAM. Without
Numba - or with ``ALPACCAROO_KERNELS=0`` - nothing changes: the NumPy and
pure-Python paths remain the reference implementations and the fallback.

Pin policy: Numba is locked to ``NUMBA_PIN`` below, a combination
validated against this code and the supported NumPy range. The pin is
never updated implicitly; a different installed Numba version deactivates
the kernels (set ``ALPACCAROO_KERNELS=force`` to override at your own risk).
"""

from __future__ import annotations

import os

from . import _platform

NUMBA_PIN = "0.65.1"

# Batch at which the wide (batch-contiguous) matmul kernel overtakes the
# narrow one. Measured on a 14336x4096 Q4_K matrix, milliseconds:
#     batch      1     4     8    16    32    64   128
#     narrow   1.5   4.3  11.3  23.7  39.0  87.1  248.9
#     wide    15.6  23.0  16.9  19.4  23.5  41.6   76.4
# The wide kernel allocates one row accumulator, which dominates when there
# are few columns to amortize it over; the narrow one strides the batch,
# which dominates once there are many. They cross between 8 and 16.
NARROW_BATCH = 8

# ---- narrow-matrix dispatch (PERFORMANCE_PLAN Package E) --------------------
#
# Fanning a matvec across the thread pool is not free: entering the parallel
# region, chunking the row loop, and having several threads write into one
# small output array all cost. Below some amount of work that cost exceeds
# what the extra threads save, and running the SAME kernel on one thread is
# faster. Measured with bench.paired_compare (round-robin between thread
# counts, so a machine whose clock wanders cannot fake the result) on a
# 4-core Tiger Lake, microseconds per matvec:
#
#     shape                elems    t=1     t=2     t=4    best
#     Q4_K   64x2048     131072    49.0    49.5    63.9   1   (1.30x)
#     Q4_K  128x2048     262144    76.3    69.8    67.7   4
#     Q4_K  256x2048     524288   157.6   114.0   102.7   4   (1.53x)
#     Q6_K  256x2048     524288   220.6   327.0   420.2   1   (1.90x)
#     Q4_K  512x2048    1048576   307.0   202.4   270.4   2
#     Q4_K 2048x2048    4194304  2443.5  1849.3  1755.6   4
#
# The crossover is real but shape- AND dtype-dependent, so the default is
# set at the smallest size where every measured dtype agreed (131072) - it
# can only help shapes that were losing and cannot regress a larger one.
# `alpaccaroo tune --crossover` measures the real threshold for a given
# machine and caches it; ALPACCAROO_SERIAL_MATVEC_ELEMS overrides both.
#
# Note the same table taken WITHOUT paired comparison reported the opposite
# winner for Q4_K 256x2048. On a laptop that cycles its power limits, two
# separate timing runs are not comparable. Re-measure this way, or not at
# all.
SERIAL_MATVEC_ELEMS_DEFAULT = 131072

# Activation quantization is a parallel region over cols/256 blocks - eight
# of them for a 2048-wide model. Same rule as above, its own knob, sized in
# columns because that is what the loop trips on. Paired measurement on the
# same 4-core Tiger Lake, microseconds per call:
#
#     cols       512   1024   2048   4096   8192  11008  16384
#     pool      19.5   34.7   35.0   44.3   89.3  119.2  135.1
#     serial    16.6   38.1   29.5   40.5   84.9  113.6  161.0
#     ratio     0.84   0.90   0.90   0.92   0.94   0.91   1.30
#
# Serial wins everywhere up to 11008 and loses at 16384, so in ISOLATION
# the crossover on four cores is around 12k.
#
# It ships OFF anyway, and the reason is the most useful thing in this file.
# Setting it to 8192 makes the per-call benchmark 6-10% better and made the
# real decode loop worse: an end-to-end paired A/B on qwen2.5-3B (25 ABBA
# rounds, one process, one KV state) lost 22 of 25 rounds at a median of
# 1.20x. Rerunning the identical A/B with only this knob returned to 0 gave
# 14 of 25 and a median of 0.976 - the regression tracked this setting.
#
# The mechanism is NOT established. The obvious suspect was the thread-pool
# resize this forces ~216 times per token (a 2048-wide model quantizes
# below the threshold and then matvecs above it, three times per layer),
# but a direct probe - the same kernel work with and without a resize
# between each pair - measured no cost at all (11/25 rounds, ratio 1.06).
# So: the empirical result stands, its cause does not. Do not write a
# confident explanation here until someone has one.
#
# Leave it at 0 unless you have re-measured END TO END with a sign test,
# not per call. The knob stays because the isolated win is real.
SERIAL_QUANTIZE_COLS_DEFAULT = 0

_tuned_serial_elems: int | None = None   # set by tuning.apply_cached_thresholds


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def serial_matvec_elems() -> int:
    """Weight count at or below which a matvec runs on one thread."""
    if os.environ.get("ALPACCAROO_SERIAL_MATVEC_ELEMS", "").strip():
        return _env_int("ALPACCAROO_SERIAL_MATVEC_ELEMS",
                        SERIAL_MATVEC_ELEMS_DEFAULT)
    if _tuned_serial_elems is not None:
        return _tuned_serial_elems
    return SERIAL_MATVEC_ELEMS_DEFAULT


def set_tuned_serial_elems(value: "int | None") -> None:
    """Install a measured threshold (see alpaccaroo.tuning). The environment
    variable still wins - a user who pinned a number keeps it."""
    global _tuned_serial_elems
    _tuned_serial_elems = None if value is None else max(0, int(value))


_state: dict | None = None  # lazy: {"matvec": compiled fn} or {} if inactive
_cur_threads = 0            # what we last asked numba for; avoids a call per
#                             matvec just to read it back
_max_threads = 0


def int_dot_enabled() -> bool:
    """True when the integer-dot decode path is on (default when kernels
    are active; ALPACCAROO_INT_DOT=0 reverts to the f32-activation kernels)."""
    return (os.environ.get("ALPACCAROO_INT_DOT", "").strip().lower()
            not in ("0", "off", "no")) and available()


def _init() -> dict:
    global _state
    if _state is not None:
        return _state
    _state = {}
    mode = os.environ.get("ALPACCAROO_KERNELS", "").strip().lower()
    if mode in ("0", "off", "no") or os.environ.get("ALPACCAROO_PURE"):
        return _state
    try:
        import numpy as np
        import numba
        from numba import njit, prange
    except Exception:
        return _state
    if numba.__version__ != NUMBA_PIN and mode != "force":
        import sys
        print(f"alpaccaroo: numba {numba.__version__} != pinned {NUMBA_PIN}; "
              f"kernels disabled (ALPACCAROO_KERNELS=force to override)",
              file=sys.stderr)
        return _state

    # Thread count: prefer physical cores for the bandwidth-bound decode
    # loop unless the user chose explicitly (either knob wins over us).
    threads_env = os.environ.get("ALPACCAROO_THREADS", "").strip()
    if threads_env:
        try:
            numba.set_num_threads(max(1, int(threads_env)))
        except (ValueError, RuntimeError):
            pass
    elif "NUMBA_NUM_THREADS" not in os.environ:
        phys = _platform.physical_cores()
        if 0 < phys < numba.get_num_threads():
            try:
                numba.set_num_threads(phys)
            except RuntimeError:
                pass

    @njit(parallel=True, fastmath=True, cache=True)
    def _matvec_codes(q3, d, m, xs, xsums, affine):
        # out[r] = sum_s d[r,s] * (q3[r,s,:] . xs[s,:])  (+ m[r,s]*xsums[s])
        rows, nsub, sub_len = q3.shape
        out = np.empty(rows, np.float32)
        for r in prange(rows):
            acc = np.float32(0.0)
            for s in range(nsub):
                dot = np.float32(0.0)
                qb = q3[r, s]
                xb = xs[s]
                for j in range(sub_len):
                    dot += np.float32(qb[j]) * xb[j]
                acc += d[r, s] * dot
            if affine:
                macc = np.float32(0.0)
                for s in range(nsub):
                    macc += m[r, s] * xsums[s]
                acc += macc
            out[r] = acc
        return out

    @njit(parallel=True, fastmath=True, cache=True)
    def _matmul_codes(q3, d, m, Xs, Xsums, affine):
        # out[b,r] = sum_s d[r,s]*(q3[r,s,:] . Xs[b,s,:]) (+ m[r,s]*Xsums[b,s])
        #
        # The row loop is outermost so each row's codes are read from RAM
        # once and reused across the whole batch. The alternative - the
        # NumPy path's dequantize-to-float32-then-GEMM - pays a cost
        # proportional to the *weights*, not the batch, which is why a
        # one-token batch used to cost as much as a 256-token one.
        rows, nsub, sub_len = q3.shape
        batch = Xs.shape[0]
        out = np.zeros((batch, rows), np.float32)
        for r in prange(rows):
            for s in range(nsub):
                qb = q3[r, s]
                drs = d[r, s]
                for b in range(batch):
                    xb = Xs[b, s]
                    dot = np.float32(0.0)
                    for j in range(sub_len):
                        dot += np.float32(qb[j]) * xb[j]
                    out[b, r] += drs * dot
            if affine:
                for s in range(nsub):
                    mrs = m[r, s]
                    for b in range(batch):
                        out[b, r] += mrs * Xsums[b, s]
        return out

    @njit(parallel=True, fastmath=True, cache=True)
    def _matmul_codes_wide(q3, d, m, Xt, XsumsT, affine):
        # Same result as _matmul_codes, laid out for wider batches: Xt is
        # (nsub, sub_len, batch) so the innermost loop sweeps the batch
        # contiguously and vectorizes, where the narrow kernel strides it.
        # The scale is folded into the code value, which keeps the row
        # accumulator the only scratch and matches the tiled path's
        # scale-then-sum order.
        rows, nsub, sub_len = q3.shape
        batch = Xt.shape[2]
        out = np.empty((rows, batch), np.float32)
        for r in prange(rows):
            acc = np.zeros(batch, np.float32)
            for s in range(nsub):
                qb = q3[r, s]
                drs = d[r, s]
                for j in range(sub_len):
                    qv = drs * np.float32(qb[j])
                    xj = Xt[s, j]
                    for b in range(batch):
                        acc[b] += qv * xj[b]
                if affine:
                    mrs = m[r, s]
                    xsb = XsumsT[s]
                    for b in range(batch):
                        acc[b] += mrs * xsb[b]
            for b in range(batch):
                out[r, b] = acc[b]
        return out

    # ---- integer-dot decode kernels --------------------------------------
    #
    # Numba's scalar integer arithmetic promotes to int64, which compiles a
    # dot product to 8-lane vpmuldq/vpaddq and hides every 8/16-bit SIMD
    # pattern from LLVM. Re-casting the accumulator each step
    # (`acc = np.int32(acc + i32 * i32)`) keeps the IR in i32, and LLVM 20
    # then forms vpmaddwd and folds it into AVX-512 VNNI's vpdpwssd - 32
    # int16 MACs per instruction, measured 220 Gw/s flat vs the 105 Gw/s
    # f32 FMA ceiling on the same machine. Every kernel below relies on it.
    #
    # Weight-side values are the file's own: 6-bit sub-block scales as
    # integers and the per-256-block d/dmin as raw f16 bits (decoded through
    # a 65536-entry f32 table, L2-resident). Activations are quantized to
    # int8 per 256-element block (scale = absmax/127), so a sub-block dot is
    # sum(sc * code * xq) with sc*code <= 63*15 in int16, exact in vpdpwssd's
    # pairwise-int32 accumulation; float32 touches each 256-block twice.

    @njit(parallel=True, fastmath=True, cache=True)
    def _quantize_acts(x):
        # per-256-block int8 quantization + per-32 sums of the quantized
        # values (the Q4_K min term needs them; exact by construction)
        cols = x.shape[0]
        nblk = cols // 256
        xq = np.empty(cols, np.int8)
        ascale = np.empty(nblk, np.float32)
        bsums = np.empty(nblk * 8, np.int32)
        for b in prange(nblk):
            base = b * 256
            amax = np.float32(0.0)
            for j in range(256):
                a = np.abs(x[base + j])
                if a > amax:
                    amax = a
            if amax > np.float32(0.0):
                scale = amax / np.float32(127.0)
                inv = np.float32(127.0) / amax
            else:
                scale = np.float32(1.0)
                inv = np.float32(0.0)
            ascale[b] = scale
            for s in range(8):
                off = base + s * 32
                bsum = np.int32(0)
                for j in range(32):
                    t = x[off + j] * inv
                    v = np.int32(t + (np.float32(0.5) if t >= np.float32(0.0)
                                      else np.float32(-0.5)))
                    xq[off + j] = np.int8(v)
                    bsum = np.int32(bsum + v)
                bsums[b * 8 + s] = bsum
        return xq, ascale, bsums

    @njit(parallel=True, fastmath=True, cache=True)
    def _matvec_q4k_int(qp, sc, mn, dh, dmh, lut, xq, ascale, bsums):
        # qp: (rows, cols//2) u8, the file's own split-nibble qs layout -
        # byte j of 32-byte chunk c holds elements 64c+j (low nibble) and
        # 64c+32+j (high). The j-loop walks all four chunks of a 256-block
        # at once with the 6-bit scales premultiplied into the int16 codes
        # (sc*code <= 945 < 2^15), so the whole block accumulates into one
        # vector register and reduces to scalar once per 256 weights.
        rows = qp.shape[0]
        nblk = ascale.shape[0]
        out = np.empty(rows, np.float32)
        for r in prange(rows):
            acc = np.float32(0.0)
            for b in range(nblk):
                p0 = b * 128
                x0 = b * 256
                s0 = np.int16(sc[r, b, 0])
                s1 = np.int16(sc[r, b, 1])
                s2 = np.int16(sc[r, b, 2])
                s3 = np.int16(sc[r, b, 3])
                s4 = np.int16(sc[r, b, 4])
                s5 = np.int16(sc[r, b, 5])
                s6 = np.int16(sc[r, b, 6])
                s7 = np.int16(sc[r, b, 7])
                blk_i = np.int32(0)
                for j in range(32):
                    v0 = qp[r, p0 + j]
                    v1 = qp[r, p0 + 32 + j]
                    v2 = qp[r, p0 + 64 + j]
                    v3 = qp[r, p0 + 96 + j]
                    blk_i = np.int32(
                        blk_i
                        + np.int32(np.int16(s0 * np.int16(v0 & np.uint8(15)))) * np.int32(xq[x0 + j])
                        + np.int32(np.int16(s1 * np.int16(v0 >> np.uint8(4)))) * np.int32(xq[x0 + 32 + j])
                        + np.int32(np.int16(s2 * np.int16(v1 & np.uint8(15)))) * np.int32(xq[x0 + 64 + j])
                        + np.int32(np.int16(s3 * np.int16(v1 >> np.uint8(4)))) * np.int32(xq[x0 + 96 + j])
                        + np.int32(np.int16(s4 * np.int16(v2 & np.uint8(15)))) * np.int32(xq[x0 + 128 + j])
                        + np.int32(np.int16(s5 * np.int16(v2 >> np.uint8(4)))) * np.int32(xq[x0 + 160 + j])
                        + np.int32(np.int16(s6 * np.int16(v3 & np.uint8(15)))) * np.int32(xq[x0 + 192 + j])
                        + np.int32(np.int16(s7 * np.int16(v3 >> np.uint8(4)))) * np.int32(xq[x0 + 224 + j]))
                min_i = np.int32(0)
                for t in range(8):
                    min_i = np.int32(min_i + np.int32(mn[r, b, t])
                                     * bsums[b * 8 + t])
                acc += ascale[b] * (lut[dh[r, b]] * np.float32(blk_i)
                                    - lut[dmh[r, b]] * np.float32(min_i))
            out[r] = acc
        return out

    @njit(parallel=True, fastmath=True, cache=True)
    def _matvec_q5k_int(qp, qh, sc, mn, dh, dmh, lut, xq, ascale, bsums):
        # Q4_K's j-outer structure plus the fifth-bit plane: one qh byte per
        # j carries the high bit for all eight streams (bit 2c for the low
        # nibble of chunk c, bit 2c+1 for its high nibble). Codes stay
        # unsigned [0, 31]; sc*code <= 63*31 < 2^15 so the premultiply is
        # exact in int16. Measured 69.4 Gw/s at 0.70 B/weight vs the 45.7 of
        # the f32 path this replaces.
        rows = qp.shape[0]
        nblk = ascale.shape[0]
        out = np.empty(rows, np.float32)
        for r in prange(rows):
            acc = np.float32(0.0)
            for b in range(nblk):
                p0 = b * 128
                h0 = b * 32
                x0 = b * 256
                s0 = np.int16(sc[r, b, 0])
                s1 = np.int16(sc[r, b, 1])
                s2 = np.int16(sc[r, b, 2])
                s3 = np.int16(sc[r, b, 3])
                s4 = np.int16(sc[r, b, 4])
                s5 = np.int16(sc[r, b, 5])
                s6 = np.int16(sc[r, b, 6])
                s7 = np.int16(sc[r, b, 7])
                blk_i = np.int32(0)
                for j in range(32):
                    v0 = qp[r, p0 + j]
                    v1 = qp[r, p0 + 32 + j]
                    v2 = qp[r, p0 + 64 + j]
                    v3 = qp[r, p0 + 96 + j]
                    hh = qh[r, h0 + j]
                    blk_i = np.int32(
                        blk_i
                        + np.int32(np.int16(s0 * np.int16((v0 & np.uint8(15)) | ((hh & np.uint8(1)) << np.uint8(4))))) * np.int32(xq[x0 + j])
                        + np.int32(np.int16(s1 * np.int16((v0 >> np.uint8(4)) | (((hh >> np.uint8(1)) & np.uint8(1)) << np.uint8(4))))) * np.int32(xq[x0 + 32 + j])
                        + np.int32(np.int16(s2 * np.int16((v1 & np.uint8(15)) | (((hh >> np.uint8(2)) & np.uint8(1)) << np.uint8(4))))) * np.int32(xq[x0 + 64 + j])
                        + np.int32(np.int16(s3 * np.int16((v1 >> np.uint8(4)) | (((hh >> np.uint8(3)) & np.uint8(1)) << np.uint8(4))))) * np.int32(xq[x0 + 96 + j])
                        + np.int32(np.int16(s4 * np.int16((v2 & np.uint8(15)) | (((hh >> np.uint8(4)) & np.uint8(1)) << np.uint8(4))))) * np.int32(xq[x0 + 128 + j])
                        + np.int32(np.int16(s5 * np.int16((v2 >> np.uint8(4)) | (((hh >> np.uint8(5)) & np.uint8(1)) << np.uint8(4))))) * np.int32(xq[x0 + 160 + j])
                        + np.int32(np.int16(s6 * np.int16((v3 & np.uint8(15)) | (((hh >> np.uint8(6)) & np.uint8(1)) << np.uint8(4))))) * np.int32(xq[x0 + 192 + j])
                        + np.int32(np.int16(s7 * np.int16((v3 >> np.uint8(4)) | ((hh >> np.uint8(7)) << np.uint8(4))))) * np.int32(xq[x0 + 224 + j]))
                min_i = np.int32(0)
                for t in range(8):
                    min_i = np.int32(min_i + np.int32(mn[r, b, t])
                                     * bsums[b * 8 + t])
                acc += ascale[b] * (lut[dh[r, b]] * np.float32(blk_i)
                                    - lut[dmh[r, b]] * np.float32(min_i))
            out[r] = acc
        return out

    @njit(parallel=True, fastmath=True, cache=True)
    def _dequant_q5k(qp, qh, sc, mn, dh, dmh, lut, out):
        # expand native Q5_K rows to float32 in one parallel pass (the
        # prefill GEMM and row-gather paths need f32 tiles)
        nr = qp.shape[0]
        nblk = dh.shape[1]
        for r in prange(nr):
            for b in range(nblk):
                d = lut[dh[r, b]]
                dm = lut[dmh[r, b]]
                for c in range(4):
                    d0 = d * np.float32(sc[r, b, 2 * c])
                    d1 = d * np.float32(sc[r, b, 2 * c + 1])
                    m0 = dm * np.float32(mn[r, b, 2 * c])
                    m1 = dm * np.float32(mn[r, b, 2 * c + 1])
                    pb = b * 128 + c * 32
                    hb = b * 32
                    ob = b * 256 + c * 64
                    lo_sh = np.uint8(2 * c)
                    hi_sh = np.uint8(2 * c + 1)
                    for j in range(32):
                        v = qp[r, pb + j]
                        hh = qh[r, hb + j]
                        e0 = np.float32((v & np.uint8(15))
                                        | (((hh >> lo_sh) & np.uint8(1)) << np.uint8(4)))
                        e1 = np.float32((v >> np.uint8(4))
                                        | (((hh >> hi_sh) & np.uint8(1)) << np.uint8(4)))
                        out[r, ob + j] = d0 * e0 - m0
                        out[r, ob + 32 + j] = d1 * e1 - m1

    @njit(parallel=True, fastmath=True, cache=True)
    def _matvec_q6k_int(qf, sc, dh, lut, xq, ascale):
        # qf: (rows, cols) int8 codes already centered to [-32, 31], so
        # there is no min term; sc is the per-16 int8 scale and d the
        # per-256-block f16. All 16 sub-scales of a block are premultiplied
        # into the int16 codes (|sc*code| <= 127*32 < 2^15) so the block
        # accumulates in two vector chains and reduces once per 256 weights;
        # per-sub-block reduces measured 33.8 Gw/s, this shape 52.2 - 94% of
        # the 1.066 B/weight bandwidth wall.
        rows = qf.shape[0]
        nblk = ascale.shape[0]
        out = np.empty(rows, np.float32)
        for r in prange(rows):
            acc = np.float32(0.0)
            for b in range(nblk):
                e0 = b * 256
                t0 = np.int16(sc[r, b * 16 + 0])
                t1 = np.int16(sc[r, b * 16 + 1])
                t2 = np.int16(sc[r, b * 16 + 2])
                t3 = np.int16(sc[r, b * 16 + 3])
                t4 = np.int16(sc[r, b * 16 + 4])
                t5 = np.int16(sc[r, b * 16 + 5])
                t6 = np.int16(sc[r, b * 16 + 6])
                t7 = np.int16(sc[r, b * 16 + 7])
                t8 = np.int16(sc[r, b * 16 + 8])
                t9 = np.int16(sc[r, b * 16 + 9])
                t10 = np.int16(sc[r, b * 16 + 10])
                t11 = np.int16(sc[r, b * 16 + 11])
                t12 = np.int16(sc[r, b * 16 + 12])
                t13 = np.int16(sc[r, b * 16 + 13])
                t14 = np.int16(sc[r, b * 16 + 14])
                t15 = np.int16(sc[r, b * 16 + 15])
                a0 = np.int32(0)
                a1 = np.int32(0)
                for j in range(16):
                    a0 = np.int32(
                        a0
                        + np.int32(np.int16(t0 * np.int16(qf[r, e0 + j]))) * np.int32(xq[e0 + j])
                        + np.int32(np.int16(t1 * np.int16(qf[r, e0 + 16 + j]))) * np.int32(xq[e0 + 16 + j])
                        + np.int32(np.int16(t2 * np.int16(qf[r, e0 + 32 + j]))) * np.int32(xq[e0 + 32 + j])
                        + np.int32(np.int16(t3 * np.int16(qf[r, e0 + 48 + j]))) * np.int32(xq[e0 + 48 + j])
                        + np.int32(np.int16(t4 * np.int16(qf[r, e0 + 64 + j]))) * np.int32(xq[e0 + 64 + j])
                        + np.int32(np.int16(t5 * np.int16(qf[r, e0 + 80 + j]))) * np.int32(xq[e0 + 80 + j])
                        + np.int32(np.int16(t6 * np.int16(qf[r, e0 + 96 + j]))) * np.int32(xq[e0 + 96 + j])
                        + np.int32(np.int16(t7 * np.int16(qf[r, e0 + 112 + j]))) * np.int32(xq[e0 + 112 + j]))
                    a1 = np.int32(
                        a1
                        + np.int32(np.int16(t8 * np.int16(qf[r, e0 + 128 + j]))) * np.int32(xq[e0 + 128 + j])
                        + np.int32(np.int16(t9 * np.int16(qf[r, e0 + 144 + j]))) * np.int32(xq[e0 + 144 + j])
                        + np.int32(np.int16(t10 * np.int16(qf[r, e0 + 160 + j]))) * np.int32(xq[e0 + 160 + j])
                        + np.int32(np.int16(t11 * np.int16(qf[r, e0 + 176 + j]))) * np.int32(xq[e0 + 176 + j])
                        + np.int32(np.int16(t12 * np.int16(qf[r, e0 + 192 + j]))) * np.int32(xq[e0 + 192 + j])
                        + np.int32(np.int16(t13 * np.int16(qf[r, e0 + 208 + j]))) * np.int32(xq[e0 + 208 + j])
                        + np.int32(np.int16(t14 * np.int16(qf[r, e0 + 224 + j]))) * np.int32(xq[e0 + 224 + j])
                        + np.int32(np.int16(t15 * np.int16(qf[r, e0 + 240 + j]))) * np.int32(xq[e0 + 240 + j]))
                acc += ascale[b] * lut[dh[r, b]] * np.float32(a0 + a1)
            out[r] = acc
        return out

    @njit(parallel=True, fastmath=True, cache=True)
    def _matvec_q4k_q6k_pair(qp, sc4, mn4, dh4, dmh4, q6, sc6, dh6,
                             lut, xq, ascale, bsums, out4, out6):
        # PERFORMANCE_PLAN Package D: two matrices that read the SAME input
        # vector, computed in ONE parallel region.
        #
        # Same-dtype neighbours are already concatenated at load (see
        # model.fused_mat), which is strictly better - one contiguous
        # stream. What that cannot do is join a Q4_K to a Q6_K, and a
        # Q4_K_M file produces exactly that pair: attn_q/attn_k are Q4_K
        # while attn_v is Q6_K. This kernel joins them anyway, by ranging
        # one prange over the COMBINED row space. Two wins, not one: a
        # single fan-out instead of two, and the narrow matrix's rows are
        # spread across the pool with the wide one's instead of being cut
        # into slivers of its own.
        #
        # Each branch is the body of _matvec_q4k_int / _matvec_q6k_int
        # unchanged, so a row's accumulation order is the same one the
        # per-matrix kernels use.
        rows4 = qp.shape[0]
        rows6 = q6.shape[0]
        nblk = ascale.shape[0]
        for r in prange(rows4 + rows6):
            if r < rows4:
                acc = np.float32(0.0)
                for b in range(nblk):
                    p0 = b * 128
                    x0 = b * 256
                    s0 = np.int16(sc4[r, b, 0])
                    s1 = np.int16(sc4[r, b, 1])
                    s2 = np.int16(sc4[r, b, 2])
                    s3 = np.int16(sc4[r, b, 3])
                    s4 = np.int16(sc4[r, b, 4])
                    s5 = np.int16(sc4[r, b, 5])
                    s6 = np.int16(sc4[r, b, 6])
                    s7 = np.int16(sc4[r, b, 7])
                    blk_i = np.int32(0)
                    for j in range(32):
                        v0 = qp[r, p0 + j]
                        v1 = qp[r, p0 + 32 + j]
                        v2 = qp[r, p0 + 64 + j]
                        v3 = qp[r, p0 + 96 + j]
                        blk_i = np.int32(
                            blk_i
                            + np.int32(np.int16(s0 * np.int16(v0 & np.uint8(15)))) * np.int32(xq[x0 + j])
                            + np.int32(np.int16(s1 * np.int16(v0 >> np.uint8(4)))) * np.int32(xq[x0 + 32 + j])
                            + np.int32(np.int16(s2 * np.int16(v1 & np.uint8(15)))) * np.int32(xq[x0 + 64 + j])
                            + np.int32(np.int16(s3 * np.int16(v1 >> np.uint8(4)))) * np.int32(xq[x0 + 96 + j])
                            + np.int32(np.int16(s4 * np.int16(v2 & np.uint8(15)))) * np.int32(xq[x0 + 128 + j])
                            + np.int32(np.int16(s5 * np.int16(v2 >> np.uint8(4)))) * np.int32(xq[x0 + 160 + j])
                            + np.int32(np.int16(s6 * np.int16(v3 & np.uint8(15)))) * np.int32(xq[x0 + 192 + j])
                            + np.int32(np.int16(s7 * np.int16(v3 >> np.uint8(4)))) * np.int32(xq[x0 + 224 + j]))
                    min_i = np.int32(0)
                    for t in range(8):
                        min_i = np.int32(min_i + np.int32(mn4[r, b, t])
                                         * bsums[b * 8 + t])
                    acc += ascale[b] * (lut[dh4[r, b]] * np.float32(blk_i)
                                        - lut[dmh4[r, b]] * np.float32(min_i))
                out4[r] = acc
            else:
                rb = r - rows4
                acc = np.float32(0.0)
                for b in range(nblk):
                    e0 = b * 256
                    t0 = np.int16(sc6[rb, b * 16 + 0])
                    t1 = np.int16(sc6[rb, b * 16 + 1])
                    t2 = np.int16(sc6[rb, b * 16 + 2])
                    t3 = np.int16(sc6[rb, b * 16 + 3])
                    t4 = np.int16(sc6[rb, b * 16 + 4])
                    t5 = np.int16(sc6[rb, b * 16 + 5])
                    t6 = np.int16(sc6[rb, b * 16 + 6])
                    t7 = np.int16(sc6[rb, b * 16 + 7])
                    t8 = np.int16(sc6[rb, b * 16 + 8])
                    t9 = np.int16(sc6[rb, b * 16 + 9])
                    t10 = np.int16(sc6[rb, b * 16 + 10])
                    t11 = np.int16(sc6[rb, b * 16 + 11])
                    t12 = np.int16(sc6[rb, b * 16 + 12])
                    t13 = np.int16(sc6[rb, b * 16 + 13])
                    t14 = np.int16(sc6[rb, b * 16 + 14])
                    t15 = np.int16(sc6[rb, b * 16 + 15])
                    a0 = np.int32(0)
                    a1 = np.int32(0)
                    for j in range(16):
                        a0 = np.int32(
                            a0
                            + np.int32(np.int16(t0 * np.int16(q6[rb, e0 + j]))) * np.int32(xq[e0 + j])
                            + np.int32(np.int16(t1 * np.int16(q6[rb, e0 + 16 + j]))) * np.int32(xq[e0 + 16 + j])
                            + np.int32(np.int16(t2 * np.int16(q6[rb, e0 + 32 + j]))) * np.int32(xq[e0 + 32 + j])
                            + np.int32(np.int16(t3 * np.int16(q6[rb, e0 + 48 + j]))) * np.int32(xq[e0 + 48 + j])
                            + np.int32(np.int16(t4 * np.int16(q6[rb, e0 + 64 + j]))) * np.int32(xq[e0 + 64 + j])
                            + np.int32(np.int16(t5 * np.int16(q6[rb, e0 + 80 + j]))) * np.int32(xq[e0 + 80 + j])
                            + np.int32(np.int16(t6 * np.int16(q6[rb, e0 + 96 + j]))) * np.int32(xq[e0 + 96 + j])
                            + np.int32(np.int16(t7 * np.int16(q6[rb, e0 + 112 + j]))) * np.int32(xq[e0 + 112 + j]))
                        a1 = np.int32(
                            a1
                            + np.int32(np.int16(t8 * np.int16(q6[rb, e0 + 128 + j]))) * np.int32(xq[e0 + 128 + j])
                            + np.int32(np.int16(t9 * np.int16(q6[rb, e0 + 144 + j]))) * np.int32(xq[e0 + 144 + j])
                            + np.int32(np.int16(t10 * np.int16(q6[rb, e0 + 160 + j]))) * np.int32(xq[e0 + 160 + j])
                            + np.int32(np.int16(t11 * np.int16(q6[rb, e0 + 176 + j]))) * np.int32(xq[e0 + 176 + j])
                            + np.int32(np.int16(t12 * np.int16(q6[rb, e0 + 192 + j]))) * np.int32(xq[e0 + 192 + j])
                            + np.int32(np.int16(t13 * np.int16(q6[rb, e0 + 208 + j]))) * np.int32(xq[e0 + 208 + j])
                            + np.int32(np.int16(t14 * np.int16(q6[rb, e0 + 224 + j]))) * np.int32(xq[e0 + 224 + j])
                            + np.int32(np.int16(t15 * np.int16(q6[rb, e0 + 240 + j]))) * np.int32(xq[e0 + 240 + j]))
                    acc += ascale[b] * lut[dh6[rb, b]] * np.float32(a0 + a1)
                out6[rb] = acc

    @njit(parallel=True, fastmath=True, cache=True)
    def _matvec_q4k_q5k_pair(qp, sc4, mn4, dh4, dmh4,
                             qp5, qh5, sc5, mn5, dh5, dmh5,
                             lut, xq, ascale, bsums, out4, out5):
        # Package K: the same one-parallel-region trick as the Q4_K+Q6_K
        # pair above, for the pairing a **Q4_K_S** file produces. Q4_K_S
        # stores attn_v as Q5_K (03-RESULTS round 5 measured that census),
        # so a Q4_K_S model's attention group is Q4_K q/k beside a Q5_K v
        # and fell through to per-matrix dispatch, getting nothing.
        #
        # Each branch is the body of _matvec_q4k_int / _matvec_q5k_int
        # unchanged - same j-outer walk, same premultiplied int16 codes,
        # same one reduce per 256 weights - so a row's accumulation order is
        # the one the per-matrix kernels use and the result is bit-identical
        # by construction rather than by inspection.
        rows4 = qp.shape[0]
        rows5 = qp5.shape[0]
        nblk = ascale.shape[0]
        for r in prange(rows4 + rows5):
            if r < rows4:
                acc = np.float32(0.0)
                for b in range(nblk):
                    p0 = b * 128
                    x0 = b * 256
                    s0 = np.int16(sc4[r, b, 0])
                    s1 = np.int16(sc4[r, b, 1])
                    s2 = np.int16(sc4[r, b, 2])
                    s3 = np.int16(sc4[r, b, 3])
                    s4 = np.int16(sc4[r, b, 4])
                    s5 = np.int16(sc4[r, b, 5])
                    s6 = np.int16(sc4[r, b, 6])
                    s7 = np.int16(sc4[r, b, 7])
                    blk_i = np.int32(0)
                    for j in range(32):
                        v0 = qp[r, p0 + j]
                        v1 = qp[r, p0 + 32 + j]
                        v2 = qp[r, p0 + 64 + j]
                        v3 = qp[r, p0 + 96 + j]
                        blk_i = np.int32(
                            blk_i
                            + np.int32(np.int16(s0 * np.int16(v0 & np.uint8(15)))) * np.int32(xq[x0 + j])
                            + np.int32(np.int16(s1 * np.int16(v0 >> np.uint8(4)))) * np.int32(xq[x0 + 32 + j])
                            + np.int32(np.int16(s2 * np.int16(v1 & np.uint8(15)))) * np.int32(xq[x0 + 64 + j])
                            + np.int32(np.int16(s3 * np.int16(v1 >> np.uint8(4)))) * np.int32(xq[x0 + 96 + j])
                            + np.int32(np.int16(s4 * np.int16(v2 & np.uint8(15)))) * np.int32(xq[x0 + 128 + j])
                            + np.int32(np.int16(s5 * np.int16(v2 >> np.uint8(4)))) * np.int32(xq[x0 + 160 + j])
                            + np.int32(np.int16(s6 * np.int16(v3 & np.uint8(15)))) * np.int32(xq[x0 + 192 + j])
                            + np.int32(np.int16(s7 * np.int16(v3 >> np.uint8(4)))) * np.int32(xq[x0 + 224 + j]))
                    min_i = np.int32(0)
                    for t in range(8):
                        min_i = np.int32(min_i + np.int32(mn4[r, b, t])
                                         * bsums[b * 8 + t])
                    acc += ascale[b] * (lut[dh4[r, b]] * np.float32(blk_i)
                                        - lut[dmh4[r, b]] * np.float32(min_i))
                out4[r] = acc
            else:
                rb = r - rows4
                acc = np.float32(0.0)
                for b in range(nblk):
                    p0 = b * 128
                    h0 = b * 32
                    x0 = b * 256
                    s0 = np.int16(sc5[rb, b, 0])
                    s1 = np.int16(sc5[rb, b, 1])
                    s2 = np.int16(sc5[rb, b, 2])
                    s3 = np.int16(sc5[rb, b, 3])
                    s4 = np.int16(sc5[rb, b, 4])
                    s5 = np.int16(sc5[rb, b, 5])
                    s6 = np.int16(sc5[rb, b, 6])
                    s7 = np.int16(sc5[rb, b, 7])
                    blk_i = np.int32(0)
                    for j in range(32):
                        v0 = qp5[rb, p0 + j]
                        v1 = qp5[rb, p0 + 32 + j]
                        v2 = qp5[rb, p0 + 64 + j]
                        v3 = qp5[rb, p0 + 96 + j]
                        hh = qh5[rb, h0 + j]
                        blk_i = np.int32(
                            blk_i
                            + np.int32(np.int16(s0 * np.int16((v0 & np.uint8(15)) | ((hh & np.uint8(1)) << np.uint8(4))))) * np.int32(xq[x0 + j])
                            + np.int32(np.int16(s1 * np.int16((v0 >> np.uint8(4)) | (((hh >> np.uint8(1)) & np.uint8(1)) << np.uint8(4))))) * np.int32(xq[x0 + 32 + j])
                            + np.int32(np.int16(s2 * np.int16((v1 & np.uint8(15)) | (((hh >> np.uint8(2)) & np.uint8(1)) << np.uint8(4))))) * np.int32(xq[x0 + 64 + j])
                            + np.int32(np.int16(s3 * np.int16((v1 >> np.uint8(4)) | (((hh >> np.uint8(3)) & np.uint8(1)) << np.uint8(4))))) * np.int32(xq[x0 + 96 + j])
                            + np.int32(np.int16(s4 * np.int16((v2 & np.uint8(15)) | (((hh >> np.uint8(4)) & np.uint8(1)) << np.uint8(4))))) * np.int32(xq[x0 + 128 + j])
                            + np.int32(np.int16(s5 * np.int16((v2 >> np.uint8(4)) | (((hh >> np.uint8(5)) & np.uint8(1)) << np.uint8(4))))) * np.int32(xq[x0 + 160 + j])
                            + np.int32(np.int16(s6 * np.int16((v3 & np.uint8(15)) | (((hh >> np.uint8(6)) & np.uint8(1)) << np.uint8(4))))) * np.int32(xq[x0 + 192 + j])
                            + np.int32(np.int16(s7 * np.int16((v3 >> np.uint8(4)) | ((hh >> np.uint8(7)) << np.uint8(4))))) * np.int32(xq[x0 + 224 + j]))
                    min_i = np.int32(0)
                    for t in range(8):
                        min_i = np.int32(min_i + np.int32(mn5[rb, b, t])
                                         * bsums[b * 8 + t])
                    acc += ascale[b] * (lut[dh5[rb, b]] * np.float32(blk_i)
                                        - lut[dmh5[rb, b]] * np.float32(min_i))
                out5[rb] = acc

    @njit(parallel=True, fastmath=True, cache=True)
    def _attention_decode(q3, K, V, scores, out, inv_sqrt):
        # Single-token grouped-query attention over the KV cache, one prange
        # worker per query head, softmax fused in. This exists because the
        # NumPy path's np.matmul enters OpenBLAS, whose OWN thread pool fans
        # out once the context passes its size threshold (~600 tokens) - 64
        # pool fan-outs per token fighting the kernels' 6 omp threads took
        # decode from 124 to 800-2260 ms/token. This kernel runs on the same
        # omp pool as the matvecs, so there is nothing to fight.
        #   q3 (n_kv, group, hd); K, V (t, n_kv, hd); scores (heads, t)
        t = K.shape[0]
        group = q3.shape[1]
        hd = q3.shape[2]
        heads = q3.shape[0] * group
        for h in prange(heads):
            kv = h // group
            g = h % group
            srow = scores[h]
            for i in range(t):
                dot = np.float32(0.0)
                for d in range(hd):
                    dot += q3[kv, g, d] * K[i, kv, d]
                srow[i] = dot * inv_sqrt
            # seed from the first score, not a constant: a finite score can
            # sit below any magic seed and a stuck seed zeroes the softmax
            m = srow[0]
            for i in range(1, t):
                if srow[i] > m:
                    m = srow[i]
            ssum = np.float32(0.0)
            for i in range(t):
                e = np.exp(srow[i] - m)
                srow[i] = e
                ssum += e
            inv = np.float32(1.0) / ssum
            acc = out[kv, g]
            for d in range(hd):
                acc[d] = np.float32(0.0)
            for i in range(t):
                w = srow[i] * inv
                vrow = V[i, kv]
                for d in range(hd):
                    acc[d] += w * vrow[d]

    @njit(parallel=True, fastmath=True, cache=True)
    def _dequant_q4k(qp, sc, mn, dh, dmh, lut, out):
        # expand native Q4_K rows to float32 in one parallel pass; the
        # prefill GEMM path needs f32 tiles and NumPy's multi-pass nibble
        # unpack measured ~3x slower than this
        nr = qp.shape[0]
        nblk = dh.shape[1]
        for r in prange(nr):
            for b in range(nblk):
                d = lut[dh[r, b]]
                dm = lut[dmh[r, b]]
                for c in range(4):
                    d0 = d * np.float32(sc[r, b, 2 * c])
                    d1 = d * np.float32(sc[r, b, 2 * c + 1])
                    m0 = dm * np.float32(mn[r, b, 2 * c])
                    m1 = dm * np.float32(mn[r, b, 2 * c + 1])
                    pb = b * 128 + c * 32
                    ob = b * 256 + c * 64
                    for j in range(32):
                        v = qp[r, pb + j]
                        out[r, ob + j] = (d0 * np.float32(v & np.uint8(15))
                                          - m0)
                        out[r, ob + 32 + j] = (d1 * np.float32(v >> np.uint8(4))
                                               - m1)

    @njit(parallel=True, fastmath=True, cache=True)
    def _dequant_q6k(qf, sc, dh, lut, out):
        # qf: (nr, cols) int8 codes in element order, sc per 16, d per 256
        nr = qf.shape[0]
        nsub = sc.shape[1]
        for r in prange(nr):
            for su in range(nsub):
                d_eff = lut[dh[r, su // 16]] * np.float32(sc[r, su])
                e0 = su * 16
                for j in range(16):
                    out[r, e0 + j] = d_eff * np.float32(qf[r, e0 + j])

    @njit(cache=True)
    def _rope_norm(v, cos, sin, n_heads, hd, n_rot, out):
        # adjacent-pair rotation (llama/mistral). fastmath stays OFF: with
        # strict FP semantics LLVM may not contract mul+sub into FMA, so
        # this is BIT-IDENTICAL to the NumPy elementwise path it replaces -
        # the pinned logit tests cannot drift.
        half = n_rot // 2
        for h in range(n_heads):
            base = h * hd
            for i in range(half):
                a = base + 2 * i
                b = a + 1
                x0 = v[a]
                x1 = v[b]
                out[a] = x0 * cos[i] - x1 * sin[i]
                out[b] = x0 * sin[i] + x1 * cos[i]
            for i in range(n_rot, hd):
                out[base + i] = v[base + i]

    @njit(cache=True)
    def _rope_neox(v, cos, sin, n_heads, hd, n_rot, out):
        # split-half rotation (qwen/gemma), same strict-FP contract
        half = n_rot // 2
        for h in range(n_heads):
            base = h * hd
            for i in range(half):
                a = base + i
                b = base + half + i
                x0 = v[a]
                x1 = v[b]
                out[a] = x0 * cos[i] - x1 * sin[i]
                out[b] = x0 * sin[i] + x1 * cos[i]
            for i in range(n_rot, hd):
                out[base + i] = v[base + i]

    lut = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)

    global _cur_threads, _max_threads
    _cur_threads = _max_threads = int(numba.get_num_threads())

    _state = {"np": np, "numba": numba,
              "matvec": _matvec_codes, "matmul": _matmul_codes,
              "matmul_wide": _matmul_codes_wide,
              "quantize_acts": _quantize_acts,
              "matvec_q4k_q6k_pair": _matvec_q4k_q6k_pair,
              "matvec_q4k_q5k_pair": _matvec_q4k_q5k_pair,
              "matvec_q4k_int": _matvec_q4k_int,
              "matvec_q5k_int": _matvec_q5k_int,
              "matvec_q6k_int": _matvec_q6k_int,
              "dequant_q4k": _dequant_q4k,
              "dequant_q5k": _dequant_q5k,
              "dequant_q6k": _dequant_q6k,
              "attention_decode": _attention_decode,
              "rope_norm": _rope_norm, "rope_neox": _rope_neox,
              "f16_lut": lut,
              "numba_version": numba.__version__}
    return _state


def available() -> bool:
    """True when the pinned JIT is importable and kernels are enabled."""
    return bool(_init())


# ---- per-dispatch thread scaling --------------------------------------------
#
# One compiled kernel, run on a chosen number of threads - NOT a second
# hand-written serial copy of the same unrolled arithmetic. numba's prange
# splits the ROW loop, and each row's accumulator is computed by the same
# instruction sequence regardless of how many threads share the loop, so
# narrowing the fan-out is bit-identical by construction. A duplicated
# kernel would only be bit-identical by inspection, and would have to stay
# that way through every future edit.

def _set_threads(st: dict, n: int) -> None:
    # numba.set_num_threads measured at 3.3 us on the reference machine -
    # cheap, but not free against a 49 us matvec. Skipping the call when the
    # pool is already the right size is what makes this affordable, and it
    # is why the count is tracked here rather than read back per dispatch.
    global _cur_threads
    if n == _cur_threads:
        return
    try:
        st["numba"].set_num_threads(n)
        _cur_threads = n
    except (ValueError, RuntimeError):
        pass


def _want(st: dict, work_elements: int) -> None:
    """Size the pool for a dispatch touching `work_elements` weights.

    Set-and-leave, not set-and-restore: restoring would double the number
    of set_num_threads calls, and there is nothing to restore *to* - every
    kernel entry point in this module declares what it needs before it
    runs, so the pool is always right for the dispatch about to happen.
    A model whose shapes all sit on one side of the threshold - the common
    case - therefore pays nothing at all.
    """
    if _max_threads <= 1:
        return
    _set_threads(st, 1 if work_elements <= serial_matvec_elems()
                 else _max_threads)


def _want_full(st: dict) -> None:
    """Full pool: for the kernels that are always worth fanning out (batched
    matmul, prefill dequantize, decode attention over a long context)."""
    if _max_threads > 1:
        _set_threads(st, _max_threads)


def set_threads(n: int) -> int:
    """Resize the kernel thread pool, keeping the dispatch bookkeeping in
    sync. Callers must use this rather than numba.set_num_threads directly,
    or _want will believe the pool is a size it is not."""
    global _max_threads
    st = _init()
    if not st:
        return 0
    _set_threads(st, max(1, int(n)))
    _max_threads = _cur_threads
    return _cur_threads


def dispatch_threads(work_elements: int) -> int:
    """How many threads a matvec of this size will use. For the profiler and
    for tests that pin the dispatch DECISION rather than its timing."""
    if not _init():
        return 0
    if _max_threads <= 1:
        return _max_threads
    return 1 if work_elements <= serial_matvec_elems() else _max_threads


def kernel_version() -> str:
    """`numba==X` when the JIT is live, `off` otherwise - for backend labels."""
    st = _init()
    return f"numba=={st['numba_version']}" if st else "off"


def threads() -> int:
    """The kernel pool's configured size, or 0 when the kernels are inactive.

    The pool SIZE, not numba.get_num_threads(): per-dispatch scaling leaves
    that at 1 whenever the last kernel to run was a narrow one, and a
    profiler reporting "1 thread" for a machine tuned to 8 would be lying.
    """
    if not _init():
        return 0
    return _max_threads


def status() -> str:
    st = _init()
    if st:
        return (f"alpaccaroo-kernels active (numba=={st['numba_version']}, "
                f"pin {NUMBA_PIN}, our Python source)")
    return "alpaccaroo-kernels inactive (pure/NumPy paths in use)"


def matvec_codes(q3, d_eff, m_eff, x):
    """Fused quantized matvec over int8 codes + per-sub-block scales.

    q3: int8 (rows, n_sub, sub_len) C-contiguous, element order.
    d_eff/m_eff: float32 (rows, n_sub); m_eff may be None.
    x: float32 (cols,). Returns float32 (rows,).
    """
    st = _init()
    np = st["np"]
    rows, nsub, sub_len = q3.shape
    xs = np.ascontiguousarray(x, dtype=np.float32).reshape(nsub, sub_len)
    _want(st, rows * nsub * sub_len)
    if m_eff is None:
        xsums = xs[:1, :1].reshape(1)  # unused dummy
        m_arg, affine = d_eff, False
    else:
        xsums = xs.sum(axis=1)
        m_arg, affine = m_eff, True
    return st["matvec"](q3, d_eff, m_arg, xs, xsums, affine)


def matmul_codes(q3, d_eff, m_eff, X):
    """Fused quantized matmul: X (batch, cols) @ self.T -> (batch, rows).

    Same layout contract as :func:`matvec_codes`, with X float32
    (batch, cols). Weights are streamed once for the whole batch.
    """
    st = _init()
    np = st["np"]
    _want_full(st)   # a batch always has enough work to repay the fan-out
    rows, nsub, sub_len = q3.shape
    batch = X.shape[0]
    Xs = np.ascontiguousarray(X, dtype=np.float32).reshape(batch, nsub, sub_len)
    affine = m_eff is not None
    m_arg = m_eff if affine else d_eff  # unused when not affine
    if batch <= NARROW_BATCH:
        # few columns: keep the batch in the outer loop, so no per-row
        # scratch buffer is allocated and one-token batches stay ~1.5 ms
        xsums = Xs.sum(axis=2) if affine else Xs[:, :1, 0]
        return st["matmul"](q3, d_eff, m_arg, Xs, xsums, affine)
    Xt = np.ascontiguousarray(Xs.transpose(1, 2, 0))
    XsumsT = (np.ascontiguousarray(Xs.sum(axis=2).T) if affine
              else Xt[:, :1, 0])
    wide = st["matmul_wide"](q3, d_eff, m_arg, Xt, XsumsT, affine)
    # the wide kernel accumulates into (rows, batch) so its innermost loop
    # stays contiguous; every other path returns a C-contiguous (batch, rows),
    # so pay the transpose rather than hand back a view that behaves
    # differently. Measured at 1.4% of the matmul it follows.
    return np.ascontiguousarray(wide.T)


def quantize_acts(x):
    """Quantize a float32 activation vector for the integer-dot kernels.

    Returns (xq int8 (cols,), ascale f32 (cols//256,), bsums i32 (cols//32,))
    with a per-256-block scale of absmax/127 and per-32 sums of the
    quantized values. `cols` must be a multiple of 256.
    """
    st = _init()
    np = st["np"]
    xc = np.ascontiguousarray(x, dtype=np.float32)
    # a 2048-wide model gives this kernel eight 256-element blocks to
    # divide; the fan-out costs more than the work. Same reasoning as
    # _dispatch, sized in columns because that is what the loop trips on.
    limit = _env_int("ALPACCAROO_SERIAL_QUANTIZE_COLS",
                     SERIAL_QUANTIZE_COLS_DEFAULT)
    # 0 forces the serial side of _want's test, 2**62 forces the pool side;
    # this kernel's threshold is in columns, not weights, so it cannot share
    # the matvec knob
    _want(st, 0 if xc.shape[0] <= limit else (1 << 62))
    return st["quantize_acts"](xc)


def matvec_q4k_int(qp, sc, mn, dh, dmh, x, pre=None):
    """Fused Q4_K matvec over the native block fields with an int8-activation
    integer dot. qp u8 (rows, cols//2) split-nibble codes; sc/mn u8
    (rows, nblk, 8); dh/dmh u16 f16-bits (rows, nblk); x f32 (cols,).
    `pre` is an optional quantize_acts(x) result so matrices sharing an
    input (attn qk and v read the same normed vector) quantize it once."""
    st = _init()
    xq, ascale, bsums = pre if pre is not None else quantize_acts(x)
    _want(st, qp.shape[0] * qp.shape[1] * 2)   # two 4-bit codes per byte
    return st["matvec_q4k_int"](qp, sc, mn, dh, dmh, st["f16_lut"],
                                xq, ascale, bsums)


def matvec_q5k_int(qp, qh, sc, mn, dh, dmh, x, pre=None):
    """Fused Q5_K matvec over the native block fields: qp u8 (rows, cols//2)
    split-nibble low bits, qh u8 (rows, cols//8) fifth bits, sc/mn u8
    (rows, nblk, 8), dh/dmh u16 f16-bits (rows, nblk)."""
    st = _init()
    xq, ascale, bsums = pre if pre is not None else quantize_acts(x)
    _want(st, qp.shape[0] * qp.shape[1] * 2)
    return st["matvec_q5k_int"](qp, qh, sc, mn, dh, dmh, st["f16_lut"],
                                xq, ascale, bsums)


#: SIMD patterns worth looking for in a kernel's generated code, and what
#: their presence or absence means. The integer kernels' whole premise is
#: that LLVM contracts the int32 re-cast idiom into 16-bit multiply-add;
#: if it does not on some host, that is the first thing to know.
ASM_MARKERS = (
    ("vpdpwssd", "AVX-512 VNNI int16 dot-product accumulate - the best case"),
    ("vpdpbusd", "AVX-512/AVX VNNI int8 dot-product accumulate"),
    ("vpmaddwd", "packed int16 multiply-add - the fallback the kernels rely on"),
    ("vpmullw", "packed int16 multiply without the add - partial contraction"),
    ("vfmadd", "float FMA - fine for the f32-scale kernels, wrong for the int ones"),
    ("vpmuldq", "64-bit lane multiply - the int64 promotion the kernels exist to avoid"),
    ("zmm", "512-bit registers"),
    ("ymm", "256-bit registers"),
)


def kernel_names() -> list:
    """Compiled kernels available to :func:`inspect_asm`."""
    st = _init()
    if not st:
        return []
    return sorted(k for k, v in st.items() if hasattr(v, "inspect_asm"))


def inspect_asm(name: str) -> dict:
    """Host assembly for one compiled kernel, plus a SIMD-pattern census.

    PERFORMANCE_PLAN Track 4 asks what instructions these Python loops
    actually become on a given CPU. Numba knows; this surfaces it without
    a build step, so a future engineer on AMD Zen, on a machine without
    AVX-512, or on Linux rather than Windows can check whether the integer
    contraction the kernels depend on survived - rather than assume it did.

    Returns {"name", "signatures", "counts", "asm"} or {"error"}.
    """
    st = _init()
    if not st:
        return {"error": "kernels inactive"}
    fn = st.get(name)
    if fn is None or not hasattr(fn, "inspect_asm"):
        return {"error": f"no compiled kernel named {name!r}; "
                         f"try one of {', '.join(kernel_names())}"}
    try:
        asm_map = fn.inspect_asm()
    except Exception as e:  # pragma: no cover - numba internals
        return {"error": f"{type(e).__name__}: {e}"}
    if not asm_map:
        return {"error": f"{name} has not been compiled yet - "
                         f"call kernels.warmup() first"}
    text = "\n".join(str(v) for v in asm_map.values())
    counts = {marker: text.count(marker) for marker, _why in ASM_MARKERS}
    return {"name": name,
            "signatures": [str(k) for k in asm_map],
            "counts": counts,
            "asm": text}


def group_kernel_enabled() -> bool:
    """Whether the grouped Q4_K+Q6_K kernel is used (Package D).

    Default on; ALPACCAROO_GROUP_KERNEL=0 reverts to one dispatch per
    matrix. It is a flag rather than a permanent choice because the win is
    a saved fan-out plus a better row split, and both depend on the pool
    size and the shapes - a machine where it does not pay must be able to
    turn it off without turning off the kernels.
    """
    return (os.environ.get("ALPACCAROO_GROUP_KERNEL", "").strip().lower()
            not in ("0", "off", "no"))


def matvec_q4k_q6k_pair(a, b, x, pre=None):
    """One dispatch for a native Q4_K matrix `a` and a native Q6_K matrix
    `b` that share the input `x`.

    Returns (out_a, out_b) float32. Both must have the same column count -
    they are reading the same vector - which the caller has already checked.
    """
    st = _init()
    np = st["np"]
    xq, ascale, bsums = pre if pre is not None else quantize_acts(x)
    rows_a, rows_b = a.rows, b.rows
    _want(st, rows_a * a.cols + rows_b * b.cols)
    out_a = np.empty(rows_a, np.float32)
    out_b = np.empty(rows_b, np.float32)
    st["matvec_q4k_q6k_pair"](a._qp, a._sci, a._mni, a._dh, a._dmh,
                              b._q3.reshape(rows_b, -1), b._sci, b._dh,
                              st["f16_lut"], xq, ascale, bsums, out_a, out_b)
    return out_a, out_b


def matvec_q4k_q5k_pair(a, b, x, pre=None):
    """One dispatch for a native Q4_K matrix `a` and a native Q5_K matrix
    `b` that share the input `x` - the pairing a Q4_K_S file produces.

    Returns (out_a, out_b) float32. Both must have the same column count -
    they are reading the same vector - which the caller has already checked.
    """
    st = _init()
    np = st["np"]
    xq, ascale, bsums = pre if pre is not None else quantize_acts(x)
    rows_a, rows_b = a.rows, b.rows
    _want(st, rows_a * a.cols + rows_b * b.cols)
    out_a = np.empty(rows_a, np.float32)
    out_b = np.empty(rows_b, np.float32)
    st["matvec_q4k_q5k_pair"](a._qp, a._sci, a._mni, a._dh, a._dmh,
                              b._qp, b._qh, b._sci, b._mni, b._dh, b._dmh,
                              st["f16_lut"], xq, ascale, bsums, out_a, out_b)
    return out_a, out_b


def dequant_q5k_tile(qp, qh, sc, mn, dh, dmh, out=None):
    """Expand native Q5_K rows to a float32 (nr, cols) tile."""
    st = _init()
    np = st["np"]
    _want_full(st)   # prefill tiles are large by construction
    if out is None:
        out = np.empty((qp.shape[0], qp.shape[1] * 2), np.float32)
    st["dequant_q5k"](qp, qh, sc, mn, dh, dmh, st["f16_lut"], out)
    return out


def matvec_q6k_int(q3, sc, dh, x, pre=None):
    """Fused Q6_K matvec: int8 codes (rows, n_sub, 16), int8 per-16 scales
    (rows, n_sub), per-256-block f16-bit scales dh (rows, nblk)."""
    st = _init()
    xq, ascale, _bsums = pre if pre is not None else quantize_acts(x)
    qf = q3.reshape(q3.shape[0], -1)  # contiguous flat view, no copy
    _want(st, qf.shape[0] * qf.shape[1])
    return st["matvec_q6k_int"](qf, sc, dh, st["f16_lut"], xq, ascale)


def rope_decode(v, cos_row, sin_row, n_heads, hd, n_rot, style):
    """Rotate one token's q or k vector in place-shape: returns a fresh
    (n_heads*hd,) float32 array, bit-identical to the NumPy slicing path."""
    st = _init()
    np = st["np"]
    out = np.empty(n_heads * hd, np.float32)
    fn = st["rope_norm"] if style == "norm" else st["rope_neox"]
    fn(v, cos_row, sin_row, n_heads, hd, n_rot, out)
    return out


def attention_decode(q, K, V, group, inv_sqrt):
    """Single-token grouped-query attention over the KV cache.

    q f32 (n_head*head_dim,), K/V f32 (t, n_kv, head_dim); returns
    (n_head, head_dim) float32, numerically the softmax(q.K/sqrt d).V of
    the NumPy path (fastmath reassociation differs in the last ulps).
    """
    st = _init()
    np = st["np"]
    # this kernel exists to keep attention off OpenBLAS's pool; it must not
    # inherit a narrow count from whichever matvec ran last
    _want_full(st)
    t, n_kv, hd = K.shape
    q3 = np.ascontiguousarray(q, dtype=np.float32).reshape(n_kv, group, hd)
    scores = np.empty((n_kv * group, t), np.float32)
    out = np.empty((n_kv, group, hd), np.float32)
    st["attention_decode"](q3, K, V, scores, out, np.float32(inv_sqrt))
    return out.reshape(n_kv * group, hd)


def dequant_q4k_tile(qp, sc, mn, dh, dmh, out=None):
    """Expand native Q4_K rows to a float32 (nr, cols) tile.

    Pass a preallocated `out` when calling in a loop: a fresh tile buffer
    is newly mapped memory and the page faults cost more than the kernel.
    """
    st = _init()
    np = st["np"]
    _want_full(st)   # prefill tiles are large by construction
    if out is None:
        out = np.empty((qp.shape[0], qp.shape[1] * 2), np.float32)
    st["dequant_q4k"](qp, sc, mn, dh, dmh, st["f16_lut"], out)
    return out


def dequant_q6k_tile(q3, sc, dh, out=None):
    """Expand native Q6_K rows to a float32 (nr, cols) tile."""
    st = _init()
    np = st["np"]
    _want_full(st)
    qf = q3.reshape(q3.shape[0], -1)
    if out is None:
        out = np.empty(qf.shape, np.float32)
    st["dequant_q6k"](qf, sc, dh, st["f16_lut"], out)
    return out


def warmup() -> None:
    """Trigger JIT compilation once (cached on disk afterwards)."""
    st = _init()
    if not st:
        return
    np = st["np"]
    q = np.zeros((2, 1, 32), dtype=np.int8)
    d = np.zeros((2, 1), dtype=np.float32)
    matvec_codes(q, d, None, np.zeros(32, dtype=np.float32))
    matvec_codes(q, d, d, np.zeros(32, dtype=np.float32))
    for b in (2, NARROW_BATCH + 1):          # both matmul kernels
        X = np.zeros((b, 32), dtype=np.float32)
        matmul_codes(q, d, None, X)
        matmul_codes(q, d, d, X)
    x256 = np.zeros(256, dtype=np.float32)
    matvec_q4k_int(np.zeros((2, 128), dtype=np.uint8),
                   np.zeros((2, 1, 8), dtype=np.uint8),
                   np.zeros((2, 1, 8), dtype=np.uint8),
                   np.zeros((2, 1), dtype=np.uint16),
                   np.zeros((2, 1), dtype=np.uint16), x256)
    matvec_q6k_int(np.zeros((2, 16, 16), dtype=np.int8),
                   np.zeros((2, 16), dtype=np.int8),
                   np.zeros((2, 1), dtype=np.uint16), x256)
    matvec_q5k_int(np.zeros((2, 128), dtype=np.uint8),
                   np.zeros((2, 32), dtype=np.uint8),
                   np.zeros((2, 1, 8), dtype=np.uint8),
                   np.zeros((2, 1, 8), dtype=np.uint8),
                   np.zeros((2, 1), dtype=np.uint16),
                   np.zeros((2, 1), dtype=np.uint16), x256)
    dequant_q4k_tile(np.zeros((2, 128), dtype=np.uint8),
                     np.zeros((2, 1, 8), dtype=np.uint8),
                     np.zeros((2, 1, 8), dtype=np.uint8),
                     np.zeros((2, 1), dtype=np.uint16),
                     np.zeros((2, 1), dtype=np.uint16))
    dequant_q5k_tile(np.zeros((2, 128), dtype=np.uint8),
                     np.zeros((2, 32), dtype=np.uint8),
                     np.zeros((2, 1, 8), dtype=np.uint8),
                     np.zeros((2, 1, 8), dtype=np.uint8),
                     np.zeros((2, 1), dtype=np.uint16),
                     np.zeros((2, 1), dtype=np.uint16))
    dequant_q6k_tile(np.zeros((2, 16, 16), dtype=np.int8),
                     np.zeros((2, 16), dtype=np.int8),
                     np.zeros((2, 1), dtype=np.uint16))
    attention_decode(np.zeros(2 * 4, dtype=np.float32),
                     np.zeros((3, 2, 4), dtype=np.float32),
                     np.zeros((3, 2, 4), dtype=np.float32), 1, 0.5)
    for style in ("norm", "neox"):
        # rope's kernels are njit without parallel=True, so the pool size
        # does not reach them and _want has nothing to say here
        rope_decode(np.zeros(8, dtype=np.float32),
                    np.zeros(2, dtype=np.float32),
                    np.zeros(2, dtype=np.float32), 2, 4, 4, style)
    # The grouped pair kernels, which used to be missed here. Package D
    # added _matvec_q4k_q6k_pair in round 4 without a warmup entry, so on a
    # cold Numba cache it compiled lazily inside the FIRST DECODE TOKEN of
    # any Q4_K_M model. Measured on machine C (round 5, qwen2.5-3B Q4_K_M,
    # ctx 4096, cold short-short): one token at 1865.4 ms against a 91.3 ms
    # worst case warm - and p50 did not move (82.0 vs 83.0), because a
    # single slow token cannot shift a median. Only the mean-based
    # decode tok/s showed it, 7.24 against 12.12.
    #
    # This is the same defect round 3 fixed for dense models ("the first
    # decode token paid a 0.68 s in-token JIT compile"). tests/smoke.py
    # pins it structurally now - every dispatcher in _state must have a
    # compiled signature after warmup - so the next kernel cannot
    # reintroduce it by being added without a line here.
    #
    # These call the compiled functions rather than the public wrappers
    # because the wrappers take QuantizedMatrix objects; the array dtypes
    # and layouts below are the ones qmatrix builds, so the signature
    # compiled here is the one a real dispatch uses.
    xq256, asc256, bs256 = quantize_acts(x256)
    u8 = np.zeros((2, 128), dtype=np.uint8)
    sc8 = np.zeros((2, 1, 8), dtype=np.uint8)
    u16 = np.zeros((2, 1), dtype=np.uint16)
    out2 = np.empty(2, np.float32)
    st["matvec_q4k_q6k_pair"](u8, sc8, sc8, u16, u16,
                              np.zeros((2, 256), dtype=np.int8),
                              np.zeros((2, 16), dtype=np.int8), u16,
                              st["f16_lut"], xq256, asc256, bs256,
                              out2, np.empty(2, np.float32))
    st["matvec_q4k_q5k_pair"](u8, sc8, sc8, u16, u16,
                              u8, np.zeros((2, 32), dtype=np.uint8),
                              sc8, sc8, u16, u16,
                              st["f16_lut"], xq256, asc256, bs256,
                              out2, np.empty(2, np.float32))
    # every warmup shape above is tiny, so _want left the pool at one
    # thread; hand it back sized for real work
    _want_full(st)
