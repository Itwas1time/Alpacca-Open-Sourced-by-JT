# Alpaccaroo - GGUF quantization formats, implemented from the spec in pure
# Python (with optional NumPy fast paths). MIT License. See LICENSE.
"""Dequantize GGUF tensor data to float32, and quantize for the writer.

Every decoder has a pure-Python implementation (standard library only).
When NumPy is importable, vectorized fast paths are used for the common
formats; the remaining ones fall back to the pure code transparently.
"""

from __future__ import annotations

import struct

try:  # optional accelerator only - everything works without it
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

import os

if os.environ.get("ALPACCAROO_PURE"):
    _np = None

QK = 32      # block size of the classic quants
QK_K = 256   # block size of the K-quants

_IQ4_NL_VALUES = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)

# Pinned llama.cpp b10507 iq3s_grid. Each uint32 is four little-endian
# positive grid magnitudes; block sign masks are applied separately.
_IQ3S_GRID = (
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
)


def _f16(b: bytes | memoryview, off: int) -> float:
    return struct.unpack_from("<e", b, off)[0]


# ---- pure-Python decoders ------------------------------------------------

def _deq_f32(data, n):
    return list(struct.unpack(f"<{n}f", bytes(data)))


def _deq_f16(data, n):
    return list(struct.unpack(f"<{n}e", bytes(data)))


def _deq_bf16(data, n):
    out = [0.0] * n
    raw = struct.unpack(f"<{n}H", bytes(data))
    for i, h in enumerate(raw):
        out[i] = struct.unpack("<f", struct.pack("<I", h << 16))[0]
    return out


def _deq_q8_0(data, n):
    out = [0.0] * n
    nb = n // QK
    for i in range(nb):
        off = i * 34
        d = _f16(data, off)
        qs = struct.unpack_from("<32b", data, off + 2)
        base = i * QK
        for j in range(QK):
            out[base + j] = d * qs[j]
    return out


def _deq_q4_0(data, n):
    out = [0.0] * n
    nb = n // QK
    for i in range(nb):
        off = i * 18
        d = _f16(data, off)
        base = i * QK
        for j in range(16):
            q = data[off + 2 + j]
            out[base + j] = d * ((q & 0x0F) - 8)
            out[base + j + 16] = d * ((q >> 4) - 8)
    return out


def _deq_q4_1(data, n):
    out = [0.0] * n
    nb = n // QK
    for i in range(nb):
        off = i * 20
        d = _f16(data, off)
        m = _f16(data, off + 2)
        base = i * QK
        for j in range(16):
            q = data[off + 4 + j]
            out[base + j] = d * (q & 0x0F) + m
            out[base + j + 16] = d * (q >> 4) + m
    return out


def _deq_q5_0(data, n):
    out = [0.0] * n
    nb = n // QK
    for i in range(nb):
        off = i * 22
        d = _f16(data, off)
        qh, = struct.unpack_from("<I", data, off + 2)
        base = i * QK
        for j in range(16):
            q = data[off + 6 + j]
            xh0 = ((qh >> j) << 4) & 0x10
            xh1 = (qh >> (j + 12)) & 0x10
            out[base + j] = d * (((q & 0x0F) | xh0) - 16)
            out[base + j + 16] = d * (((q >> 4) | xh1) - 16)
    return out


def _deq_q5_1(data, n):
    out = [0.0] * n
    nb = n // QK
    for i in range(nb):
        off = i * 24
        d = _f16(data, off)
        m = _f16(data, off + 2)
        qh, = struct.unpack_from("<I", data, off + 4)
        base = i * QK
        for j in range(16):
            q = data[off + 8 + j]
            xh0 = ((qh >> j) << 4) & 0x10
            xh1 = (qh >> (j + 12)) & 0x10
            out[base + j] = d * ((q & 0x0F) | xh0) + m
            out[base + j + 16] = d * ((q >> 4) | xh1) + m
    return out


def _scale_min_k4(j, scales):
    """6-bit packed scale/min pairs used by Q4_K / Q5_K."""
    if j < 4:
        return scales[j] & 63, scales[j + 4] & 63
    sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4)
    mn = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)
    return sc, mn


def _deq_q4_k(data, n):
    out = [0.0] * n
    nb = n // QK_K
    for i in range(nb):
        off = i * 144
        d = _f16(data, off)
        dmin = _f16(data, off + 2)
        scales = data[off + 4:off + 16]
        qs = data[off + 16:off + 144]
        y = i * QK_K
        q = 0
        is_ = 0
        for _ in range(0, QK_K, 64):
            sc, mn = _scale_min_k4(is_, scales)
            d1, m1 = d * sc, dmin * mn
            sc, mn = _scale_min_k4(is_ + 1, scales)
            d2, m2 = d * sc, dmin * mn
            for l in range(32):
                out[y] = d1 * (qs[q + l] & 0xF) - m1
                y += 1
            for l in range(32):
                out[y] = d2 * (qs[q + l] >> 4) - m2
                y += 1
            q += 32
            is_ += 2
    return out


def _deq_q5_k(data, n):
    out = [0.0] * n
    nb = n // QK_K
    for i in range(nb):
        off = i * 176
        d = _f16(data, off)
        dmin = _f16(data, off + 2)
        scales = data[off + 4:off + 16]
        qh = data[off + 16:off + 48]
        ql = data[off + 48:off + 176]
        y = i * QK_K
        q = 0
        is_ = 0
        u1, u2 = 1, 2
        for _ in range(0, QK_K, 64):
            sc, mn = _scale_min_k4(is_, scales)
            d1, m1 = d * sc, dmin * mn
            sc, mn = _scale_min_k4(is_ + 1, scales)
            d2, m2 = d * sc, dmin * mn
            for l in range(32):
                out[y] = d1 * ((ql[q + l] & 0xF) + (16 if qh[l] & u1 else 0)) - m1
                y += 1
            for l in range(32):
                out[y] = d2 * ((ql[q + l] >> 4) + (16 if qh[l] & u2 else 0)) - m2
                y += 1
            q += 32
            is_ += 2
            u1 <<= 2
            u2 <<= 2
    return out


def _deq_q6_k(data, n):
    out = [0.0] * n
    nb = n // QK_K
    for i in range(nb):
        off = i * 210
        ql = data[off:off + 128]
        qh = data[off + 128:off + 192]
        sc = struct.unpack_from("<16b", data, off + 192)
        d = _f16(data, off + 208)
        y = i * QK_K
        qloff = 0
        qhoff = 0
        soff = 0
        for _ in range(0, QK_K, 128):
            for l in range(32):
                is_ = l // 16
                q1 = ((ql[qloff + l] & 0xF) | (((qh[qhoff + l] >> 0) & 3) << 4)) - 32
                q2 = ((ql[qloff + l + 32] & 0xF) | (((qh[qhoff + l] >> 2) & 3) << 4)) - 32
                q3 = ((ql[qloff + l] >> 4) | (((qh[qhoff + l] >> 4) & 3) << 4)) - 32
                q4 = ((ql[qloff + l + 32] >> 4) | (((qh[qhoff + l] >> 6) & 3) << 4)) - 32
                out[y + l] = d * sc[soff + is_] * q1
                out[y + l + 32] = d * sc[soff + is_ + 2] * q2
                out[y + l + 64] = d * sc[soff + is_ + 4] * q3
                out[y + l + 96] = d * sc[soff + is_ + 6] * q4
            y += 128
            qloff += 64
            qhoff += 32
            soff += 8
    return out


def _deq_q2_k(data, n):
    out = [0.0] * n
    nb = n // QK_K
    for i in range(nb):
        off = i * 84
        scales = data[off:off + 16]
        qs = data[off + 16:off + 80]
        d = _f16(data, off + 80)
        dmin = _f16(data, off + 82)
        y = i * QK_K
        is_ = 0
        qoff = 0
        for _ in range(0, QK_K, 128):
            shift = 0
            for _j in range(4):
                sc = scales[is_]
                is_ += 1
                dl, ml = d * (sc & 0xF), dmin * (sc >> 4)
                for l in range(16):
                    out[y] = dl * ((qs[qoff + l] >> shift) & 3) - ml
                    y += 1
                sc = scales[is_]
                is_ += 1
                dl, ml = d * (sc & 0xF), dmin * (sc >> 4)
                for l in range(16):
                    out[y] = dl * ((qs[qoff + 16 + l] >> shift) & 3) - ml
                    y += 1
                shift += 2
            qoff += 32
    return out


def _deq_q3_k(data, n):
    kmask1, kmask2 = 0x03030303, 0x0F0F0F0F
    out = [0.0] * n
    nb = n // QK_K
    for i in range(nb):
        off = i * 110
        hmask = data[off:off + 32]
        qs = data[off + 32:off + 96]
        aux = list(struct.unpack_from("<3I", data, off + 96))
        d_all = _f16(data, off + 108)
        tmp = aux[2]
        a0 = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4)
        a1 = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4)
        a2 = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
        a3 = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
        packed = struct.pack("<4I", a0, a1, a2, a3)
        scales = struct.unpack("<16b", packed)
        y = i * QK_K
        is_ = 0
        m = 1
        qoff = 0
        for _ in range(0, QK_K, 128):
            shift = 0
            for _j in range(4):
                dl = d_all * (scales[is_] - 32)
                is_ += 1
                for l in range(16):
                    q = (qs[qoff + l] >> shift) & 3
                    h = 0 if (hmask[l] & m) else 4
                    out[y] = dl * (q - h)
                    y += 1
                dl = d_all * (scales[is_] - 32)
                is_ += 1
                for l in range(16):
                    q = (qs[qoff + 16 + l] >> shift) & 3
                    h = 0 if (hmask[16 + l] & m) else 4
                    out[y] = dl * (q - h)
                    y += 1
                shift += 2
                m <<= 1
            qoff += 32
    return out


def _deq_iq4_nl(data, n):
    out = [0.0] * n
    for block in range(n // QK):
        offset = block * 18
        scale = _f16(data, offset)
        base = block * QK
        for index in range(16):
            packed = data[offset + 2 + index]
            out[base + index] = scale * _IQ4_NL_VALUES[packed & 0x0F]
            out[base + 16 + index] = scale * _IQ4_NL_VALUES[packed >> 4]
    return out


def _deq_iq4_xs(data, n):
    out = [0.0] * n
    for block in range(n // QK_K):
        offset = block * 136
        scale = _f16(data, offset)
        scales_high, = struct.unpack_from("<H", data, offset + 2)
        scales_low = data[offset + 4:offset + 8]
        quants = data[offset + 8:offset + 136]
        base = block * QK_K
        for subblock in range(8):
            low = (scales_low[subblock // 2] >> (4 * (subblock % 2))) & 0x0F
            high = ((scales_high >> (2 * subblock)) & 0x03) << 4
            effective = scale * ((low | high) - 32)
            quant_offset = subblock * 16
            output_offset = base + subblock * 32
            for index in range(16):
                packed = quants[quant_offset + index]
                out[output_offset + index] = (
                    effective * _IQ4_NL_VALUES[packed & 0x0F]
                )
                out[output_offset + 16 + index] = (
                    effective * _IQ4_NL_VALUES[packed >> 4]
                )
    return out


def _deq_iq3_s(data, n):
    out = [0.0] * n
    for block in range(n // QK_K):
        offset = block * 110
        scale = _f16(data, offset)
        quants = data[offset + 2:offset + 66]
        high = data[offset + 66:offset + 74]
        signs = data[offset + 74:offset + 106]
        scales = data[offset + 106:offset + 110]
        base = block * QK_K
        for subblock in range(8):
            packed_scale = scales[subblock // 2]
            nibble = (
                packed_scale & 0x0F if subblock % 2 == 0
                else packed_scale >> 4
            )
            effective = scale * (1 + 2 * nibble)
            high_bits = high[subblock]
            for lane in range(4):
                index0 = (
                    quants[subblock * 8 + 2 * lane]
                    | (((high_bits >> (2 * lane)) & 1) << 8)
                )
                index1 = (
                    quants[subblock * 8 + 2 * lane + 1]
                    | (((high_bits >> (2 * lane + 1)) & 1) << 8)
                )
                grid0 = _IQ3S_GRID[index0]
                grid1 = _IQ3S_GRID[index1]
                sign_mask = signs[subblock * 4 + lane]
                output_offset = base + subblock * 32 + lane * 8
                for coordinate in range(4):
                    magnitude0 = (grid0 >> (8 * coordinate)) & 0xFF
                    magnitude1 = (grid1 >> (8 * coordinate)) & 0xFF
                    out[output_offset + coordinate] = effective * magnitude0 * (
                        -1.0 if sign_mask & (1 << coordinate) else 1.0
                    )
                    out[output_offset + 4 + coordinate] = effective * magnitude1 * (
                        -1.0 if sign_mask & (1 << (4 + coordinate)) else 1.0
                    )
    return out


_PURE_DECODERS = {
    "F32": _deq_f32, "F16": _deq_f16, "BF16": _deq_bf16,
    "Q8_0": _deq_q8_0, "Q4_0": _deq_q4_0, "Q4_1": _deq_q4_1,
    "Q5_0": _deq_q5_0, "Q5_1": _deq_q5_1,
    "Q4_K": _deq_q4_k, "Q5_K": _deq_q5_k, "Q6_K": _deq_q6_k,
    "Q2_K": _deq_q2_k, "Q3_K": _deq_q3_k,
    "IQ3_S": _deq_iq3_s, "IQ4_NL": _deq_iq4_nl, "IQ4_XS": _deq_iq4_xs,
}


# ---- quantized block geometry ---------------------------------------------
# dtype -> (block_elements, block_bytes, sub_block_len, affine)
# `sub_block_len` is the run of consecutive elements sharing one effective
# scale (and offset, when `affine`). Available without NumPy so the pure
# backend can slice rows out of raw block bytes.
QUANT_GEOMETRY = {
    "Q8_0": (QK, 34, 32, False),
    "Q4_0": (QK, 18, 32, False),
    "Q4_1": (QK, 20, 32, True),
    "Q5_0": (QK, 22, 32, False),
    "Q5_1": (QK, 24, 32, True),
    "Q2_K": (QK_K, 84, 16, True),
    "Q3_K": (QK_K, 110, 16, False),
    "Q4_K": (QK_K, 144, 32, True),
    "Q5_K": (QK_K, 176, 32, True),
    "Q6_K": (QK_K, 210, 16, False),
    "IQ3_S": (QK_K, 110, 32, False),
    "IQ4_NL": (QK, 18, 32, False),
    "IQ4_XS": (QK_K, 136, 32, False),
}


# ---- NumPy fast paths ----------------------------------------------------
#
# Each unpacker decodes raw blocks into the shared compact representation
# used by both `dequantize` and `alpaccaroo.qmatrix.QuantMatrix`:
#   codes int8 (nb, block_elements)  - quant codes in element order
#   d_eff float32 (nb, n_sub)        - effective scale per sub-block
#   m_eff float32 (nb, n_sub) | None - effective offset per sub-block
# so that value = d_eff * code (+ m_eff).

def _np_blocks(data, nb, block_bytes):
    return _np.frombuffer(data, dtype=_np.uint8).reshape(nb, block_bytes)


def _np_f16_col(b, off):
    return b[:, off:off + 2].copy().view(_np.float16).astype(_np.float32)


def _np_unpack_q8_0(b):
    d = _np_f16_col(b, 0)
    q = b[:, 2:34].view(_np.int8).copy()
    return q, d, None


def _np_unpack_q4_0(b):
    d = _np_f16_col(b, 0)
    qs = b[:, 2:18]
    q = _np.empty((b.shape[0], QK), dtype=_np.int8)
    q[:, :16] = (qs & 0x0F).view(_np.int8)
    q[:, 16:] = (qs >> 4).view(_np.int8)
    q -= 8
    return q, d, None


def _np_unpack_q4_1(b):
    d = _np_f16_col(b, 0)
    m = _np_f16_col(b, 2)
    qs = b[:, 4:20]
    q = _np.empty((b.shape[0], QK), dtype=_np.int8)
    q[:, :16] = (qs & 0x0F).view(_np.int8)
    q[:, 16:] = (qs >> 4).view(_np.int8)
    return q, d, m


def _np_high_bits(qh_u32):
    """Per-element 5th bit (already shifted to 0x10) from a u32 mask column."""
    shifts = _np.arange(32, dtype=_np.uint32)
    return (((qh_u32 >> shifts) & 1) << 4).astype(_np.uint8)


def _np_unpack_q5_0(b):
    d = _np_f16_col(b, 0)
    qh = b[:, 2:6].copy().view(_np.uint32)
    hi5 = _np_high_bits(qh)
    qs = b[:, 6:22]
    q = _np.empty((b.shape[0], QK), dtype=_np.int8)
    q[:, :16] = ((qs & 0x0F) | hi5[:, :16]).view(_np.int8)
    q[:, 16:] = ((qs >> 4) | hi5[:, 16:]).view(_np.int8)
    q -= 16
    return q, d, None


def _np_unpack_q5_1(b):
    d = _np_f16_col(b, 0)
    m = _np_f16_col(b, 2)
    qh = b[:, 4:8].copy().view(_np.uint32)
    hi5 = _np_high_bits(qh)
    qs = b[:, 8:24]
    q = _np.empty((b.shape[0], QK), dtype=_np.int8)
    q[:, :16] = ((qs & 0x0F) | hi5[:, :16]).view(_np.int8)
    q[:, 16:] = ((qs >> 4) | hi5[:, 16:]).view(_np.int8)
    return q, d, m


def _np_unpack_k_scales_int(scales):
    """6-bit packed scale/min pairs of Q4_K/Q5_K -> uint8 (nb, 8) each."""
    nb = scales.shape[0]
    sc = _np.empty((nb, 8), dtype=_np.uint8)
    mn = _np.empty((nb, 8), dtype=_np.uint8)
    for j in range(8):
        if j < 4:
            sc[:, j] = scales[:, j] & 63
            mn[:, j] = scales[:, j + 4] & 63
        else:
            sc[:, j] = (scales[:, j + 4] & 0x0F) | ((scales[:, j - 4] >> 6) << 4)
            mn[:, j] = (scales[:, j + 4] >> 4) | ((scales[:, j] >> 6) << 4)
    return sc, mn


def _np_unpack_k_scales(scales):
    """6-bit packed scale/min pairs of Q4_K/Q5_K -> float32 (nb, 8) each."""
    sc, mn = _np_unpack_k_scales_int(scales)
    return sc.astype(_np.float32), mn.astype(_np.float32)


def _np_unpack_q4_k(b):
    d = _np_f16_col(b, 0)
    dmin = _np_f16_col(b, 2)
    sc, mn = _np_unpack_k_scales(b[:, 4:16])
    qs = b[:, 16:144]
    q = _np.empty((b.shape[0], QK_K), dtype=_np.int8)
    for half in range(4):  # 4 chunks of 32 bytes -> 2 sub-blocks each
        chunk = qs[:, half * 32:(half + 1) * 32]
        q[:, half * 64:half * 64 + 32] = (chunk & 0x0F).view(_np.int8)
        q[:, half * 64 + 32:half * 64 + 64] = (chunk >> 4).view(_np.int8)
    return q, d * sc, -(dmin * mn)


def _np_unpack_q5_k(b):
    d = _np_f16_col(b, 0)
    dmin = _np_f16_col(b, 2)
    sc, mn = _np_unpack_k_scales(b[:, 4:16])
    qh = b[:, 16:48]
    qs = b[:, 48:176]
    q = _np.empty((b.shape[0], QK_K), dtype=_np.int8)
    for half in range(4):
        chunk = qs[:, half * 32:(half + 1) * 32]
        j1, j2 = 2 * half, 2 * half + 1
        hb1 = ((qh >> j1) & 1) << 4
        hb2 = ((qh >> j2) & 1) << 4
        q[:, j1 * 32:(j1 + 1) * 32] = ((chunk & 0x0F) | hb1).view(_np.int8)
        q[:, j2 * 32:(j2 + 1) * 32] = ((chunk >> 4) | hb2).view(_np.int8)
    return q, d * sc, -(dmin * mn)


def _np_q6_k_codes(b):
    """Q6_K blocks -> int8 codes (nb, 256) already centered to [-32, 31]."""
    ql = b[:, 0:128]
    qh = b[:, 128:192]
    q = _np.empty((b.shape[0], QK_K), dtype=_np.int8)
    for half in range(2):  # two 128-element halves
        qlh = ql[:, half * 64:(half + 1) * 64]
        qhh = qh[:, half * 32:(half + 1) * 32]
        base = half * 128
        q[:, base + 0:base + 32] = (
            ((qlh[:, :32] & 0xF) | (((qhh >> 0) & 3) << 4)).view(_np.int8))
        q[:, base + 32:base + 64] = (
            ((qlh[:, 32:] & 0xF) | (((qhh >> 2) & 3) << 4)).view(_np.int8))
        q[:, base + 64:base + 96] = (
            ((qlh[:, :32] >> 4) | (((qhh >> 4) & 3) << 4)).view(_np.int8))
        q[:, base + 96:base + 128] = (
            ((qlh[:, 32:] >> 4) | (((qhh >> 6) & 3) << 4)).view(_np.int8))
    q -= 32
    return q


def _np_unpack_q6_k(b):
    sc = b[:, 192:208].view(_np.int8).astype(_np.float32)  # (nb, 16)
    d = _np_f16_col(b, 208)
    return _np_q6_k_codes(b), d * sc, None


def _np_unpack_q2_k(b):
    # 16 uint8 scales, 64 bytes of 2-bit codes, then d and dmin as f16.
    # Each scale byte packs the sub-block's scale in its low nibble and its
    # offset in the high one: value = d*(sc & 0xF)*code - dmin*(sc >> 4).
    scales = b[:, 0:16]
    qs = b[:, 16:80]
    d = _np_f16_col(b, 80)
    dmin = _np_f16_col(b, 82)
    q = _np.empty((b.shape[0], QK_K), dtype=_np.int8)
    for half in range(2):                     # two 128-element halves
        chunk = qs[:, half * 32:(half + 1) * 32]
        for j in range(4):                    # four 2-bit lanes per byte
            sub = 8 * half + 2 * j
            code = (chunk >> (2 * j)) & 3
            q[:, sub * 16:(sub + 1) * 16] = code[:, :16].view(_np.int8)
            q[:, (sub + 1) * 16:(sub + 2) * 16] = code[:, 16:].view(_np.int8)
    d_eff = d * (scales & 0x0F).astype(_np.float32)
    m_eff = -(dmin * (scales >> 4).astype(_np.float32))
    return q, d_eff, m_eff


def _np_unpack_q3_k(b):
    # 32 bytes of high bits, 64 bytes of 2-bit codes, 12 bytes of packed
    # 6-bit scales, then d. The high-bit mask is INVERTED: a set bit means
    # "do not subtract 4", so the code lands in [-4, 3].
    hmask = b[:, 0:32]
    qs = b[:, 32:96]
    aux = b[:, 96:108].view(_np.uint32).reshape(b.shape[0], 3)
    d_all = _np_f16_col(b, 108)
    kmask1, kmask2 = _np.uint32(0x03030303), _np.uint32(0x0F0F0F0F)
    tmp = aux[:, 2]
    packed = _np.empty((b.shape[0], 4), dtype=_np.uint32)
    packed[:, 0] = (aux[:, 0] & kmask2) | (((tmp >> 0) & kmask1) << 4)
    packed[:, 1] = (aux[:, 1] & kmask2) | (((tmp >> 2) & kmask1) << 4)
    packed[:, 2] = ((aux[:, 0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
    packed[:, 3] = ((aux[:, 1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
    scales = packed.view(_np.int8).astype(_np.float32)   # (nb, 16)
    q = _np.empty((b.shape[0], QK_K), dtype=_np.int8)
    for half in range(2):
        chunk = qs[:, half * 32:(half + 1) * 32]
        for j in range(4):
            sub = 8 * half + 2 * j
            bit = _np.uint8(1 << (4 * half + j))
            code = ((chunk >> (2 * j)) & 3).astype(_np.int8)
            lo = code[:, :16] - _np.where(hmask[:, :16] & bit, 0, 4).astype(_np.int8)
            hi = code[:, 16:] - _np.where(hmask[:, 16:] & bit, 0, 4).astype(_np.int8)
            q[:, sub * 16:(sub + 1) * 16] = lo
            q[:, (sub + 1) * 16:(sub + 2) * 16] = hi
    return q, d_all * (scales - 32.0), None


def _np_unpack_iq4_nl(b):
    d = _np_f16_col(b, 0)
    packed = b[:, 2:18]
    table = _np.asarray(_IQ4_NL_VALUES, dtype=_np.int8)
    q = _np.empty((b.shape[0], QK), dtype=_np.int8)
    q[:, :16] = table[packed & 0x0F]
    q[:, 16:] = table[packed >> 4]
    return q, d, None


def _np_unpack_iq4_xs(b):
    d = _np_f16_col(b, 0)
    scales_high = b[:, 2:4].copy().view(_np.uint16).reshape(-1)
    scales_low = b[:, 4:8]
    packed = b[:, 8:136]
    table = _np.asarray(_IQ4_NL_VALUES, dtype=_np.int8)
    q = _np.empty((b.shape[0], QK_K), dtype=_np.int8)
    scales = _np.empty((b.shape[0], 8), dtype=_np.float32)
    for subblock in range(8):
        low = (
            scales_low[:, subblock // 2] >> (4 * (subblock % 2))
        ) & 0x0F
        high = ((scales_high >> (2 * subblock)) & 0x03) << 4
        scales[:, subblock] = (low | high).astype(_np.int16) - 32
        quant_offset = subblock * 16
        output_offset = subblock * 32
        chunk = packed[:, quant_offset:quant_offset + 16]
        q[:, output_offset:output_offset + 16] = table[chunk & 0x0F]
        q[:, output_offset + 16:output_offset + 32] = table[chunk >> 4]
    return q, d * scales, None


def _np_unpack_iq3_s(b):
    d = _np_f16_col(b, 0)
    quants = b[:, 2:66]
    high = b[:, 66:74]
    signs = b[:, 74:106]
    packed_scales = b[:, 106:110]
    table_words = _np.asarray(_IQ3S_GRID, dtype=_np.uint32)
    table = _np.empty((512, 4), dtype=_np.int8)
    for coordinate in range(4):
        table[:, coordinate] = (
            (table_words >> (8 * coordinate)) & 0xFF
        ).astype(_np.int8)
    q = _np.empty((b.shape[0], QK_K), dtype=_np.int8)
    scales = _np.empty((b.shape[0], 8), dtype=_np.float32)
    for subblock in range(8):
        packed = packed_scales[:, subblock // 2]
        nibble = (
            packed & 0x0F if subblock % 2 == 0 else packed >> 4
        )
        scales[:, subblock] = 1.0 + 2.0 * nibble.astype(_np.float32)
        high_bits = high[:, subblock].astype(_np.uint16)
        for lane in range(4):
            index0 = (
                quants[:, subblock * 8 + 2 * lane].astype(_np.uint16)
                | (((high_bits >> (2 * lane)) & 1) << 8)
            )
            index1 = (
                quants[:, subblock * 8 + 2 * lane + 1].astype(_np.uint16)
                | (((high_bits >> (2 * lane + 1)) & 1) << 8)
            )
            sign_mask = signs[:, subblock * 4 + lane]
            output_offset = subblock * 32 + lane * 8
            for coordinate in range(4):
                sign0 = _np.where(
                    sign_mask & (1 << coordinate), -1, 1,
                ).astype(_np.int8)
                sign1 = _np.where(
                    sign_mask & (1 << (4 + coordinate)), -1, 1,
                ).astype(_np.int8)
                q[:, output_offset + coordinate] = (
                    table[index0, coordinate] * sign0
                )
                q[:, output_offset + 4 + coordinate] = (
                    table[index1, coordinate] * sign1
                )
    return q, d * scales, None


_NP_UNPACKERS = {
    "Q8_0": _np_unpack_q8_0,
    "Q4_0": _np_unpack_q4_0,
    "Q4_1": _np_unpack_q4_1,
    "Q5_0": _np_unpack_q5_0,
    "Q5_1": _np_unpack_q5_1,
    "Q2_K": _np_unpack_q2_k,
    "Q3_K": _np_unpack_q3_k,
    "Q4_K": _np_unpack_q4_k,
    "Q5_K": _np_unpack_q5_k,
    "Q6_K": _np_unpack_q6_k,
    "IQ3_S": _np_unpack_iq3_s,
    "IQ4_NL": _np_unpack_iq4_nl,
    "IQ4_XS": _np_unpack_iq4_xs,
}


def np_unpack_q4k_native(data, n: int):
    """Q4_K blocks kept in their native fields, nothing widened.

    Returns (packed u8 (nb, 128), sc u8 (nb, 8), mn u8 (nb, 8),
    d_bits u16 (nb,), dmin_bits u16 (nb,)): the split-nibble qs bytes as
    stored, the 6-bit sub-block scales/mins as integers, and the per-block
    super-scales as raw f16 bits. value = f16(d)*sc*code - f16(dmin)*mn,
    where byte j of 32-byte chunk c holds element 64c+j in its low nibble
    and element 64c+32+j in its high one. ~0.58 B/weight."""
    if _np is None:
        raise RuntimeError("np_unpack_q4k_native requires NumPy")
    if n % QK_K:
        raise ValueError(f"Q4_K needs a multiple of {QK_K} elements")
    b = _np_blocks(data, n // QK_K, 144)
    d_bits = b[:, 0:2].copy().view(_np.uint16).reshape(-1)
    dmin_bits = b[:, 2:4].copy().view(_np.uint16).reshape(-1)
    sc, mn = _np_unpack_k_scales_int(b[:, 4:16])
    packed = _np.ascontiguousarray(b[:, 16:144])
    return packed, sc, mn, d_bits, dmin_bits


def np_unpack_q5k_native(data, n: int):
    """Q5_K blocks kept in their native fields, nothing widened.

    Returns (qs u8 (nb, 128) split-nibble low-4 bits, qh u8 (nb, 32) fifth
    bits, sc u8 (nb, 8), mn u8 (nb, 8), d_bits u16, dmin_bits u16):
    element 64c+j of a block is (qs[c*32+j] & 0xF) | ((qh[j] >> 2c) & 1) << 4
    and element 64c+32+j is (qs[c*32+j] >> 4) | ((qh[j] >> (2c+1)) & 1) << 4;
    value = f16(d)*sc*code - f16(dmin)*mn. ~0.70 B/weight."""
    if _np is None:
        raise RuntimeError("np_unpack_q5k_native requires NumPy")
    if n % QK_K:
        raise ValueError(f"Q5_K needs a multiple of {QK_K} elements")
    b = _np_blocks(data, n // QK_K, 176)
    d_bits = b[:, 0:2].copy().view(_np.uint16).reshape(-1)
    dmin_bits = b[:, 2:4].copy().view(_np.uint16).reshape(-1)
    sc, mn = _np_unpack_k_scales_int(b[:, 4:16])
    qh = _np.ascontiguousarray(b[:, 16:48])
    qs = _np.ascontiguousarray(b[:, 48:176])
    return qs, qh, sc, mn, d_bits, dmin_bits


def np_unpack_q6k_native(data, n: int):
    """Q6_K blocks as int8 codes plus their native scale fields.

    Returns (codes i8 (nb, 256) centered to [-32, 31], sc i8 (nb, 16),
    d_bits u16 (nb,)): value = f16(d)*sc*code, no offset term.
    ~1.07 B/weight (the 6-bit code packing stays future work)."""
    if _np is None:
        raise RuntimeError("np_unpack_q6k_native requires NumPy")
    if n % QK_K:
        raise ValueError(f"Q6_K needs a multiple of {QK_K} elements")
    b = _np_blocks(data, n // QK_K, 210)
    codes = _np_q6_k_codes(b)
    sc = b[:, 192:208].copy().view(_np.int8)
    d_bits = b[:, 208:210].copy().view(_np.uint16).reshape(-1)
    return codes, sc, d_bits


def np_unpack(data, n: int, dtype: str):
    """Unpack `n` elements of raw GGUF blocks into (codes, d_eff, m_eff).

    codes is int8 (nb, block_elements) in element order; d_eff/m_eff are
    float32 (nb, n_sub) so that value = d_eff * code (+ m_eff) per sub-block
    of QUANT_GEOMETRY[dtype] sub_block_len elements. Requires NumPy.
    """
    if _np is None:
        raise RuntimeError("np_unpack requires NumPy")
    if dtype not in _NP_UNPACKERS:
        raise ValueError(f"np_unpack does not support {dtype}")
    block_n, block_b, _sub, _aff = QUANT_GEOMETRY[dtype]
    if n % block_n:
        raise ValueError(f"{dtype} needs a multiple of {block_n} elements")
    return _NP_UNPACKERS[dtype](_np_blocks(data, n // block_n, block_b))


def _np_assemble(q, d_eff, m_eff, sub_len):
    nb, block_n = q.shape
    out = q.astype(_np.float32).reshape(nb, block_n // sub_len, sub_len)
    out *= d_eff[:, :, None]
    if m_eff is not None:
        out += m_eff[:, :, None]
    return out.reshape(-1)


def dequantize(data, n: int, dtype: str):
    """Decode `n` elements of GGUF tensor `data` to float32.

    Returns a NumPy float32 array when NumPy is available, else list[float].
    """
    if dtype not in _PURE_DECODERS:
        raise ValueError(
            f"tensor type {dtype} is not supported by the alpaccaroo engine "
            f"(supported: {', '.join(sorted(_PURE_DECODERS))})")
    if _np is not None:
        if dtype == "F32":
            return _np.frombuffer(data, dtype=_np.float32).copy()
        if dtype == "F16":
            return _np.frombuffer(data, dtype=_np.float16).astype(_np.float32)
        if dtype == "BF16":
            raw = _np.frombuffer(data, dtype=_np.uint16).astype(_np.uint32) << 16
            return raw.view(_np.float32).copy()
        if dtype in _NP_UNPACKERS:
            q, d_eff, m_eff = np_unpack(data, n, dtype)
            return _np_assemble(q, d_eff, m_eff, QUANT_GEOMETRY[dtype][2])
        return _np.asarray(_PURE_DECODERS[dtype](data, n), dtype=_np.float32)
    return _PURE_DECODERS[dtype](data, n)


# ---- quantizers (used by the GGUF writer / tests) ------------------------

def quantize_q8_0(values: list[float]) -> bytes:
    if len(values) % QK:
        raise ValueError("Q8_0 needs a multiple of 32 values")
    out = bytearray()
    for i in range(0, len(values), QK):
        block = values[i:i + QK]
        amax = max(abs(v) for v in block)
        d = amax / 127.0 if amax else 0.0
        inv = 1.0 / d if d else 0.0
        qs = [max(-128, min(127, round(v * inv))) for v in block]
        out += struct.pack("<e32b", d, *qs)
    return bytes(out)


def quantize_q4_0(values: list[float]) -> bytes:
    if len(values) % QK:
        raise ValueError("Q4_0 needs a multiple of 32 values")
    out = bytearray()
    for i in range(0, len(values), QK):
        block = values[i:i + QK]
        vmax = max(block, key=abs)
        d = vmax / -8.0
        inv = 1.0 / d if d else 0.0
        q = [max(0, min(15, int(v * inv + 8.5))) for v in block]
        packed = bytes((q[j] & 0x0F) | (q[j + 16] << 4) for j in range(16))
        out += struct.pack("<e", d) + packed
    return bytes(out)
