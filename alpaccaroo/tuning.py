# Alpaccaroo - opt-in, bounded autotuning of the kernel thread count.
# MIT License. See LICENSE.
"""Measure this machine instead of assuming it.

No single thread count is best everywhere. Physical cores is a good default
for a bandwidth-bound decode loop, but a machine with more memory channels
than cores, an SMT design that hides latency well, or a small matrix whose
work does not repay a thread-pool wake-up, all disagree with it - and they
disagree in different directions.

So: run a **bounded** warm micro-benchmark over a handful of thread counts,
on the shapes the model actually dispatches, and cache the winner. The
rules the plan sets, kept literally:

* opt-in. Nothing here runs unless ``alpaccaroo tune`` is invoked or
  ``ALPACCAROO_AUTOTUNE=1`` is set. A normal CLI start never benchmarks.
* ``ALPACCAROO_THREADS`` always wins. A user who chose a number keeps it.
* the cache is keyed on everything that can change the answer - CPU,
  core counts, OS, Python/NumPy/Numba/llvmlite versions, the kernel
  module's own source - so an upgrade invalidates it rather than serving a
  stale winner.
* the probe is bounded: a fixed repetition count over synthetic matrices
  sized like the model's, with a wall-clock ceiling.

The probe matrices are synthesised from deterministic bytes, so tuning
needs no model file and touches no weights the user cares about; what
matters for a bandwidth-bound kernel is the shape and the dtype, not the
values.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

from . import kernels as _kernels

#: Thread counts worth trying, as multiples/fractions of the topology. The
#: plan's set: 1, half the physical cores, the physical cores, every
#: logical CPU. Duplicates and non-positive values are dropped.
def candidate_thread_counts() -> list[int]:
    from . import _platform
    logical = os.cpu_count() or 1
    physical = _platform.physical_cores() or logical
    raw = [1, max(1, physical // 2), physical, logical]
    out: list[int] = []
    for n in raw:
        n = int(n)
        if 0 < n <= max(logical, physical) and n not in out:
            out.append(n)
    return sorted(out)


#: A generic size ladder for `alpaccaroo tune` with no model: a narrow GQA
#: projection, a square attention output, a wide FFN, and a large vocabulary
#: head. Sizes are round numbers spanning the range llama-class models use,
#: not any one model's dimensions.
DEFAULT_PROBE_SHAPES = (
    ("narrow", "Q4_K", 256, 2048),
    ("square", "Q4_K", 2048, 2048),
    ("wide-ffn", "Q4_K", 11008, 2048),
    ("vocab-head", "Q6_K", 32000, 2048),
)


def gguf_matvec_shapes(path: str) -> list[tuple]:
    """The same shape class as :func:`model_matvec_shapes`, read from the
    GGUF header alone - so `alpaccaroo tune -m 8b-model` costs milliseconds
    and a few kilobytes instead of loading five gigabytes of weights.

    Mirrors the loader's decisions that change a dispatch shape: the
    attn_q+attn_k and ffn_gate+ffn_up fusions (same dtype, adjacent rows),
    the tied output projection, and skipping anything the quantized matvec
    cannot take. Anything that cannot be read returns an empty list, and
    the caller falls back to the generic ladder.
    """
    from . import tensor as T
    from .gguf import GGUFFile
    from .model import SUPPORTED_ARCHES

    try:
        gf = GGUFFile.open(path)
    except Exception:
        return []
    try:
        arch = gf.architecture
        if arch not in SUPPORTED_ARCHES:
            return []
        n_layer = int(gf.get(f"{arch}.block_count", 0) or 0)
        if n_layer <= 0:
            return []
        fuse_on = (T.HAS_NUMPY and not os.environ.get("ALPACCAROO_F32")
                   and os.environ.get("ALPACCAROO_FUSE", "").strip().lower()
                   not in ("0", "off", "no"))
        counts: dict[tuple, list] = {}

        def note(label: str, info) -> None:
            if info is None or len(info.shape) < 2:
                return
            cols, rows = int(info.shape[0]), int(info.shape[1])
            if not T.can_quantized_matvec(info.dtype, cols):
                return
            entry = counts.setdefault((info.dtype, rows, cols), [label, 0])
            entry[1] += 1

        def fused(a, b):
            """The loader concatenates two same-dtype row blocks into one
            matrix, so the dispatch is one call at the summed row count."""
            if not fuse_on or a is None or b is None or a.dtype != b.dtype:
                return None
            if len(a.shape) < 2 or len(b.shape) < 2:
                return None
            if int(a.shape[0]) != int(b.shape[0]):
                return None
            from .gguf import TensorInfo
            return TensorInfo(name=f"{a.name}+{b.name}",
                              shape=(int(a.shape[0]),
                                     int(a.shape[1]) + int(b.shape[1])),
                              dtype=a.dtype, offset=0)

        for i in range(n_layer):
            p = f"blk.{i}."
            g = gf.tensors.get
            qk = fused(g(p + "attn_q.weight"), g(p + "attn_k.weight"))
            if qk is not None:
                note("attn_qk", qk)
            else:
                note("attn_q", g(p + "attn_q.weight"))
                note("attn_k", g(p + "attn_k.weight"))
            note("attn_v", g(p + "attn_v.weight"))
            note("attn_output", g(p + "attn_output.weight"))
            gu = fused(g(p + "ffn_gate.weight"), g(p + "ffn_up.weight"))
            if gu is not None:
                note("ffn_gate_up", gu)
            else:
                note("ffn_gate", g(p + "ffn_gate.weight"))
                note("ffn_up", g(p + "ffn_up.weight"))
            note("ffn_down", g(p + "ffn_down.weight"))
        head = gf.tensors.get("output.weight") or gf.tensors.get(
            "token_embd.weight")
        note("output", head)
        out = [(lbl, dt, r, c, n) for (dt, r, c), (lbl, n) in counts.items()]
        out.sort(key=lambda e: -(e[2] * e[3] * e[4]))
        return out
    except Exception:
        return []
    finally:
        gf.close()


def model_matvec_shapes(model) -> list[tuple]:
    """(label, dtype, rows, cols, per_token_calls) for the distinct decode
    matvec shapes of a loaded model, heaviest first.

    Derived from the model, never from a hard-coded architecture: a
    hypothetical model whose FFN is narrower than its attention tunes for
    that, and a fused q/k matrix is probed at its fused width because that
    is the dispatch that happens.
    """
    from . import tensor as T
    counts: dict[tuple, list] = {}

    def note(label: str, W) -> None:
        if W is None or not T.is_quantized_matrix(W):
            return
        key = (W.dtype, int(W.rows), int(W.cols))
        entry = counts.setdefault(key, [label, 0])
        entry[1] += 1

    for ly in getattr(model, "layers", ()):
        for label, get in (("attn_q", lambda l: l.wq), ("attn_k", lambda l: l.wk),
                           ("attn_qk", lambda l: l.wqk), ("attn_v", lambda l: l.wv),
                           ("attn_output", lambda l: l.wo),
                           ("ffn_gate", lambda l: l.w_gate),
                           ("ffn_up", lambda l: l.w_up),
                           ("ffn_gate_up", lambda l: l.wgu),
                           ("ffn_down", lambda l: l.w_down)):
            try:
                note(label, get(ly))
            except AttributeError:
                pass
    note("output", getattr(model, "output", None))
    out = [(lbl, dt, r, c, n) for (dt, r, c), (lbl, n) in counts.items()]
    out.sort(key=lambda e: -(e[2] * e[3] * e[4]))
    return out


# ---- probe matrices ---------------------------------------------------------

def _probe_bytes(dtype: str, n_elements: int) -> bytes:
    """Deterministic, structurally valid blocks for a quantized dtype.

    A matvec's cost depends on the layout, not the values, so the codes are
    an arbitrary repeating pattern. The SCALE fields are not arbitrary:
    every K-quant block ends or begins with f16 scales, and a scale that
    decodes to zero makes every output zero - which still times correctly
    but turns any correctness check run against these matrices into a
    comparison of zeros. So each supported layout gets its scales written
    where that layout actually keeps them:

        Q4_K  d, dmin, scales[12], qs[128]                      (144 B)
        Q5_K  d, dmin, scales[12], qh[32], ql[128]              (176 B)
        Q6_K  ql[128], qh[64], scales[16], d                    (210 B)

    For any other dtype the layout is not enumerated here, so every byte is
    held below 0x40 instead: an f16 whose high byte is <= 0x3F is finite by
    construction, wherever in the block it happens to sit.
    """
    import struct

    from .quants import QUANT_GEOMETRY

    geom = QUANT_GEOMETRY.get(dtype)
    if geom is None:
        raise ValueError(f"no probe layout for {dtype}")
    block_elements, block_bytes, _sub_len, _affine = geom
    nblocks = n_elements // block_elements
    out = bytearray()
    for b in range(nblocks):
        d = 0.015625 + (b % 7) * 0.001953125
        dmin = 0.00390625 + (b % 5) * 0.0009765625
        if dtype in ("Q4_K", "Q5_K"):
            body = bytes(((b * 31 + i * 7) & 0xFF)
                         for i in range(block_bytes - 4))
            out += struct.pack("<ee", d, dmin) + body
        elif dtype == "Q6_K":
            body = bytes(((b * 31 + i * 7) & 0xFF)
                         for i in range(block_bytes - 2))
            out += body + struct.pack("<e", d)
        else:
            out += bytes(((b * 31 + i * 7) & 0x3F)
                         for i in range(block_bytes))
    return bytes(out[:nblocks * block_bytes])


def make_probe_matrix(dtype: str, rows: int, cols: int):
    from .qmatrix import QuantMatrix
    data = _probe_bytes(dtype, rows * cols)
    return QuantMatrix(data, dtype, rows, cols)


def time_matvec(W, x, reps: int) -> float:
    """Best-of-`reps` seconds for one matvec.

    Best-of, not mean: a laptop's scheduler, a background process or a
    turbo-clock dip inflate individual samples and never deflate them, so
    the minimum is the closest thing to the machine's real capability.
    """
    W.matvec(x)  # warm the code path, fault the pages, resolve the JIT
    best = float("inf")
    for _ in range(max(1, reps)):
        t0 = time.perf_counter()
        W.matvec(x)
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return best


# ---- the sweep --------------------------------------------------------------

def _set_threads(n: int) -> bool:
    # through kernels, never numba directly: the per-dispatch thread scope
    # restores the count it recorded, and bypassing the setter desyncs it
    try:
        return _kernels.set_threads(n) > 0
    except Exception:
        return False


def sweep(shapes, thread_counts, reps: int = 12,
          budget_seconds: float = 25.0, log=None) -> dict:
    """Time each shape at each thread count. Returns a result record.

    `shapes` is an iterable of (label, dtype, rows, cols, weight) where
    weight is how many times per token that shape is dispatched - so the
    score is the model's actual per-token matvec cost, not an unweighted
    average that over-values a shape it runs once.
    """
    if not _kernels.available():
        return {"error": "kernels inactive: nothing to tune "
                         "(install the pinned numba, or unset ALPACCAROO_KERNELS=0)"}
    import numpy as np

    # kernels.threads(), never numba.get_num_threads(): per-dispatch thread
    # scaling leaves the latter at 1 whenever the last kernel to run was a
    # narrow one, and restoring THAT at the end would shrink the pool to one
    # thread for the rest of the process.
    original = _kernels.threads()
    if original <= 0:
        return {"error": "numba is not importable"}

    built = []
    for label, dtype, rows, cols, weight in shapes:
        try:
            W = make_probe_matrix(dtype, rows, cols)
        except Exception as e:  # unsupported dtype, or cols % block != 0
            if log:
                log(f"  skipping {label} {dtype} {rows}x{cols}: {e}")
            continue
        rng = np.random.default_rng(7)
        x = rng.standard_normal(cols).astype(np.float32)
        built.append((label, dtype, rows, cols, weight, W, x))
    if not built:
        return {"error": "no probe shape could be built"}

    t_start = time.perf_counter()
    measurements: list[dict] = []
    try:
        for n in thread_counts:
            if not _set_threads(n):
                continue
            for label, dtype, rows, cols, weight, W, x in built:
                if time.perf_counter() - t_start > budget_seconds:
                    if log:
                        log(f"  budget of {budget_seconds:.0f}s reached; "
                            f"stopping the sweep early")
                    raise TimeoutError
                secs = time_matvec(W, x, reps)
                measurements.append({
                    "threads": n, "label": label, "dtype": dtype,
                    "rows": rows, "cols": cols, "per_token_calls": weight,
                    "seconds": secs, "weights": rows * cols,
                    "gweights_per_s": (rows * cols) / secs / 1e9 if secs else None,
                })
                if log:
                    log(f"  threads={n:<3} {label:<14} {dtype} {rows}x{cols} "
                        f"{secs * 1e3:8.3f} ms  "
                        f"{(rows * cols) / secs / 1e9:6.2f} Gw/s")
    except TimeoutError:
        pass
    finally:
        _set_threads(original)

    totals: dict[int, float] = {}
    for m in measurements:
        totals[m["threads"]] = (totals.get(m["threads"], 0.0)
                                + m["seconds"] * m["per_token_calls"])
    # only thread counts that completed the whole shape set can be compared
    complete = {n: t for n, t in totals.items()
                if sum(1 for m in measurements if m["threads"] == n) == len(built)}
    if not complete:
        return {"error": "no thread count completed the probe set",
                "measurements": measurements}
    best = min(complete, key=lambda n: complete[n])
    return {
        "threads": best,
        "per_token_matvec_seconds": complete,
        "measurements": measurements,
        "default_threads": original,
        "speedup_vs_default": (complete.get(original, complete[best])
                               / complete[best]) if complete[best] > 0 else None,
    }


# ---- narrow-matrix crossover (Package E) ------------------------------------

#: Sizes to bracket the serial/parallel crossover. Powers of two either side
#: of the built-in default, so the answer is a measured bracket rather than
#: a yes/no on one shape.
CROSSOVER_ROWS = (64, 128, 256, 512, 1024, 2048)
CROSSOVER_COLS = 2048
CROSSOVER_DTYPES = ("Q4_K", "Q6_K")


def measure_crossover(rounds: int = 15, log=None) -> dict:
    """Largest matvec that is faster on one thread than on the whole pool.

    Uses :func:`alpaccaroo.bench.paired_compare`, which alternates the two
    variants round-robin. That matters more than it sounds: taken as two
    separate timing runs on a laptop that cycles its power limits, this
    measurement reported the opposite winner for one of the shapes below.

    Returns the threshold in weight elements - the largest shape where
    serial won across every probed dtype, or 0 when serial never won.
    """
    if not _kernels.available():
        return {"error": "kernels inactive"}
    import numpy as np

    from .bench import paired_compare

    pool = _kernels.threads()
    if pool <= 1:
        return {"error": f"thread pool is {pool}: nothing to compare against"}

    results = []
    serial_wins: dict[str, int] = {}
    for dtype in CROSSOVER_DTYPES:
        for rows in CROSSOVER_ROWS:
            try:
                W = make_probe_matrix(dtype, rows, CROSSOVER_COLS)
            except Exception:
                continue
            x = np.random.default_rng(1).standard_normal(
                CROSSOVER_COLS).astype(np.float32)

            def runner(n):
                def go():
                    _kernels.set_threads(n)
                    for _ in range(16):   # keep a round well above the clock
                        W.matvec(x)       # resolution
                return go

            try:
                cmp = paired_compare([("pool", runner(pool)),
                                      ("serial", runner(1))], rounds=rounds)
            finally:
                _kernels.set_threads(pool)
            ratio = cmp["ratios"]["serial"]["median_vs_baseline"]
            elems = rows * CROSSOVER_COLS
            results.append({"dtype": dtype, "rows": rows,
                            "cols": CROSSOVER_COLS, "elements": elems,
                            "serial_over_pool": ratio,
                            "serial_wins": ratio < 1.0})
            if log:
                log(f"  {dtype} {rows:>5}x{CROSSOVER_COLS} "
                    f"({elems:>8} elems)  serial/pool = {ratio:5.2f}  "
                    + ("serial wins" if ratio < 1.0 else "pool wins"))
            if ratio < 1.0:
                serial_wins[dtype] = max(serial_wins.get(dtype, 0), elems)

    # the threshold every probed dtype agrees on: taking the minimum keeps a
    # dtype whose kernel parallelizes well from being forced onto one thread
    threshold = min(serial_wins.values()) if len(
        serial_wins) == len(CROSSOVER_DTYPES) else 0
    if not serial_wins:
        threshold = 0
    elif len(serial_wins) < len(CROSSOVER_DTYPES):
        # only some dtypes ever preferred serial; be conservative and use
        # the smallest winning size rather than the largest
        threshold = min(serial_wins.values())
    return {"threshold_elements": threshold, "pool_threads": pool,
            "measurements": results,
            "builtin_default": _kernels.SERIAL_MATVEC_ELEMS_DEFAULT}


# ---- cache ------------------------------------------------------------------

def _kernel_source_digest() -> str:
    """Hash of the kernel module's own source: editing a kernel changes what
    the winning thread count is, and a stale cache would hide it."""
    try:
        src = Path(_kernels.__file__).read_bytes()
    except OSError:
        return "unknown"
    return hashlib.sha256(src).hexdigest()[:16]


def cache_key() -> dict:
    """Everything that can change the answer. Any difference is a miss."""
    from . import __version__, _platform
    from .profiling import cpu_features
    cpu = cpu_features()
    key = {
        "alpaccaroo": __version__,
        "kernels_sha256": _kernel_source_digest(),
        "os": platform.system(),
        "release": platform.release().split(".")[0],
        "arch": platform.machine(),
        "cpu": cpu.get("model"),
        "cpu_features": ",".join(cpu.get("features") or []),
        "logical_cores": os.cpu_count(),
        "physical_cores": _platform.physical_cores(),
        "python": platform.python_version(),
    }
    for mod in ("numpy", "numba", "llvmlite"):
        try:
            key[mod] = __import__(mod).__version__
        except Exception:
            key[mod] = None
    return key


def cache_dir() -> Path:
    from .store import alpaccaroo_home
    return alpaccaroo_home() / "tuning"


def cache_path(key: dict, shape_tag: str) -> Path:
    blob = json.dumps(key, sort_keys=True) + "|" + shape_tag
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]
    return cache_dir() / f"threads-{digest}.json"


def shape_tag(shapes) -> str:
    """A short, stable name for a shape class, so a 1B and an 8B model on the
    same machine keep separate cache entries."""
    parts = sorted(f"{dt}:{r}x{c}x{w}" for _l, dt, r, c, w in shapes)
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]


def load_cached(shapes) -> "dict | None":
    path = cache_path(cache_key(), shape_tag(shapes))
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if rec.get("key") != cache_key():
        return None  # belt and braces: the filename already encodes it
    return rec


def save_cached(shapes, result: dict, crossover: "dict | None" = None) -> Path:
    path = cache_path(cache_key(), shape_tag(shapes))
    path.parent.mkdir(parents=True, exist_ok=True)
    prev = load_cached(shapes) or {}
    rec = {
        "key": cache_key(),
        "shape_tag": shape_tag(shapes),
        "shapes": [list(s) for s in shapes],
        "threads": result.get("threads"),
        "per_token_matvec_seconds": result.get("per_token_matvec_seconds"),
        "speedup_vs_default": result.get("speedup_vs_default"),
        # a --crossover run and a thread-count run write the same file, so
        # neither erases what the other measured
        "serial_matvec_elements": (
            crossover.get("threshold_elements") if crossover is not None
            else prev.get("serial_matvec_elements")),
        "crossover": (crossover if crossover is not None
                      else prev.get("crossover")),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


# ---- application ------------------------------------------------------------

def autotune_enabled() -> bool:
    return (os.environ.get("ALPACCAROO_AUTOTUNE", "").strip().lower()
            in ("1", "on", "yes", "true"))


def apply_cached_threads(model, log=None) -> "int | None":
    """Set the kernel thread count from cache, if autotuning is on and a
    valid entry exists. Never benchmarks - a normal CLI start must not.

    Returns the thread count applied, or None.
    """
    if not autotune_enabled():
        return None
    if os.environ.get("ALPACCAROO_THREADS", "").strip():
        # the user chose; nothing here overrules that
        return None
    if not _kernels.available():
        return None
    shapes = model_matvec_shapes(model)
    if not shapes:
        return None
    rec = load_cached(shapes)
    if rec is None:
        if log:
            log("ALPACCAROO_AUTOTUNE=1 but no cached tuning for this machine "
                "and model shape - run `alpaccaroo tune -m <model>` once "
                "(it does not run implicitly)")
        return None
    applied = []
    elems = rec.get("serial_matvec_elements")
    if isinstance(elems, int) and elems >= 0:
        _kernels.set_tuned_serial_elems(elems)
        applied.append(f"serial below {elems} weights")
    n = rec.get("threads")
    if not isinstance(n, int) or n <= 0 or not _set_threads(n):
        if applied and log:
            log(f"autotune: {', '.join(applied)} (cached "
                f"{rec.get('created_at', 'earlier')})")
        return None
    applied.insert(0, f"{n} kernel threads")
    if log:
        log(f"autotune: {', '.join(applied)} (cached "
            f"{rec.get('created_at', 'earlier')}, "
            f"{rec.get('speedup_vs_default') or 1.0:.2f}x vs default)")
    return n
