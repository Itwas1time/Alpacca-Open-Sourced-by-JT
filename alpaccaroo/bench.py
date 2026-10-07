# Alpaccaroo - the repeatable benchmark harness. MIT License. See LICENSE.
"""Measure prefill and decode separately, cold and warm, and say where.

A number without its machine is not a measurement, so every record carries
the OS, CPU, thread counts, BLAS identity, package versions and the
per-role execution paths that produced it (see
:mod:`alpaccaroo.profiling`). Records go to stdout as a table, to JSON, or
to CSV; the JSON keeps the environment once at the top and the rows flat
underneath, the CSV flattens the environment into every row so a
spreadsheet of several machines still compares like with like.

Cold and warm are different workloads and are labelled as such:

``cold``
    the model's first use in this process - GGUF open and unpack, the JIT
    compiling or loading its cache, the first touch of every weight page.
    This is what a one-shot ``alpaccaroo run`` pays.
``warm``
    the same model after a full discarded pass, with the KV cache reset.
    This is what a resident ``alpaccaroo serve`` or an interactive session
    pays per turn.

Never quote one as the other.

Model references resolve **locally first**: a nickname, an installed
canonical name and a file path all resolve without touching the network,
and a reference that is not installed is an error unless ``--allow-pull``
is given. A benchmark that silently downloads 4 GiB is not repeatable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

from . import profiling as _prof

#: (prefill tokens, decode tokens) for the four prompt shapes the plan's
#: benchmark matrix asks for. `--shapes 512x128` names one inline instead.
SHAPES: dict[str, tuple[int, int]] = {
    "short-short": (32, 32),
    "short-long": (32, 256),
    "long-short": (1024, 32),
    "long-long": (1024, 256),
    # the historical tests/bench.py default, kept so old numbers stay
    # comparable to new ones
    "default": (512, 128),
}

_PROMPT_SEED = (
    "Once upon a time there was a small alpaca who liked clear Python. "
    "The model reads the same sentence again for a deterministic prompt. "
)


# ---- model resolution -------------------------------------------------------

def resolve_model(ref: str, allow_pull: bool = False):
    """(path, display_name) for a benchmark model reference.

    Order matters: a raw path wins, then the same nickname/installed-name
    resolution the CLI uses, and only then - with ``allow_pull`` - the
    network. The old harness went straight to ``parse_model_ref``, which
    reads a nickname as an unknown registry name and pulls it.
    """
    from .store import find_local, parse_model_ref, resolve_model_input

    p = Path(ref).expanduser()
    if p.exists():
        return p, p.name
    model_ref = parse_model_ref(resolve_model_input(ref))
    local = find_local(model_ref)
    if local is not None:
        return local.model_path, model_ref.display()
    if not allow_pull:
        raise SystemExit(
            f"alpaccaroo bench: {ref!r} is not installed locally. Benchmarks "
            f"do not download models by default - run `alpaccaroo pull "
            f"{model_ref.display()}` first, or pass --allow-pull.")
    from .pull import pull_model
    return pull_model(model_ref).model_path, model_ref.display()


# ---- memory -----------------------------------------------------------------

def peak_rss_mb() -> "float | None":
    """Peak resident set size in MiB, or None where the host cannot say.

    POSIX answers through getrusage; Windows has no such call but exposes
    the same number as PeakWorkingSetSize, so both report a real peak
    rather than one platform reporting nothing.
    """
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD),
                            ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]

            counters = _PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            k32 = ctypes.WinDLL("kernel32")
            # the pseudo-handle is (HANDLE)-1; left at the default c_int
            # restype it truncates to 32 bits and the call fails silently
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            fn = getattr(k32, "K32GetProcessMemoryInfo", None)
            if fn is None:  # pre-Win7 layout keeps it in psapi.dll
                fn = ctypes.WinDLL("psapi").GetProcessMemoryInfo
            fn.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
            if fn(k32.GetCurrentProcess(), ctypes.byref(counters),
                  counters.cb):
                return counters.PeakWorkingSetSize / (1024.0 * 1024.0)
        except Exception:
            return None
        return None
    try:
        import resource
    except Exception:
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kibibytes, macOS bytes
    return rss / (1024.0 * 1024.0) if sys.platform == "darwin" else rss / 1024.0


# ---- one measurement --------------------------------------------------------

def synthetic_prompt(model, n_tokens: int) -> list[int]:
    """A deterministic prompt of exactly `n_tokens` ids for this tokenizer."""
    ids = model.tok.encode(_PROMPT_SEED, add_bos=True)
    if not ids and model.tok.bos_id >= 0:
        ids = [model.tok.bos_id]
    if not ids:
        raise ValueError("synthetic prompt produced no tokens")
    reps = (n_tokens + len(ids) - 1) // len(ids)
    return (ids * reps)[:n_tokens]


def _percentiles(samples: list[float]) -> dict:
    ts = sorted(samples)
    if not ts:
        return {}

    def pct(p: float) -> float:
        i = max(0, min(len(ts) - 1, int(round(p * len(ts) + 0.5)) - 1))
        return ts[i]

    return {"min": ts[0], "p50": pct(0.50), "p95": pct(0.95),
            "p99": pct(0.99), "max": ts[-1], "mean": sum(ts) / len(ts)}


def run_case(model, prefill_tokens: int, decode_tokens: int, seed: int) -> dict:
    """One prefill + greedy decode pass against an already-loaded model.

    Resets the KV cache first, so a case never inherits the previous one's
    prefix and reports a prefill that did not happen.
    """
    from .sample import Sampler, SamplerParams

    model.reset()
    # reset() empties the live cache but leaves saved prefix snapshots, and
    # a restore would report a prefill tok/s for work that never ran. Every
    # case must forward its whole prompt.
    if hasattr(model, "clear_prefix_cache"):
        model.clear_prefix_cache()
    prompt = synthetic_prompt(model, prefill_tokens)
    sampler = Sampler(SamplerParams(temperature=0.0, seed=seed))
    for tid in prompt:
        sampler.accept(tid)

    t0 = time.perf_counter()
    logits = model.prefill(prompt)
    prefill_s = time.perf_counter() - t0

    latencies: list[float] = []
    decoded = 0
    stopped = "budget"
    t_decode0 = time.perf_counter()
    while decoded < decode_tokens and model.n_past < model.n_ctx:
        tid = sampler.sample(logits)
        sampler.accept(tid)
        if model.tok.is_eog(tid):
            stopped = "eog"
            break
        t1 = time.perf_counter()
        logits = model.forward(tid)
        latencies.append(time.perf_counter() - t1)
        decoded += 1
    else:
        if decoded < decode_tokens:
            stopped = "context"
    decode_s = time.perf_counter() - t_decode0

    return {
        "prefill_tokens": len(prompt),
        "prefill_seconds": prefill_s,
        "prefill_tok_per_s": len(prompt) / prefill_s if prefill_s > 0 else None,
        "decode_tokens": decoded,
        "decode_seconds": decode_s,
        "decode_tok_per_s": decoded / decode_s if decode_s > 0 else None,
        # what a user waits before the first character appears
        "time_to_first_token_seconds": (prefill_s + latencies[0]
                                        if latencies else prefill_s),
        "token_latency_seconds": _percentiles(latencies),
        "stopped": stopped,
    }


# ---- clock stability ---------------------------------------------------------

def stability_probe(seconds: float = 12.0, bucket: float = 1.5) -> dict:
    """Does this machine hold its throughput while it works?

    Runs one fixed kernel continuously and reports per-bucket throughput.
    A desktop that sustains its clocks returns spread ~1.0; a thin laptop
    cycling between its short and long power limits returns 2x or more, and
    every single-shot number taken on it is worth only as much as that
    spread allows.

    This exists because it is the difference between "this change made
    decode 30% faster" and "this run happened to land in a turbo window".
    Any A/B on a machine with a spread above ~1.3 needs
    :func:`paired_compare`, not two separate runs.
    """
    from . import kernels as _kernels
    if not _kernels.available():
        return {"supported": False,
                "reason": "kernels inactive: no fixed-cost probe to run"}
    try:
        import numpy as np

        from .tuning import make_probe_matrix
    except Exception as e:
        return {"supported": False, "reason": str(e)}

    # big enough to be memory-bound (so it measures the thing decode is
    # limited by) and small enough to fit any machine that runs a 1B model
    W = make_probe_matrix("Q4_K", 4096, 2048)
    nbytes = W.storage_nbytes()
    x = np.random.default_rng(5).standard_normal(2048).astype(np.float32)
    W.matvec(x)

    rates: list[float] = []
    t0 = time.perf_counter()
    b_start, calls = t0, 0
    while True:
        W.matvec(x)
        calls += 1
        now = time.perf_counter()
        if now - b_start >= bucket:
            rates.append(nbytes * calls / (now - b_start) / 1e9)
            b_start, calls = now, 0
        if now - t0 >= seconds:
            break
    if len(rates) < 2:
        return {"supported": False, "reason": "probe too short to bucket"}
    lo, hi = min(rates), max(rates)
    ordered = sorted(rates)
    return {
        "supported": True,
        "seconds": seconds,
        "bucket_seconds": bucket,
        "gb_per_s_buckets": [round(r, 3) for r in rates],
        "gb_per_s_min": round(lo, 3),
        "gb_per_s_median": round(ordered[len(ordered) // 2], 3),
        "gb_per_s_max": round(hi, 3),
        "spread": round(hi / lo, 3) if lo > 0 else None,
        # the first bucket is the burst window on a laptop that has just
        # been idle; the last is what it can hold
        "first_over_last": round(rates[0] / rates[-1], 3) if rates[-1] else None,
        "stable": bool(lo > 0 and hi / lo < 1.3),
    }


def paired_compare(variants, rounds: int = 12, warmup: int = 2) -> dict:
    """Alternate between variants round-robin and compare them pairwise.

    Two separate benchmark runs cannot be compared on a machine whose clock
    wanders - the earlier run may simply have been cooler. Interleaving
    them puts every variant in the same thermal window, so the per-round
    RATIO is meaningful even when the absolute numbers are not.

    `variants` is a sequence of (label, callable) where the callable runs
    one unit of work. Returns per-variant statistics plus, for each variant
    after the first, the median of the per-round ratio against the first -
    which is the number to quote.
    """
    labels = [lbl for lbl, _ in variants]
    fns = [fn for _, fn in variants]
    for fn in fns:
        for _ in range(max(0, warmup)):
            fn()
    samples: dict[str, list] = {lbl: [] for lbl in labels}
    order = list(range(len(labels)))
    for r in range(max(1, rounds)):
        # ABBA, not AABB: running the variants in a fixed order every round
        # charges whichever goes first for whatever the round's first call
        # costs - a cold branch predictor, a scheduler slice boundary - and
        # that bias does not cancel however many rounds you add. Reversing
        # on alternate rounds makes it cancel.
        for i in (order if r % 2 == 0 else order[::-1]):
            t = time.perf_counter()
            fns[i]()
            samples[labels[i]].append(time.perf_counter() - t)

    def stats(xs):
        s = sorted(xs)
        return {"min": s[0], "median": s[len(s) // 2], "max": s[-1],
                "mean": sum(s) / len(s)}

    out = {"rounds": rounds,
           "variants": {lbl: stats(samples[lbl]) for lbl in labels}}
    base = labels[0]
    out["baseline"] = base
    ratios = {}
    for lbl in labels[1:]:
        paired = [b / a if a > 0 else float("inf")
                  for a, b in zip(samples[base], samples[lbl])]
        per_round = sorted(paired)
        wins = sum(1 for v in paired if v < 1.0)
        med_base = stats(samples[base])["median"]
        med_lbl = stats(samples[lbl])["median"]
        ratios[lbl] = {
            "median_vs_baseline": per_round[len(per_round) // 2],
            "best_vs_baseline": per_round[0],
            "worst_vs_baseline": per_round[-1],
            # the ratio of the medians and the median of the ratios agree
            # when the pairing is working and diverge when it is not, so
            # both are reported rather than whichever looks better
            "median_ratio_of_medians": (med_lbl / med_base
                                        if med_base > 0 else None),
            # a sign test: with no real effect this lands near half the
            # rounds, and a spread of 4x cannot hide a consistent winner
            "rounds_won": wins,
            "rounds": len(paired),
        }
    out["ratios"] = ratios
    return out


# ---- the sweep --------------------------------------------------------------

def _flat_env(runtime: dict) -> dict:
    """The environment as scalar columns, for CSV."""
    cpu = runtime.get("cpu") or {}
    blas = runtime.get("blas") or {}
    return {
        "os": runtime.get("os"),
        "arch": runtime.get("arch"),
        "cpu_model": cpu.get("model"),
        "cpu_features": " ".join(cpu.get("features") or []),
        "physical_cores": runtime.get("physical_cores"),
        "logical_cores": runtime.get("logical_cores"),
        "numba_threads": runtime.get("numba_threads"),
        "blas_name": blas.get("name"),
        "blas_threads": blas.get("threads"),
        "backend": runtime.get("backend"),
        "backend_detail": runtime.get("backend_detail"),
        "python": runtime.get("python"),
        "numpy": runtime.get("numpy"),
        "numba": runtime.get("numba"),
        "llvmlite": runtime.get("llvmlite"),
        "kernels_active": runtime.get("kernels_active"),
        "int_dot_enabled": runtime.get("int_dot_enabled"),
        "gpu_active": runtime.get("gpu_active"),
    }


def _flat_record(rec: dict, env: dict) -> dict:
    lat = rec.get("token_latency_seconds") or {}
    flat = {k: v for k, v in rec.items() if k != "token_latency_seconds"}
    for k in ("min", "p50", "p95", "p99", "max", "mean"):
        flat[f"token_latency_{k}_s"] = lat.get(k)
    flat.update(env)
    return flat


def sweep(model_refs, shapes, ctxs, repeat: int, seed: int,
          allow_pull: bool, warm_only: bool, cold_only: bool,
          profile: bool, on_record=None) -> dict:
    """Run every (model, ctx, shape, phase, repeat) case; return the report."""
    from .model import Model

    records: list[dict] = []
    profile_snapshot = None
    for ref in model_refs:
        path, display = resolve_model(ref, allow_pull=allow_pull)
        file_mb = path.stat().st_size / (1024.0 * 1024.0)
        for ctx in ctxs:
            profiler = _prof.enable() if profile else None
            if profiler is not None:
                profiler.measure_overhead()
            t_load = time.perf_counter()
            model = Model.load(str(path), n_ctx=ctx, progress=False)
            load_s = time.perf_counter() - t_load
            paths = _prof.model_paths(model)
            try:
                for shape_name, (pf, dc) in shapes:
                    phases = []
                    if not warm_only:
                        phases.append("cold")
                    if not cold_only:
                        phases.append("warm")
                    for phase in phases:
                        if phase == "warm":
                            # a full discarded pass: JIT resolved, weight
                            # pages faulted in, allocator settled
                            run_case(model, pf, min(dc, 8), seed)
                        for i in range(repeat):
                            case = run_case(model, pf, dc, seed)
                            rec = {
                                "model": display,
                                "model_file": path.name,
                                "model_file_mb": round(file_mb, 1),
                                "storage": model._storage_description(),
                                "paths": paths["summary"],
                                "n_ctx": model.n_ctx,
                                "shape": shape_name,
                                "phase": phase,
                                "repeat": i,
                                # only the cold phase can honestly claim the
                                # load: the warm one is reusing that work
                                "load_seconds": load_s if phase == "cold" else None,
                                "peak_rss_mb": peak_rss_mb(),
                                **case,
                            }
                            records.append(rec)
                            if on_record is not None:
                                on_record(rec)
                            # cold happens once per loaded model, by
                            # definition; repeating it would measure warm
                            if phase == "cold":
                                break
            finally:
                if profiler is not None:
                    profile_snapshot = _prof.finish_snapshot(profiler, model)
                    _prof.disable()
                del model

    report = {"schema": 1, "records": records, "runtime": _prof.runtime_info()}
    if profile_snapshot is not None:
        report["profile"] = profile_snapshot
    return report


# ---- output -----------------------------------------------------------------

_TABLE_COLUMNS = (
    ("model", "MODEL", 22, "s"),
    ("shape", "SHAPE", 12, "s"),
    ("phase", "PHASE", 5, "s"),
    ("n_ctx", "CTX", 6, "d"),
    ("load_seconds", "LOAD s", 8, ".2f"),
    ("prefill_tok_per_s", "PREFILL t/s", 12, ".2f"),
    ("decode_tok_per_s", "DECODE t/s", 11, ".2f"),
    ("time_to_first_token_seconds", "TTFT s", 8, ".2f"),
    ("token_latency_p50_s", "p50 ms", 9, ".1f"),
    ("token_latency_p95_s", "p95 ms", 9, ".1f"),
    ("peak_rss_mb", "RSS MiB", 9, ".0f"),
)


def format_table(records: list[dict]) -> str:
    head = "  ".join(f"{title:<{w}}" if fmt == "s" else f"{title:>{w}}"
                     for _, title, w, fmt in _TABLE_COLUMNS)
    lines = [head, "-" * len(head)]
    for rec in records:
        lat = rec.get("token_latency_seconds") or {}
        cells = []
        for key, _title, w, fmt in _TABLE_COLUMNS:
            if key.startswith("token_latency_"):
                v = lat.get(key.split("_")[2])
                v = v * 1000.0 if v is not None else None  # report ms
            else:
                v = rec.get(key)
            if v is None:
                cells.append(f"{'-':<{w}}" if fmt == "s" else f"{'-':>{w}}")
            elif fmt == "s":
                s = str(v)
                cells.append(f"{s[:w]:<{w}}")
            else:
                cells.append(f"{v:>{w}{fmt}}")
        lines.append("  ".join(cells))
    return "\n".join(lines)


def bench_lines(records: list[dict]) -> list[str]:
    """One greppable `BENCH key=value ...` line per record.

    The shape the pre-harness `tests/bench.py` printed, kept so numbers
    recorded from it stay comparable - with `phase=` added, because that
    field is the difference between a cold and a warm claim.
    """
    out = []
    for r in records:
        lat = r.get("token_latency_seconds") or {}
        rss = r.get("peak_rss_mb")
        load = r.get("load_seconds")
        out.append(
            "BENCH "
            f"model={r['model_file']} shape={r['shape']} phase={r['phase']} "
            f"ctx={r['n_ctx']} backend={r.get('paths') or '-'} "
            f"load_s={'n/a' if load is None else f'{load:.3f}'} "
            f"prefill_tps={r.get('prefill_tok_per_s') or 0:.3f} "
            f"decode_tps={r.get('decode_tok_per_s') or 0:.3f} "
            f"p50_ms={1000.0 * lat.get('p50', 0.0):.2f} "
            f"p95_ms={1000.0 * lat.get('p95', 0.0):.2f} "
            f"rss_mb={'n/a' if rss is None else f'{rss:.1f}'}")
    return out


def write_csv(path: str, records: list[dict], runtime: dict) -> None:
    env = _flat_env(runtime)
    rows = [_flat_record(r, env) for r in records]
    if not rows:
        return
    cols: list[str] = []
    for row in rows:
        for k in row:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


# ---- argument parsing --------------------------------------------------------

def _parse_shapes(raw: str) -> list:
    out = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if item in SHAPES:
            out.append((item, SHAPES[item]))
            continue
        if "x" in item:
            a, _, b = item.partition("x")
            try:
                out.append((item, (int(a), int(b))))
                continue
            except ValueError:
                pass
        raise SystemExit(
            f"alpaccaroo bench: unknown shape {item!r}; use one of "
            f"{', '.join(SHAPES)} or PREFILLxDECODE (e.g. 512x128)")
    if not out:
        raise SystemExit("alpaccaroo bench: --shapes selected nothing")
    return out


def _parse_ints(raw: str, what: str) -> list[int]:
    try:
        return [int(v) for v in raw.split(",") if v.strip() != ""]
    except ValueError:
        raise SystemExit(f"alpaccaroo bench: {what} must be integers, got {raw!r}")


def build_parser(prog: str = "alpaccaroo bench") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=prog,
        description="repeatable prefill/decode benchmark with cold/warm "
                    "separation and full environment capture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="shapes: " + ", ".join(
            f"{k} ({v[0]}x{v[1]})" for k, v in SHAPES.items())
        + "\n\nexamples:\n"
          "  alpaccaroo bench --model qwen3bmed\n"
          "  alpaccaroo bench --model qwen3bmed --shapes long-long --json perf.json\n"
          "  python tests/bench.py --model qwen3bmed --profile-json perf.json\n")
    ap.add_argument("--model", "-m", action="append", required=True,
                    help="local GGUF path, installed name or nickname "
                         "(repeatable)")
    ap.add_argument("--shapes", default="default",
                    help="comma-separated shape names or PREFILLxDECODE")
    ap.add_argument("--prefill", type=int, default=None,
                    help="override every shape's prompt length")
    ap.add_argument("--decode", type=int, default=None,
                    help="override every shape's generated length")
    ap.add_argument("--ctx", "-c", default="0",
                    help="comma-separated context windows (0 = model default)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="warm repetitions per case (cold always runs once)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=None,
                    help="set ALPACCAROO_THREADS for this run")
    ap.add_argument("--warm-only", action="store_true")
    ap.add_argument("--cold-only", action="store_true")
    ap.add_argument("--allow-pull", action="store_true",
                    help="permit downloading a model that is not installed")
    ap.add_argument("--json", metavar="PATH", default=None,
                    help="write records + environment as JSON")
    ap.add_argument("--csv", metavar="PATH", default=None,
                    help="write records with the environment flattened in")
    ap.add_argument("--profile", action="store_true",
                    help="also print the decode breakdown from profiling.py")
    ap.add_argument("--profile-json", metavar="PATH", default=None,
                    help="write that breakdown to PATH")
    ap.add_argument("--quiet", "-q", action="store_true",
                    help="suppress the progress lines, keep the table")
    ap.add_argument("--bench-lines", action="store_true",
                    help="also print one greppable BENCH key=value line per row")
    ap.add_argument("--stability", action="store_true",
                    help="probe whether this machine holds its clocks, and "
                         "record the spread alongside the results")
    ap.add_argument("--stability-seconds", type=float, default=12.0)
    return ap


def main(argv: "list[str] | None" = None) -> int:
    args = build_parser().parse_args(argv)
    if args.warm_only and args.cold_only:
        raise SystemExit("alpaccaroo bench: --warm-only and --cold-only "
                         "are mutually exclusive")
    if args.threads is not None:
        # set before anything imports numba: the kernels read it at init
        os.environ["ALPACCAROO_THREADS"] = str(max(1, args.threads))

    shapes = _parse_shapes(args.shapes)
    if args.prefill is not None or args.decode is not None:
        shapes = [(name, (args.prefill if args.prefill is not None else pf,
                          args.decode if args.decode is not None else dc))
                  for name, (pf, dc) in shapes]
    ctxs = _parse_ints(args.ctx, "--ctx")

    def progress(rec: dict) -> None:
        if args.quiet:
            return
        print(f"  {rec['model']} {rec['shape']} {rec['phase']} "
              f"ctx={rec['n_ctx']} -> "
              f"prefill {rec['prefill_tok_per_s'] or 0:.2f} tok/s, "
              f"decode {rec['decode_tok_per_s'] or 0:.2f} tok/s",
              file=sys.stderr, flush=True)

    stability = None
    if args.stability:
        # before the sweep: it needs the machine in whatever state the user
        # started it in, not the state a full model load leaves it in
        if not args.quiet:
            print(f"probing clock stability for {args.stability_seconds:.0f}s...",
                  file=sys.stderr, flush=True)
        stability = stability_probe(args.stability_seconds)

    want_profile = bool(args.profile or args.profile_json)
    report = sweep(args.model, shapes, ctxs, args.repeat, args.seed,
                   args.allow_pull, args.warm_only, args.cold_only,
                   want_profile, on_record=progress)
    if stability is not None:
        report["stability"] = stability

    print(format_table(report["records"]))
    if stability is not None:
        if stability.get("supported"):
            print(f"\nclock stability: {stability['gb_per_s_min']:.2f}-"
                  f"{stability['gb_per_s_max']:.2f} GB/s over "
                  f"{stability['seconds']:.0f}s "
                  f"(spread {stability['spread']:.2f}x) - "
                  + ("steady, single runs are comparable"
                     if stability["stable"] else
                     "UNSTEADY: single runs on this machine are not "
                     "comparable; alternate configurations within one "
                     "process (bench.paired_compare) before believing "
                     "any A/B"))
        else:
            print(f"\nclock stability: not probed ({stability.get('reason')})")
    if args.bench_lines:
        for line in bench_lines(report["records"]):
            print(line)
    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"wrote {args.json}", file=sys.stderr)
    if args.csv:
        write_csv(args.csv, report["records"], report["runtime"])
        print(f"wrote {args.csv}", file=sys.stderr)
    snap = report.get("profile")
    if snap is not None and args.profile_json:
        Path(args.profile_json).write_text(
            json.dumps(snap, indent=2, default=str), encoding="utf-8")
        print(f"wrote {args.profile_json}", file=sys.stderr)
    if snap is not None and args.profile:
        print(_prof.format_report(snap), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
