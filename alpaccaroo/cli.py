# Alpaccaroo - command-line interface. MIT License. See LICENSE.
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__
from .sample import SamplerParams
from .store import (LocalModel, alpaccaroo_home, clear_model_nickname, find_local,
                    human_size, list_models, models_root, nickname_for_model,
                    parse_model_ref, remove_model, resolve_model_input,
                    set_model_nickname)

EXAMPLES = """\
model references:
  llama3.2:1b                      Ollama registry (registry.ollama.ai)
  ollama:user/model:tag            Ollama registry, user namespace
  hf:org/repo  |  org/repo         Hugging Face repo (best GGUF quant)
  hf:org/repo:Q4_K_M               Hugging Face repo, specific quant/file
  ./path/to/model.gguf             local GGUF file

examples:
  alpaccaroo pull llama3.2:1b
  alpaccaroo menu                                  # terminal app menu
  alpaccaroo run llama3.2:1b                        # interactive chat
  alpaccaroo run llama3.2:1b "why is the sky blue?" # one-shot
  alpaccaroo history list                           # list saved chats
  alpaccaroo serve llama3.2:1b --port 8080          # OpenAI-compatible API
  alpaccaroo run llama3.2:1b --connect              # reuse that server, no load
  alpaccaroo run llama3.2:1b --profile              # where each token's time goes
  alpaccaroo bench --model llama3.2:1b              # cold/warm prefill+decode
"""


def _resolve_or_pull(name: str, auto_pull: bool = True) -> tuple[LocalModel, str]:
    from .pull import pull_model
    resolved = resolve_model_input(name)
    ref = parse_model_ref(resolved)
    local = find_local(ref)
    if local is not None:
        return local, ref.display()
    if ref.source == "file":
        raise SystemExit(f"alpaccaroo: model file not found: {ref.path}")
    if not auto_pull:
        raise SystemExit(f"alpaccaroo: {ref.display()} is not installed "
                         f"(try `alpaccaroo pull {ref.display()}`)")
    print(f"{ref.display()} is not installed yet - pulling it first", file=sys.stderr)
    return pull_model(ref), ref.display()



def _sampler_params(args) -> SamplerParams:
    p = SamplerParams()
    if args.temp is not None:
        p.temperature = args.temp
    if args.top_k is not None:
        p.top_k = args.top_k
    if args.top_p is not None:
        p.top_p = args.top_p
    if args.repeat_penalty is not None:
        p.repeat_penalty = args.repeat_penalty
    if args.seed is not None:
        p.seed = args.seed
    return p


def _apply_manifest_defaults(local: LocalModel, args) -> None:
    """Model-supplied parameters (e.g. from an Ollama params layer) act as
    defaults; explicit flags win."""
    params = local.manifest.get("params") or {}
    if args.temp is None and "temperature" in params:
        args.temp = float(params["temperature"])
    if args.top_k is None and "top_k" in params:
        args.top_k = int(params["top_k"])
    if args.top_p is None and "top_p" in params:
        args.top_p = float(params["top_p"])
    if args.repeat_penalty is None and "repeat_penalty" in params:
        args.repeat_penalty = float(params["repeat_penalty"])
    if args.ctx == 0 and "num_ctx" in params:
        args.ctx = int(params["num_ctx"])
    if not args.system and local.manifest.get("system"):
        args.system = local.manifest["system"]


def _cgroup_limit_remaining_mb() -> float | None:
    """Remaining memory under a cgroup limit (containers), if any."""
    try:  # cgroup v2
        raw = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if raw != "max":
            used = int(Path("/sys/fs/cgroup/memory.current").read_text())
            return max(0.0, (int(raw) - used) / (1024.0 * 1024.0))
    except (OSError, ValueError):
        pass
    try:  # cgroup v1
        limit = int(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes").read_text())
        if limit < (1 << 60):  # v1 reports ~2^63 when unlimited
            used = int(Path("/sys/fs/cgroup/memory/memory.usage_in_bytes").read_text())
            return max(0.0, (limit - used) / (1024.0 * 1024.0))
    except (OSError, ValueError):
        pass
    return None


def _available_ram_mb() -> float | None:
    """Best-effort available physical RAM in MiB, standard library only."""
    try:
        if sys.platform.startswith("linux"):
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    meminfo_mb = int(line.split()[1]) / 1024.0
                    cg = _cgroup_limit_remaining_mb()
                    return min(meminfo_mb, cg) if cg is not None else meminfo_mb
        elif sys.platform == "win32":
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_uint32),
                            ("dwMemoryLoad", ctypes.c_uint32),
                            ("ullTotalPhys", ctypes.c_uint64),
                            ("ullAvailPhys", ctypes.c_uint64),
                            ("ullTotalPageFile", ctypes.c_uint64),
                            ("ullAvailPageFile", ctypes.c_uint64),
                            ("ullTotalVirtual", ctypes.c_uint64),
                            ("ullAvailVirtual", ctypes.c_uint64),
                            ("ullAvailExtendedVirtual", ctypes.c_uint64)]

            status = _MemStatus()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.ullAvailPhys / (1024.0 * 1024.0)
        elif sys.platform == "darwin":
            # macOS has no MemAvailable equivalent in the stdlib; use half
            # of physical RAM as a conservative stand-in
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            return total / (1024.0 * 1024.0) / 2.0
    except Exception:
        return None
    return None


def _auto_dense_budget_mb(avail_mb: float, file_mb: float, n_ctx: int = 0) -> int:
    """Dense-weight budget (MiB) that fits beside the quantized residue,
    KV cache, and runtime baseline, with headroom kept free.

    Conservative on purpose: it reserves the full quantized size even
    though densified matrices never allocate their quantized form, and
    spends 85% of what is left. The KV/runtime reserve scales with
    explicitly requested context windows beyond the default 4096 clamp.
    """
    reserve = 1.2 * file_mb + 2048.0 * max(1.0, n_ctx / 4096.0)
    return max(0, int(0.85 * (avail_mb - reserve)))


def _maybe_auto_dense_budget(local: LocalModel, n_ctx: int = 0) -> None:
    """Default `alpaccaroo run`/`serve` to the fastest storage this machine
    affords: size ALPACCAROO_DENSE_WEIGHT_MB from available RAM unless the
    user pinned it (any value - `0` keeps everything quantized). This is
    CLI policy; the library default (Model.load) stays opt-in."""
    from . import tensor
    if not tensor.HAS_NUMPY or os.environ.get("ALPACCAROO_F32"):
        return
    from . import kernels
    pinned = os.environ.get("ALPACCAROO_DENSE_WEIGHT_MB") is not None
    from . import cuda
    if cuda.available():
        # weights are bound for VRAM: densifying them to host float32
        # would keep them off the GPU, so the auto budget stays unset (a
        # user-pinned ALPACCAROO_DENSE_WEIGHT_MB still flows into Model.load)
        print(cuda.status(), file=sys.stderr)
        if kernels.available():
            kernels.warmup()
            print(kernels.status(), file=sys.stderr)
        return
    if kernels.available():
        # fused quantized kernels read ~1.1-1.3 B/weight at native speed:
        # faster than dense BLAS (4 B/weight) AND ~3x less RAM, so the
        # fastest default is to keep everything quantized
        kernels.warmup()
        print(kernels.status() if pinned else
              f"{kernels.status()}: keeping weights quantized "
              f"(fastest path, lowest RAM)", file=sys.stderr)
        return
    # report the missing kernels even when the budget is pinned:
    # ALPACCAROO_DENSE_WEIGHT_MB=0 is the fully-quantized mode, which is
    # exactly where their absence costs the most
    print(f"alpaccaroo: {kernels.status()}; NumPy decode is several times slower "
          f"than the fused kernels (pip install \"numba=={kernels.NUMBA_PIN}\" "
          f"to enable them)", file=sys.stderr)
    if pinned:
        return
    avail = _available_ram_mb()
    if avail is None:
        print("alpaccaroo: could not detect available RAM; keeping weights "
              "quantized (set ALPACCAROO_DENSE_WEIGHT_MB to choose a dense "
              "budget)", file=sys.stderr)
        return
    try:
        file_mb = local.model_path.stat().st_size / (1024.0 * 1024.0)
    except OSError:
        return
    # exact-fit first: if the machine can hold every densifiable matrix
    # (checked from the GGUF header against the residual quantized storage
    # and the KV cache), spend exactly that; otherwise fall back to the
    # conservative formula
    from .model import auto_budget_fit_mb
    budget = 0
    fit = auto_budget_fit_mb(str(local.model_path), n_ctx)
    if fit is not None:
        eligible_mb, fixed_mb = fit
        if eligible_mb > 0 and avail - 1024.0 >= eligible_mb + fixed_mb:
            budget = int(eligible_mb) + 1
    if budget <= 0:
        budget = _auto_dense_budget_mb(avail, file_mb, n_ctx)
    if budget <= 0:
        return
    os.environ["ALPACCAROO_DENSE_WEIGHT_MB"] = str(budget)
    print(f"auto dense-weight budget: {budget} MiB "
          f"(~{avail:.0f} MiB RAM available; "
          f"set ALPACCAROO_DENSE_WEIGHT_MB=0 to keep weights quantized)",
          file=sys.stderr)


def _load_model(local: LocalModel, args):
    from .model import Model
    _maybe_auto_dense_budget(local, args.ctx)
    print(f"loading {local.model_path.name}...", file=sys.stderr)
    m = Model.load(str(local.model_path), n_ctx=args.ctx)
    print(m.describe(), file=sys.stderr)
    return m


def _profile_wanted(args) -> bool:
    return bool(getattr(args, "profile", False)
                or getattr(args, "profile_json", None))


def _profile_begin(args):
    """Start the profiler BEFORE the model loads, so model_load time and the
    per-role execution paths land in the same record as decode."""
    if not _profile_wanted(args):
        return None
    from . import profiling
    p = profiling.enable()
    p.measure_overhead()
    return p


def _profile_end(args, profiler, model=None) -> None:
    if profiler is None:
        return
    import json

    from . import profiling
    profiling.disable()
    snap = profiling.finish_snapshot(profiler, model)
    dest = getattr(args, "profile_json", None)
    if dest:
        Path(dest).write_text(json.dumps(snap, indent=2, default=str),
                              encoding="utf-8")
        print(f"profile written to {dest}", file=sys.stderr)
    if getattr(args, "profile", False):
        print(profiling.format_report(snap), file=sys.stderr)


def cmd_pull(args) -> int:
    from .pull import pull_model
    pull_model(parse_model_ref(resolve_model_input(args.model)),
               force=args.force, verify=not args.no_verify)
    return 0


def cmd_list(_args) -> int:
    models = list_models()
    if not models:
        print("no models installed - try: alpaccaroo pull llama3.2:1b")
        return 0
    width = max(4, max(_display_width(m["name"]) for m in models))
    nicks = {m["name"]: _clip(m.get("nickname", ""), 32) for m in models}
    nick_width = max(8, max(_display_width(n) for n in nicks.values()))
    print(f"{_pad('NAME', width)}  {_pad('NICKNAME', nick_width)}  "
          f"{'SOURCE':<8}  {'SIZE':<10}  PULLED")
    for m in models:
        print(f"{_pad(m['name'], width)}  {_pad(nicks[m['name']], nick_width)}  "
              f"{m['source']:<8}  {human_size(m['size']):<10}  {m['pulled_at']}"
              f"{'  (legacy store)' if m.get('legacy_store') else ''}")
    if any(m.get("legacy_store") for m in models):
        from .store import legacy_models_roots
        roots = ", ".join(str(r) for r in legacy_models_roots())
        print(f"\nsome models still live in a pre-rebrand store ({roots}); "
              f"they are read from there as-is. Move them under "
              f"{models_root()} to consolidate.")
    return 0



def cmd_rm(args) -> int:
    rc = 0
    for name in args.models:
        ref = parse_model_ref(resolve_model_input(name))
        if remove_model(ref):
            print(f"removed {ref.display()}")
        else:
            print(f"alpaccaroo: {ref.display()} is not installed", file=sys.stderr)
            rc = 1
    return rc



def cmd_show(args) -> int:
    import json
    ref = parse_model_ref(resolve_model_input(args.model))
    local = find_local(ref)
    if local is None:
        raise SystemExit(f"alpaccaroo: {ref.display()} is not installed")
    manifest = dict(local.manifest or {"model_file": str(local.model_path)})
    # a file ref's display() is a bare path, which re-parses as a registry
    # name - so `alpaccaroo show ./tiny` would claim the registry model's alias
    nickname = "" if ref.source == "file" else nickname_for_model(ref.display())
    if nickname:
        manifest["nickname"] = nickname
    print(json.dumps(manifest, indent=2))
    if local.dir:
        print(f"\nfiles in {local.dir}:")
        for f in sorted(local.dir.iterdir()):
            if f.is_file():
                print(f"  {f.name:<28} {human_size(f.stat().st_size)}")
    if args.metadata:
        from .gguf import GGUFFile
        with GGUFFile.open(local.model_path) as gf:
            print("\nGGUF metadata:")
            for k, v in gf.metadata.items():
                s = str(v)
                print(f"  {k} = {s[:80] + '...' if len(s) > 80 else s}")
    return 0


def cmd_inspect(args) -> int:
    """Emit a deterministic, allocation-free architecture manifest."""
    import json
    from pathlib import Path

    from .qwen35 import inspect_qwen35

    path = Path(args.model).expanduser()
    if not path.is_file():
        raise ValueError(
            f"inspect requires a local GGUF file, not a model reference: {args.model}"
        )
    manifest = inspect_qwen35(
        path,
        context=args.context,
        sequences=args.sequences,
        kv_dtype=args.kv_dtype,
        cuda_vram_mb=args.cuda_vram_mb,
        cuda_reserve_mb=args.cuda_reserve_mb,
        source_url=args.source_url,
        revision=args.revision,
        calculate_hash=not args.no_hash,
        mtp_requested=args.request_mtp,
    )
    print(json.dumps(
        manifest, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    ))
    return 0



def cmd_run(args) -> int:
    # Track 6: a resident server already holds this model, and skipping the
    # load is worth more than any decode change on a machine where loading
    # a 3B takes 30-49 s against ~1 s per token. Opt-in, and it falls back
    # to loading locally rather than failing, so the one-shot CLI is
    # unchanged for everyone who does not ask for this.
    if getattr(args, "connect", None) is not None:
        from . import client
        url = args.connect or client.default_url()
        info = client.probe(url)
        if info is not None:
            return client.run_connected(url, info, args, _sampler_params(args))
        print(f"alpaccaroo: no server answering at {url}; loading the model "
              f"here instead (start one with `alpaccaroo serve {args.model}`)",
              file=sys.stderr)

    local, model_name = _resolve_or_pull(args.model)
    _apply_manifest_defaults(local, args)
    profiler = _profile_begin(args)
    model = None
    try:
        model = _load_model(local, args)
        params = _sampler_params(args)

        from . import chat
        if args.prompt:
            prompt = " ".join(args.prompt)
            messages = []
            if args.system:
                messages.append({"role": "system", "content": args.system})
            messages.append({"role": "user", "content": prompt})
            res = chat.chat_once(model, messages, params, args.n_predict,
                                 stream=lambda s: print(s, end="", flush=True))
            print()
            print(f"[{res.tokens} tokens, {res.tok_per_sec:.1f} tok/s]",
                  file=sys.stderr)
            return 0
        chat.interactive(model, params, system=args.system,
                         n_predict=args.n_predict, model_name=model_name,
                         model_path=str(local.model_path))
        return 0
    finally:
        # a Ctrl-C mid-generation still gets its report: the tokens it did
        # decode are exactly the ones worth explaining
        _profile_end(args, profiler, model)



def cmd_serve(args) -> int:
    local, model_name = _resolve_or_pull(args.model)
    _apply_manifest_defaults(local, args)
    profiler = _profile_begin(args)
    model = None
    try:
        model = _load_model(local, args)
        from .serve import serve
        host = args.host or os.environ.get("ALPACCAROO_HOST", "127.0.0.1")
        port = (args.port if args.port is not None
                else int(os.environ.get("ALPACCAROO_PORT", "8080")))
        serve(model, model_name, host, port, defaults=_sampler_params(args))
        return 0
    finally:
        # a resident server aggregates every request into one record, which
        # is the honest way to read a long-lived process's decode cost
        _profile_end(args, profiler, model)



def cmd_doctor(_args) -> int:
    from . import profiling, tensor
    print(f"alpaccaroo {__version__} (from-scratch python engine)")
    print(f"python:      {sys.version.split()[0]} ({sys.executable})")
    print(f"backend:     {tensor.backend_name()}"
          + ("  (optional accelerator active)" if tensor.HAS_NUMPY
             else "  (pip install numpy for big speedups)"))
    # the coarse label above cannot tell an einsum fallback from the
    # integer-dot kernels, and those differ by more than 2x
    print(f"path:        {tensor.backend_detail()}")
    rt = profiling.runtime_info()
    cpu = rt["cpu"]
    print(f"cpu:         {cpu.get('model') or 'unknown'} | "
          f"{rt.get('physical_cores') or '?'} physical / "
          f"{rt.get('logical_cores') or '?'} logical cores")
    print(f"cpu flags:   {', '.join(cpu.get('features') or []) or 'none detected'}"
          f"  (via {cpu.get('source')})")
    blas = rt["blas"]
    print(f"blas:        {blas.get('name') or 'unknown'} | threads "
          f"{blas.get('threads') if blas.get('threads') is not None else '?'}"
          + (f" ({blas['parallel']})" if blas.get("parallel") else ""))
    print(f"kernels:     {rt['kernels']} | threads "
          f"{rt.get('numba_threads') or '?'}")
    from . import cuda
    print(f"gpu:         {cuda.doctor_line()}")
    root = models_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
        ok = "ok"
    except OSError as e:
        ok = f"NOT WRITABLE ({e})"
    print(f"models dir:  {root} ({ok})")
    n = len(list_models())
    print(f"installed:   {n} model(s)" + ("" if n else " - try `alpaccaroo pull llama3.2:1b`"))
    return 0


def cmd_bench(args) -> int:
    from .bench import main as bench_main
    return bench_main(args.bench_args)


def cmd_tune(args) -> int:
    """Measure this machine's best kernel thread count and cache it.

    Explicit by design: this is the only place a benchmark runs, and
    `ALPACCAROO_AUTOTUNE=1` only ever *reads* what it wrote.
    """
    import json

    if args.asm is not None and not os.environ.get("NUMBA_CACHE_DIR"):
        # Numba refuses to disassemble code it loaded from its on-disk
        # cache ("Inspection disabled for cached code") and hands back an
        # empty listing, which reads as "no SIMD at all". Pointing it at a
        # fresh cache directory forces a real compile in this process. Must
        # happen before anything imports numba, which is why the --asm
        # branch is handled ahead of the availability check below.
        import tempfile
        os.environ["NUMBA_CACHE_DIR"] = tempfile.mkdtemp(prefix="alpaccaroo-asm-")

    from . import kernels, tuning
    if not kernels.available():
        print(f"alpaccaroo tune: {kernels.status()} - there is nothing to "
              f"tune. Install the pinned JIT "
              f"(pip install \"numba=={kernels.NUMBA_PIN}\") first.",
              file=sys.stderr)
        return 1

    if args.model:
        local, display = _resolve_or_pull(args.model, auto_pull=False)
        shapes = tuning.gguf_matvec_shapes(str(local.model_path))
        if not shapes:
            print(f"alpaccaroo tune: could not read matvec shapes from "
                  f"{display}; tuning the generic size ladder instead",
                  file=sys.stderr)
            shapes = [(l, d, r, c, 1) for l, d, r, c
                      in tuning.DEFAULT_PROBE_SHAPES]
            source = "generic ladder"
        else:
            source = display
    else:
        shapes = [(l, d, r, c, 1) for l, d, r, c in tuning.DEFAULT_PROBE_SHAPES]
        source = "generic ladder (pass -m MODEL to tune its exact shapes)"

    threads = ([max(1, n) for n in args.threads] if args.threads
               else tuning.candidate_thread_counts())
    cached = None if args.force else tuning.load_cached(shapes)
    if cached is not None and not args.crossover:
        print(f"cached tuning for this machine and shape class: "
              f"{cached['threads']} threads "
              f"(measured {cached.get('created_at', 'earlier')})")
        if cached.get("serial_matvec_elements") is not None:
            print(f"  narrow-matrix threshold: "
                  f"{cached['serial_matvec_elements']} weights")
        print("re-run with --force to measure again")
        return 0

    if args.asm is not None:
        kernels.warmup()
        target = args.asm or "matvec_q4k_int"
        res = kernels.inspect_asm(target)
        if "error" in res:
            print(f"alpaccaroo tune: {res['error']}", file=sys.stderr)
            return 1
        from .profiling import cpu_features
        cpu = cpu_features()
        print(f"kernel:  {res['name']}")
        print(f"cpu:     {cpu.get('model')} "
              f"[{', '.join(cpu.get('features') or []) or 'none detected'}]")
        print(f"signatures: {len(res['signatures'])}")
        print("\ninstruction census (what these Python loops became here):")
        for marker, why in kernels.ASM_MARKERS:
            n = res["counts"].get(marker, 0)
            print(f"  {marker:<12} {n:>6}   {why}")
        if not any(res["counts"].values()):
            print("\nEvery count is zero, which means the listing is empty "
                  "rather than the codegen being scalar - numba returns a "
                  "placeholder for code it loaded from its cache. Retry in a "
                  "shell with NUMBA_CACHE_DIR pointing at an empty directory.",
                  file=sys.stderr)
        if args.json:
            Path(args.json).write_text(res["asm"], encoding="utf-8")
            print(f"\nfull assembly written to {args.json}")
        else:
            print("\npass --json PATH to dump the full assembly")
        return 0

    if args.crossover:
        print("measuring the serial/parallel crossover "
              "(paired, round-robin - see kernels.SERIAL_MATVEC_ELEMS_DEFAULT)")
        cross = tuning.measure_crossover(
            log=lambda s: print(s, file=sys.stderr, flush=True))
        if "error" in cross:
            print(f"alpaccaroo tune: {cross['error']}", file=sys.stderr)
            return 1
        thr = cross["threshold_elements"]
        print(f"\nmeasured threshold: {thr} weights "
              f"(built-in default {cross['builtin_default']})")
        print("  a matvec at or below it runs on one thread; above it, "
              "the whole pool")
        base = cached or {}
        path = tuning.save_cached(
            shapes, {"threads": base.get("threads"),
                     "per_token_matvec_seconds":
                         base.get("per_token_matvec_seconds"),
                     "speedup_vs_default": base.get("speedup_vs_default")},
            crossover=cross)
        print(f"cached in {path}")
        if args.json:
            Path(args.json).write_text(
                json.dumps(cross, indent=2, default=str), encoding="utf-8")
            print(f"wrote {args.json}")
        return 0

    print(f"tuning against {source}")
    print(f"shapes ({len(shapes)} distinct, weighted by calls per token):")
    for label, dt, r, c, w in shapes:
        print(f"  {label:<14} {dt:<5} {r:>7} x {c:<6} x{w}")
    print(f"thread counts: {', '.join(str(n) for n in threads)}")
    result = tuning.sweep(shapes, threads, reps=args.reps,
                          budget_seconds=args.budget,
                          log=lambda s: print(s, file=sys.stderr, flush=True))
    if "error" in result:
        print(f"alpaccaroo tune: {result['error']}", file=sys.stderr)
        return 1

    print("\nper-token matvec cost by thread count:")
    for n in sorted(result["per_token_matvec_seconds"]):
        secs = result["per_token_matvec_seconds"][n]
        mark = "  <- selected" if n == result["threads"] else ""
        print(f"  {n:>3} threads   {secs * 1e3:8.2f} ms/token{mark}")
    path = tuning.save_cached(shapes, result)
    speedup = result.get("speedup_vs_default")
    print(f"\nselected {result['threads']} threads"
          + (f" ({speedup:.2f}x the {result['default_threads']}-thread default)"
             if speedup else ""))
    print(f"cached in {path}")
    print("set ALPACCAROO_AUTOTUNE=1 to have run/serve apply it; "
          "ALPACCAROO_THREADS always wins over both.")
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2, default=str),
                                   encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


def cmd_tokenize(args) -> int:
    import json

    local, _model_name = _resolve_or_pull(args.model, auto_pull=False)
    from .gguf import GGUFFile
    from .tokenizer import Tokenizer
    with GGUFFile.open(local.model_path) as gf:
        tok = Tokenizer.from_gguf(gf.metadata)
    text = sys.stdin.read() if args.stdin else args.text
    ids = tok.encode(
        text,
        add_bos=args.add_bos,
        parse_special=args.parse_special,
    )
    if args.ids:
        print(json.dumps(ids))
        return 0
    for i in ids:
        print(f"{i:>8}  {ascii(tok.piece(i))}")
    return 0


def cmd_nickname(args) -> int:
    if getattr(args, "list", False):
        from .store import list_nicknames
        nicks = list_nicknames()
        if not nicks:
            print("no nicknames set")
            return 0
        from .store import find_local
        installed = {m["name"] for m in list_models()}
        width = max(8, max(_display_width(n) for n in nicks))
        print(f"{_pad('NICKNAME', width)}  MODEL")
        for nick, target in sorted(nicks.items(), key=lambda kv: kv[0].lower()):
            note = ""
            if target not in installed:
                note = "   (target not installed)"
            else:
                # installed canonical names beat nicknames, so a nickname that
                # is itself an installed model's name can never be honoured
                try:
                    if find_local(parse_model_ref(nick)) is not None:
                        note = "   (shadowed by an installed model of that name)"
                except ValueError:
                    pass
            print(f"{_pad(nick, width)}  {target}{note}")
        return 0
    if not args.model:
        raise ValueError("a model is required unless --list is used")
    if args.clear:
        removed = clear_model_nickname(args.model)
        if removed:
            print(f"removed nickname '{removed}'")
        else:
            print("no nickname set")
        return 0
    if not args.nickname:
        raise ValueError("nickname text is required unless --clear is used")
    nickname, target = set_model_nickname(args.model, " ".join(args.nickname))
    print(f"nickname set: {nickname} -> {target}")
    return 0



def _display_width(text: str) -> int:
    """Terminal columns `text` occupies. CJK and emoji are double-width, and
    combining marks take none, so len() misaligns every column after them."""
    import unicodedata
    total = 0
    for ch in text:
        if unicodedata.combining(ch):
            continue
        total += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return total


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _display_width(text))


def _clip(text: str, width: int) -> str:
    text = " ".join(str(text).split())
    if _display_width(text) <= width:
        return text
    out, used = "", 0
    for ch in text:
        w = _display_width(ch)
        if used + w > max(0, width - 3):
            break
        out += ch
        used += w
    return out.rstrip() + "..."


def _default_model_file() -> Path:
    return alpaccaroo_home() / "default-model.txt"


def _read_default_model() -> str:
    try:
        value = _default_model_file().read_text(encoding="utf-8").strip()
    except OSError:
        value = ""
    if value:
        return value
    models = list_models()
    if models:
        return models[0]["name"]
    return "llama3.2:1b"


def _write_default_model(model_ref: str) -> None:
    path = _default_model_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model_ref.strip() + "\n", encoding="utf-8")


def _model_label(model_ref: str) -> str:
    nickname = nickname_for_model(model_ref)
    return f"{nickname} ({model_ref})" if nickname else model_ref


def _prompt_line(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return ""


def _menu_pause() -> None:
    _prompt_line("Press Enter to continue...")


def _menu_error(e) -> None:
    print(f"alpaccaroo: error: {e}", file=sys.stderr)


def _print_installed_models() -> None:
    print("Installed models:")
    cmd_list(argparse.Namespace())


def _menu_run_model() -> None:
    print("\nChat with a model\n")
    _print_installed_models()
    current = _read_default_model()
    print(f"\nCurrent chat model:\n  {_model_label(current)}\n")
    model_ref = _prompt_line("Model reference or nickname (blank = current): ").strip() or current
    print()
    args = argparse.Namespace(
        model=model_ref, prompt=[], ctx=0, temp=None, top_k=None, top_p=None,
        repeat_penalty=None, seed=None, n_predict=-1, system="")
    try:
        rc = cmd_run(args)
    except (RuntimeError, ValueError, SystemExit) as e:
        _menu_error(e)
        rc = 1
    if rc:
        _menu_pause()


def _menu_model_manager() -> None:
    while True:
        print("\nAlpaccaroo Model Manager\n")
        _print_installed_models()
        print("\n1. Add/download a model")
        print("2. Switch chat model")
        print("3. Rename/nickname a model")
        print("4. Show model details")
        print("5. Delete an installed model")
        print("6. Back to main menu\n")
        choice = _prompt_line("Choose an option [1-6]: ").strip()
        if choice in ("", "6"):
            return
        if choice == "1":
            print("\nEnter any supported Alpaccaroo model reference.")
            print("Examples: llama3.2:1b, qwen2.5:0.5b, "
                  "hf:NousResearch/Hermes-3-Llama-3.1-8B")
            model_ref = _prompt_line("Model reference (blank to cancel): ").strip()
            if model_ref:
                try:
                    cmd_pull(argparse.Namespace(model=model_ref, force=False,
                                                no_verify=False))
                except (RuntimeError, ValueError, SystemExit) as e:
                    _menu_error(e)
                _menu_pause()
        elif choice == "2":
            model_ref = _prompt_line(
                "New chat model NAME or nickname (blank to cancel): "
            ).strip()
            if not model_ref:
                continue
            try:
                resolved = resolve_model_input(model_ref)
                ref = parse_model_ref(resolved)
                local = find_local(ref)
            except ValueError as e:
                print(f"alpaccaroo: error: {e}", file=sys.stderr)
                local = None
            if local is None:
                print("Model is not installed or the reference is invalid.")
                print("Use Add/download first, then switch to the installed model.")
                _menu_pause()
                continue
            _write_default_model(ref.display())
            print(f"Chat model set to:\n  {_model_label(_read_default_model())}")
            _menu_pause()
        elif choice == "3":
            model_ref = _prompt_line(
                "Model NAME or current nickname to rename (blank to cancel): "
            ).strip()
            if not model_ref:
                continue
            try:
                resolved = resolve_model_input(model_ref)
                ref = parse_model_ref(resolved)
                if find_local(ref) is None or ref.source == "file":
                    print("Model is not installed or the reference is invalid.")
                    _menu_pause()
                    continue
                current = nickname_for_model(ref.display())
                if current:
                    print(f"Current nickname: {current}")
                nickname = _prompt_line(
                    "New nickname (blank to cancel, '-' to clear): "
                ).strip()
                if not nickname:
                    continue
                if nickname == "-":
                    removed = clear_model_nickname(ref.display())
                    print(f"Removed nickname: {removed}" if removed else "No nickname set.")
                else:
                    nickname, target = set_model_nickname(ref.display(), nickname)
                    print(f"Nickname set:\n  {nickname} -> {target}")
            except (RuntimeError, ValueError, SystemExit) as e:
                _menu_error(e)
            _menu_pause()
        elif choice == "4":
            model_ref = _prompt_line(
                "Model reference or nickname to inspect (blank to cancel): "
            ).strip()
            if model_ref:
                try:
                    cmd_show(argparse.Namespace(model=model_ref, metadata=False))
                except (RuntimeError, ValueError, SystemExit) as e:
                    _menu_error(e)
                _menu_pause()
        elif choice == "5":
            model_ref = _prompt_line(
                "Model reference or nickname to delete (blank to cancel): "
            ).strip()
            if not model_ref:
                continue
            confirm = _prompt_line("Type DELETE to confirm: ")
            if confirm.upper() == "DELETE":
                try:
                    cmd_rm(argparse.Namespace(models=[model_ref]))
                except (RuntimeError, ValueError, SystemExit) as e:
                    _menu_error(e)
                _menu_pause()



def _menu_history() -> None:
    while True:
        print("\nAlpaccaroo Chat History\n")
        _history_list()
        print("\n1. View a chat")
        print("2. Delete one chat")
        print("3. Delete all chat history")
        print("4. Back to main menu\n")
        choice = _prompt_line("Choose an option [1-4]: ").strip()
        if choice in ("", "4"):
            return
        if choice == "1":
            chat = _prompt_line("Chat number or ID to view (blank to cancel): ").strip()
            if chat:
                try:
                    _history_show(chat)
                except (RuntimeError, ValueError, SystemExit) as e:
                    _menu_error(e)
                _menu_pause()
        elif choice == "2":
            chat = _prompt_line("Chat number or ID to delete (blank to cancel): ").strip()
            if chat:
                try:
                    cmd_history(argparse.Namespace(history_command="rm", chats=[chat]))
                except (RuntimeError, ValueError, SystemExit) as e:
                    _menu_error(e)
                _menu_pause()
        elif choice == "3":
            confirm = _prompt_line("Type DELETE to delete all chat history: ")
            if confirm.upper() == "DELETE":
                cmd_history(argparse.Namespace(history_command="clear", yes=True))
                _menu_pause()


def _print_controls() -> None:
    print("\nAlpaccaroo Controls\n")
    print("Core commands:")
    print("  alpaccaroo menu")
    print("  alpaccaroo list")
    print("  alpaccaroo pull <model>")
    print("  alpaccaroo nickname <model> <nickname>  |  alpaccaroo nickname --list")
    print("  alpaccaroo run <model-or-nickname> [prompt text]")
    print("  alpaccaroo run <model> --connect [URL]  # use a resident server")
    print("  alpaccaroo serve <model-or-nickname> [--host HOST] [--port PORT]")
    print("  alpaccaroo history list|show|stats|rm|clear --yes")
    print("  alpaccaroo show <model> [--metadata]")
    print("  alpaccaroo rm <model> [more models...]")
    print("  alpaccaroo tokenize -m <model> -p \"text\"")
    print("  alpaccaroo doctor")
    print("\nPerformance:")
    print("  alpaccaroo run <model> --profile        # where each token's time goes")
    print("  alpaccaroo run <model> --profile-json p.json")
    print("  alpaccaroo bench --model <model>        # cold/warm prefill+decode")
    print("  alpaccaroo bench --model <model> --json b.json --csv b.csv")
    print("  alpaccaroo tune -m <model>              # measure this machine once")
    print("  alpaccaroo tune --asm                   # what the kernels compiled to")
    print("\nInteractive chat:")
    print("  Esc or /exit returns to the menu/caller")
    print("  /clear resets the current conversation")
    print("  the oldest turns are dropped automatically to fit the context window")
    print("\nUseful environment variables:")
    print("  ALPACCAROO_HOME changes the model/history/default-model store")
    print("  ALPACCAROO_DENSE_WEIGHT_MB=0 keeps weights fully quantized")
    print("  ALPACCAROO_KERNELS=0 disables optional pinned JIT kernels")
    print("  ALPACCAROO_PURE=1 forces the standard-library backend")
    print("  ALPACCAROO_THREADS pins the kernel thread count (wins over everything)")
    print("  ALPACCAROO_AUTOTUNE=1 applies a cached `alpaccaroo tune` result")
    print("  ALPACCAROO_SMALL_MATVEC_ELEMS re-tunes the quantized matvec crossover")


def cmd_menu(_args) -> int:
    """Repo-owned local terminal app menu."""
    while True:
        print("\nAlpaccaroo\n")
        _print_installed_models()
        current = _read_default_model()
        print(f"\nCurrent chat model:\n  {_model_label(current)}\n")
        print("1. Chat with current or selected model")
        print("2. Alpaccaroo doctor")
        print("3. Open Alpaccaroo shell")
        print("4. Model manager")
        print("5. Chat history")
        print("6. Saved chat statistics")
        print("7. Controls tutorial")
        print("8. Exit\n")
        choice = _prompt_line("Choose an option [1-8]: ").strip()
        if choice in ("", "8"):
            return 0
        if choice == "1":
            _menu_run_model()
        elif choice == "2":
            cmd_doctor(argparse.Namespace())
            _menu_pause()
        elif choice == "3":
            os.system("cmd" if sys.platform == "win32"
                      else os.environ.get("SHELL", "sh"))
        elif choice == "4":
            _menu_model_manager()
        elif choice == "5":
            _menu_history()
        elif choice == "6":
            _history_stats()
            _menu_pause()
        elif choice == "7":
            _print_controls()
            _menu_pause()


def _history_list() -> int:
    from .history import list_chats
    chats = list_chats()
    if not chats:
        print("no chat history yet")
        return 0
    print(f"{'#':>3}  {'ID':<25}  {'STARTED':<20}  {'TURNS':>5}  {'MODEL':<32}  TITLE")
    for i, chat in enumerate(chats, 1):
        print(f"{i:>3}  {chat['id']:<25}  "
              f"{chat['started_at']:<20}  {chat['turns']:>5}  "
              f"{_clip(chat['model'], 32):<32}  {_clip(chat['title'], 72)}")
    return 0


def _history_show(selector: str) -> int:
    from .history import message_dicts, read_chat
    chat = read_chat(selector)
    print(f"Chat:    {chat['id']}")
    print(f"Started: {chat.get('started_at', '')}")
    if chat.get("ended_at"):
        print(f"Ended:   {chat['ended_at']}")
    print(f"Model:   {chat.get('model', '')}")
    if chat.get("model_path"):
        print(f"File:    {chat['model_path']}")
    if chat.get("title"):
        print(f"Title:   {chat['title']}")
    for msg in message_dicts(chat):
        role = msg.get("role", "?")
        created = msg.get("created_at", "")
        if role == "event":
            print(f"\n--- event {created} ---")
            print(msg.get("event", ""))
            continue
        print(f"\n--- {role} {created} ---")
        content = msg.get("content", "")
        if content:
            print(content)
        stats = []
        if "tokens" in msg:
            stats.append(f"{msg['tokens']} tokens")
        if isinstance(msg.get("seconds"), (int, float)):
            stats.append(f"{msg['seconds']:.2f}s")
        if stats:
            print(f"[{', '.join(stats)}]")
    return 0


def _history_stats() -> int:
    from .history import model_stats
    rows = model_stats()
    if not rows:
        print("no downloaded models or chat history yet")
        return 0
    print(f"{'MODEL':<40}  {'INST':<4}  {'CHATS':>5}  {'RESP':>5}  "
          f"{'TOKENS':>8}  {'SECONDS':>9}  {'AVG TOK/S':>9}")
    for row in rows:
        rate = f"{row['tok_per_sec']:.1f}" if row["responses"] else "n/a"
        print(f"{_clip(row['model'], 40):<40}  "
              f"{'yes' if row['installed'] else 'no':<4}  "
              f"{row['chats']:>5}  {row['responses']:>5}  "
              f"{row['tokens']:>8}  {row['seconds']:>9.2f}  {rate:>9}")
    return 0


def cmd_history(args) -> int:
    command = args.history_command or "list"
    if command in ("list", "ls"):
        return _history_list()
    if command == "show":
        return _history_show(args.chat)
    if command == "stats":
        return _history_stats()
    if command in ("rm", "delete"):
        from .history import list_chats, resolve_chat_entry
        chats = list_chats()
        targets = []
        seen = set()
        for sel in args.chats:
            chat = resolve_chat_entry(sel, chats)
            path = chat["path"]
            if path not in seen:
                targets.append(chat)
                seen.add(path)
        for chat in targets:
            try:
                chat["path"].unlink()
            except FileNotFoundError:
                pass
            print(f"deleted {chat['id']}")
        return 0
    if command == "clear":
        from .history import clear_history, list_chats
        count = len(list_chats())
        if not args.yes:
            if count == 0:
                print("no chat history to delete")
                return 0
            print(f"this will delete {count} chat(s); rerun with --yes to confirm",
                  file=sys.stderr)
            return 1
        deleted = clear_history()
        print(f"deleted {deleted} chat(s)")
        return 0
    raise ValueError(f"unknown history command: {command}")


def _add_model_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ctx", "-c", type=int, default=0, help="context window (tokens)")
    p.add_argument("--temp", type=float, default=None, help="sampling temperature")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--repeat-penalty", type=float, default=None)
    p.add_argument("--seed", "-s", type=int, default=None)
    p.add_argument("--n-predict", "-n", type=int, default=-1,
                   help="max tokens to generate (-1 = until end)")
    p.add_argument("--system", "-sys", default="", help="system prompt")
    p.add_argument("--profile", action="store_true",
                   help="print a decode time/execution-path breakdown at exit")
    p.add_argument("--profile-json", metavar="PATH", default=None,
                   help="write the same breakdown to PATH as JSON")


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to legacy code pages (cp1252); model output
    # is arbitrary UTF-8 and must never crash the CLI.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass
    tail = sys.argv[1:] if argv is None else list(argv)
    if tail and tail[0] == "bench":
        # delegated before argparse sees it: `--help`, `--model x --model y`
        # and every other bench flag belong to bench's own parser
        from .bench import main as bench_main
        return bench_main(tail[1:])
    ap = argparse.ArgumentParser(
        prog="alpaccaroo",
        description="alpaccaroo - LLMs in your terminal, implemented in pure Python",
        epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", "-v", action="version",
                    version=f"alpaccaroo {__version__}")
    sub = ap.add_subparsers(dest="command", metavar="<command>")

    p = sub.add_parser("pull", help="download a model into ~/.alpaccaroo/models")
    p.add_argument("model")
    p.add_argument("--force", "-f", action="store_true")
    p.add_argument("--no-verify", action="store_true")
    p.set_defaults(func=cmd_pull)

    p = sub.add_parser("run", help="chat with a model (one-shot if a prompt is given)")
    p.add_argument("model")
    p.add_argument("prompt", nargs="*")
    p.add_argument("--connect", nargs="?", const="", metavar="URL",
                   help="use a running `alpaccaroo serve` instead of loading "
                        "the model (default: $ALPACCAROO_HOST:$ALPACCAROO_PORT "
                        "or http://127.0.0.1:8080); falls back to a local "
                        "load if nothing answers")
    _add_model_flags(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("serve", help="OpenAI-compatible API server")
    p.add_argument("model")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    _add_model_flags(p)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("list", aliases=["ls"], help="list installed models")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("rm", aliases=["remove"], help="remove installed models")
    p.add_argument("models", nargs="+")
    p.set_defaults(func=cmd_rm)

    p = sub.add_parser("show", help="show a model's manifest and files")
    p.add_argument("model")
    p.add_argument("--metadata", action="store_true", help="dump GGUF metadata too")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser(
        "inspect",
        help="strictly inspect a local Qwen35 GGUF without allocating model weights",
    )
    p.add_argument("model", help="path to a single-file GGUF")
    p.add_argument("--architecture-contract", action="store_true",
                   help="require the complete Qwen35 metadata/tensor contract "
                        "(currently always enabled)")
    p.add_argument("--context", "--ctx", type=int, default=0,
                   help="context used for the memory plan (default: model metadata)")
    p.add_argument("--sequences", type=int, default=1)
    p.add_argument("--kv-dtype", choices=("f32", "f16"), default="f32")
    p.add_argument("--cuda-vram-mb", type=float, default=None,
                   help="optional hard VRAM budget to compare with the plan")
    p.add_argument("--cuda-reserve-mb", type=float, default=1536.0,
                   help="driver/JIT/workspace reserve included in the plan")
    p.add_argument("--source-url", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--no-hash", action="store_true",
                   help="skip the streaming SHA-256 (development only)")
    p.add_argument("--request-mtp", action="store_true",
                   help="validate that MTP execution is unsupported (fails closed)")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("tokenize", help="show how text tokenizes for a model")
    p.add_argument("-m", "--model", required=True)
    prompt = p.add_mutually_exclusive_group(required=True)
    prompt.add_argument("-p", "--text")
    prompt.add_argument("--stdin", action="store_true",
                        help="read exact UTF-8 prompt text from standard input")
    p.add_argument("--ids", action="store_true",
                   help="print only a JSON token-ID array")
    bos = p.add_mutually_exclusive_group()
    bos.add_argument("--bos", dest="add_bos", action="store_true",
                     help="force a BOS token when the vocabulary has one")
    bos.add_argument("--no-bos", dest="add_bos", action="store_false",
                     help="suppress BOS regardless of model metadata")
    p.set_defaults(add_bos=None)
    special = p.add_mutually_exclusive_group()
    special.add_argument("--parse-special", dest="parse_special",
                         action="store_true")
    special.add_argument("--no-parse-special", dest="parse_special",
                         action="store_false")
    p.set_defaults(parse_special=False)
    p.set_defaults(func=cmd_tokenize)

    p = sub.add_parser("nickname", aliases=["nick"],
                       help="set or clear a nickname for an installed model")
    p.add_argument("model", nargs="?")
    p.add_argument("nickname", nargs="*", help="nickname text, spaces allowed")
    p.add_argument("--clear", action="store_true", help="remove this model's nickname")
    p.add_argument("--list", action="store_true",
                   help="list every nickname and what it points at")
    p.set_defaults(func=cmd_nickname)

    p = sub.add_parser("menu", help="open the local terminal app menu")
    p.set_defaults(func=cmd_menu)

    p = sub.add_parser("history", aliases=["hist"], help="manage saved chat history")
    hsub = p.add_subparsers(dest="history_command", metavar="<history-command>")
    hp = hsub.add_parser("list", aliases=["ls"], help="list saved chats")
    hp.set_defaults(func=cmd_history)
    hp = hsub.add_parser("show", help="show a saved chat")
    hp.add_argument("chat", help="chat number, full id, or unique id prefix")
    hp.set_defaults(func=cmd_history)
    hp = hsub.add_parser("stats", help="show read-only saved chat statistics")
    hp.set_defaults(func=cmd_history)
    hp = hsub.add_parser("rm", aliases=["delete"], help="delete saved chats")
    hp.add_argument("chats", nargs="+", help="chat number/id/prefix to delete")
    hp.set_defaults(func=cmd_history)
    hp = hsub.add_parser("clear", help="delete all saved chat history")
    hp.add_argument("--yes", action="store_true", help="confirm deletion")
    hp.set_defaults(func=cmd_history)
    p.set_defaults(func=cmd_history)

    p = sub.add_parser("doctor", help="check the installation")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("tune", help="measure and cache this machine's best "
                                    "kernel thread count (opt-in)")
    p.add_argument("--model", "-m", default=None,
                   help="tune the exact matvec shapes of an installed model "
                        "(read from its GGUF header, not loaded)")
    p.add_argument("--threads", type=int, nargs="+", default=None,
                   help="thread counts to try (default: 1, cores/2, cores, "
                        "logical CPUs)")
    p.add_argument("--reps", type=int, default=12,
                   help="timed repetitions per shape; best-of is reported")
    p.add_argument("--budget", type=float, default=25.0,
                   help="wall-clock ceiling for the whole sweep, seconds")
    p.add_argument("--crossover", action="store_true",
                   help="measure the narrow-matrix serial/parallel threshold "
                        "instead of the thread count")
    p.add_argument("--asm", nargs="?", const="", metavar="KERNEL",
                   help="show which SIMD instructions a kernel compiled to "
                        "on this CPU (default: matvec_q4k_int); --json dumps "
                        "the full assembly")
    p.add_argument("--force", action="store_true",
                   help="re-measure even when a valid cached result exists")
    p.add_argument("--json", metavar="PATH", default=None)
    p.set_defaults(func=cmd_tune)

    # listed for `alpaccaroo --help`, but never parsed here: main() hands the
    # whole tail to alpaccaroo.bench so its twenty flags (and its own --help)
    # stay in one place instead of being mirrored into this parser
    p = sub.add_parser("bench", add_help=False,
                       help="benchmark prefill/decode, cold and warm "
                            "(see `alpaccaroo bench --help`)")
    p.set_defaults(func=cmd_bench, bench_args=[])

    args = ap.parse_args(argv)
    if not args.command:
        if sys.stdin.isatty() and sys.stdout.isatty():
            return cmd_menu(argparse.Namespace())
        ap.print_help()
        return 0
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("", file=sys.stderr)
        return 130
    except (RuntimeError, ValueError, OSError) as e:
        print(f"alpaccaroo: error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
