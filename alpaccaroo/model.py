# Alpaccaroo - the transformer, implemented from scratch in Python.
# Llama-class decoder: RMSNorm, rotary embeddings, grouped-query attention,
# SwiGLU MLP, KV cache. Runs on NumPy when available, pure Python otherwise.
# MIT License. See LICENSE.
from __future__ import annotations

import math
import os
import sys
import time
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter as _perf

from . import profiling as _prof
from . import tensor as T
from .architecture import backend_for, registered_architectures
from .gguf import GGML_BLOCK_INFO, GGUFFile
from .memory import ConventionalKVStateAdapter, StateIdentity, StateSnapshot
from .quants import dequantize
from .tokenizer import Tokenizer

if T.HAS_NUMPY:
    import numpy as np

# rope style per architecture: "norm" rotates adjacent pairs (llama/mistral),
# "neox" rotates split halves (qwen2 & friends)
SUPPORTED_ARCHES = {
    "llama": "norm",
    "mistral": "norm",
    "qwen2": "neox",
    "qwen3": "neox",
    "stablelm": "neox",
    "gemma": "neox",
    "gemma3": "neox",
}

# The Gemma family scales the embedding by sqrt(n_embd) on the way in and uses
# GELU rather than SiLU in the MLP. Gemma 1 otherwise runs the llama-class
# decoder; Gemma 3 has its own forward pass for the extra norms and dual RoPE.
_GEMMA_ARCHES = ("gemma", "gemma3")

_KNOWN_QUANT_DTYPES = {
    "Q2_K", "Q3_K", "Q4_0", "Q4_1", "Q4_K", "Q5_0", "Q5_1", "Q5_K",
    "Q6_K", "Q8_0", "Q8_1", "Q8_K",
    "IQ1_M", "IQ1_S", "IQ2_S", "IQ2_XS", "IQ2_XXS", "IQ3_S",
    "IQ3_XXS", "IQ4_NL", "IQ4_XS",
}

# ALPACCAROO_DENSE_WEIGHT_MB densification order: NumPy's quantized matvec has
# no BLAS-class kernel, so spending RAM on dense float32 buys decode speed
# roughly in proportion to how much of a token's matvec work a matrix does.
# FFN projections dominate llama-class decode, attention q/output come next,
# k/v are smaller (GQA), and the output projection is amortized by the
# last-token-only prefill. The token embedding is only ever row-gathered, so
# it is densified solely when it doubles as a tied output projection.
_DENSIFY_TIERS: tuple[tuple[str, ...], ...] = (
    ("ffn_gate", "ffn_up", "ffn_down"),
    ("attn_q", "attn_output"),
    ("attn_k", "attn_v"),
    ("output",),
)


def _standard_state_identity(gf: GGUFFile, path: str, hp: "Hyperparams") -> StateIdentity:
    """Fingerprint conventional state without reading multi-GiB tensor payloads.

    The model fingerprint covers execution geometry and every tensor's storage
    descriptor.  The weights identity additionally covers the resolved file
    identity (size and nanosecond mtime), so replacing a GGUF invalidates
    process-local snapshots even when its architecture is unchanged.
    """

    tensors = [
        [name, info.dtype, list(info.shape), int(info.offset), int(info.n_bytes)]
        for name, info in sorted(gf.tensors.items())
    ]
    geometry = {
        "hyperparameters": vars(hp),
        "tensors": tensors,
    }
    encoded = json.dumps(
        geometry, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")
    model_fingerprint = hashlib.sha256(encoded).hexdigest()
    resolved = Path(path).resolve()
    stat = resolved.stat()
    weight_record = {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "model_fingerprint": model_fingerprint,
    }
    weights_identity = hashlib.sha256(json.dumps(
        weight_record, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")).hexdigest()
    return StateIdentity(
        architecture=hp.arch,
        model_fingerprint=model_fingerprint,
        weights_identity=weights_identity,
        adapter_identity="conventional-kv-v1",
        positioning=(
            f"{hp.rope_style}:rot={hp.n_rot}:base={hp.rope_base}:"
            f"scale={hp.rope_freq_scale}:swa={hp.rope_base_swa}"
        ),
        numerical_mode="f32-kv",
    )


def _dense_budget_bytes() -> int:
    raw = os.environ.get("ALPACCAROO_DENSE_WEIGHT_MB")
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


# ALPACCAROO_PREFIX_CACHE_MB: byte budget for the multi-slot prefix cache -
# saved KV snapshots that let `prefill` switch between interleaved
# conversations without re-prefilling each one from scratch. Unset, blank
# or unparseable means the 1024 MiB default; zero or negative disables the
# feature entirely, which is exactly the pre-feature single-cache behavior.
_PREFIX_CACHE_DEFAULT_MB = 1024.0
# _PREFIX_CACHE_MIN_MATCH is the feature's one significance threshold, in
# tokens: a slot that diverges from the incoming prompt must share at least
# this much to be considered, a restore must beat the live cache's prefix
# by at least this much (every lcp row is copied - and re-uploaded on the
# gpu tier - so a few tokens of gain never pays for it), and a context
# switch must be discarding at least this much live tail to be worth a
# snapshot.
_PREFIX_CACHE_MIN_MATCH = 256


def _prefix_cache_budget_bytes() -> int:
    raw = os.environ.get("ALPACCAROO_PREFIX_CACHE_MB")
    if raw is None or not raw.strip():
        mb = _PREFIX_CACHE_DEFAULT_MB
    else:
        try:
            mb = float(raw.strip())
        except ValueError:
            mb = _PREFIX_CACHE_DEFAULT_MB
        if not math.isfinite(mb):
            mb = _PREFIX_CACHE_DEFAULT_MB
    if mb <= 0.0:
        return 0
    try:
        return int(mb * 1024 * 1024)
    except OverflowError:
        # a finite mb so large the float product overflows: the user asked
        # for effectively unlimited, which must not collapse into the =0
        # disable-and-drop-all-slots sentinel
        return 2 ** 63


def auto_budget_fit_mb(path: str, n_ctx: int = 0):
    """Exact sizing inputs for the CLI's auto dense budget, from the GGUF
    header alone: (eligible_mb, fixed_mb) where eligible_mb is the dense
    float32 size of every matrix the densify plan could select, and
    fixed_mb covers what stays resident regardless - residual quantized
    storage (~1.3 B/weight upper bound, counted for the token embedding
    whether or not it is tied), the KV cache at the effective context, and a
    runtime baseline. Matrices with unsupported dtypes
    load dense float32 regardless of any budget and are counted in
    neither term, so mixed-format files understate fixed memory (same
    blind spot as the fallback formula). Returns None if the header
    cannot be read or the architecture is unsupported."""
    try:
        gf = GGUFFile.open(path)
    except Exception:
        return None
    try:
        arch = gf.architecture
        if arch not in SUPPORTED_ARCHES:
            return None

        def meta(key, default=None):
            return gf.get(f"{arch}.{key}", default)

        n_layer = int(meta("block_count", 0) or 0)
        n_embd = int(meta("embedding_length", 0) or 0)
        n_head = int(meta("attention.head_count", 1) or 1)
        n_kv = int(meta("attention.head_count_kv", n_head) or n_head)
        head_dim = int(meta("attention.key_length", n_embd // max(n_head, 1))
                       or n_embd // max(n_head, 1))
        train_ctx = int(meta("context_length", 4096) or 4096)
        if n_layer <= 0 or n_embd <= 0:
            return None

        tied = "output.weight" not in gf.tensors
        eligible = 0
        residual = 0
        for tier in _DENSIFY_TIERS:
            for role in tier:
                if role == "output":
                    names = ["token_embd.weight" if tied else "output.weight"]
                else:
                    names = [f"blk.{i}.{role}.weight" for i in range(n_layer)]
                for nm in names:
                    info = gf.tensors.get(nm)
                    if info is None or len(info.shape) < 2:
                        continue
                    if T.can_quantized_matvec(info.dtype, int(info.shape[0])):
                        eligible += info.n_elements * 4
        # The token embedding is resident either way: quantized if the budget
        # does not reach it, dense (already in `eligible`) if it does. Counting
        # the quantized form unconditionally makes `fixed` an upper bound
        # rather than leaving a *tied* embedding out of both terms - on a
        # Gemma 3 1B that gap was 374 MiB of unaccounted memory.
        embd = gf.tensors.get("token_embd.weight")
        if embd is not None and len(embd.shape) >= 2 and \
                T.can_quantized_matvec(embd.dtype, int(embd.shape[0])):
            residual += int(embd.n_elements * 1.3)

        ctx_eff = min(train_ctx, n_ctx) if n_ctx else min(train_ctx, 4096)
        kv_bytes = 2 * n_layer * max(ctx_eff, 0) * n_kv * head_dim * 4
        mib = 1024.0 * 1024.0
        fixed_mb = residual / mib + kv_bytes / mib + 512.0
        return eligible / mib, fixed_mb
    except Exception:
        return None
    finally:
        gf.close()


@dataclass
class Hyperparams:
    arch: str
    n_layer: int
    n_embd: int
    n_head: int
    n_kv: int
    n_ff: int
    n_vocab: int
    n_ctx_train: int
    head_dim: int
    n_rot: int
    rms_eps: float
    rope_base: float
    rope_style: str
    rope_freq_scale: float = 1.0
    rope_base_swa: float = 0.0
    sliding_window: int = 0
    sliding_layers: tuple[bool, ...] = ()
    attention_scale: float = 0.0
    final_logit_softcap: float = 0.0
    embed_scale: float = 1.0
    full_attention_period: int = 0
    # which rule decided the sliding-window layout, so a mis-detection is
    # visible rather than silently changing every layer's attention mask
    swa_rule: str = ""
    attention_scale_from_metadata: bool = False


class Layer:
    __slots__ = ("attn_norm", "wq", "wk", "wv", "wo", "bq", "bk", "bv",
                 "wqk", "wgu",
                 "q_norm", "k_norm", "post_attn_norm",
                 "ffn_norm", "w_gate", "w_up", "w_down", "post_ffw_norm")


class Model:
    # class defaults so instances built without load() (tests use __new__)
    # take the reference paths; load() sets the real values. The two gpu
    # flags flip together when the load put matrices in VRAM, which is
    # exactly the condition ALPACCAROO_GPU=0 prevents - so both stage-2 gpu
    # paths honor the same off switch as the tier itself.
    _use_kernel_attention = False
    _use_gpu_batch_attention = False
    _use_gpu_chain = False
    _gpu_chain = None          # built lazily by cuda.chain_prefill/_forward
    _gpu_chain_dead = False    # any chain failure parks it for good
    _gpu_prefill_dead = False  # ...while the device prefill path parks
    #                            only after _PREFILL_PARK_AFTER chunk
    #                            failures IN A ROW (decode chain keeps
    #                            its own verdict): one transient VRAM
    #                            spike costs one chunk, not the path
    _gpu_prefill_fails = 0     # the consecutive-failure streak
    # multi-slot prefix cache (see the "prefix cache" section): the store
    # is created lazily so __new__-built instances get a fresh dict, never
    # a class-shared one; the counters read as class zeros until touched
    _prefix_slots = None       # tuple(ids) -> (immutable StateSnapshot, nbytes)
    _prefix_bytes = 0
    _prefix_hits = 0
    _prefix_misses = 0
    _prefix_saves = 0
    _prefix_evictions = 0

    def __init__(self, hp: Hyperparams, tokenizer: Tokenizer):
        self.hp = hp
        self.tok = tokenizer
        self.layers: list[Layer] = []
        self.tok_embd = None
        self.out_norm = None
        self.output = None
        self.metadata: dict = {}
        self.weight_storage: dict = {"dense": 0, "quantized": {}, "fallback": {},
                                     "densified": [], "densified_bytes": 0,
                                     "gpu": 0, "gpu_bytes": 0}
        self.cached_ids: list[int] = []
        self.last_prefill_forwarded = 0
        self._rope_inv_freq = None
        self._rope_inv_freq_swa = None
        self._rope_cos = None
        self._rope_sin = None
        self._rope_cos_swa = None
        self._rope_sin_swa = None
        if T.HAS_NUMPY:
            half = hp.n_rot // 2
            self._rope_inv_freq = (
                hp.rope_freq_scale *
                hp.rope_base ** (-2.0 * np.arange(half, dtype=np.float32) / hp.n_rot)
            )
            if hp.rope_base_swa > 0.0:
                self._rope_inv_freq_swa = (
                    hp.rope_base_swa **
                    (-2.0 * np.arange(half, dtype=np.float32) / hp.n_rot)
                )

    # ---- loading --------------------------------------------------------

    @classmethod
    def load(cls, path: str, n_ctx: int = 0, progress: bool = True) -> "Model":
        # Architecture selection happens exactly once from the authoritative
        # GGUF header. Standard models enter the conventional compatibility
        # loader; Qwen35 validates its complete manifest before its separate
        # hybrid runtime allocates weights or mutable state.
        with GGUFFile.open(path, prefetch=False) as identity:
            architecture = identity.architecture
        backend = backend_for(architecture)
        if backend is None:
            raise ValueError(
                f"architecture '{architecture}' is not supported by the "
                f"alpaccaroo engine yet (registered: "
                f"{', '.join(registered_architectures())})"
            )
        model = backend.load_model(
            cls, path, n_ctx=n_ctx, progress=progress
        )
        model.architecture_backend = backend
        return model

    @classmethod
    def _load_standard(cls, path: str, n_ctx: int = 0, progress: bool = True) -> "Model":
        t0 = time.time()
        gf = GGUFFile.open(path, prefetch=True)
        try:
            if int(gf.get("split.count", 1) or 1) > 1:
                raise ValueError(
                    "multi-part (split) GGUFs are not supported by the python "
                    "engine yet - pick a single-file quantization")
            arch = gf.architecture
            if arch not in SUPPORTED_ARCHES:
                raise ValueError(
                    f"architecture '{arch}' is not supported by the alpaccaroo engine yet "
                    f"(supported: {', '.join(sorted(SUPPORTED_ARCHES))})")
            # fail on unreadable storage now, in milliseconds, instead of a
            # raw per-tensor error after the tokenizer and half the layers.
            # Only the tensors the loader actually consumes count: a file
            # carrying an unused auxiliary tensor in an exotic type loaded
            # fine before this check existed and must keep loading.
            n_layer_pre = int(gf.get(f"{arch}.block_count", 0) or 0)
            consumed = {"token_embd.weight", "output.weight",
                        "output_norm.weight"}
            roles = ["attn_norm.weight", "attn_q.weight", "attn_k.weight",
                     "attn_v.weight", "attn_output.weight", "attn_q.bias",
                     "attn_k.bias", "attn_v.bias", "ffn_norm.weight",
                     "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]
            if arch == "gemma3":
                roles += ["attn_q_norm.weight", "attn_k_norm.weight",
                          "post_attention_norm.weight",
                          "post_ffw_norm.weight"]
            for i in range(n_layer_pre):
                for role in roles:
                    consumed.add(f"blk.{i}.{role}")
            unreadable = sorted({info.dtype for nm, info in gf.tensors.items()
                                 if nm in consumed
                                 and info.dtype not in GGML_BLOCK_INFO})
            if unreadable:
                raise ValueError(
                    f"{path}: stores tensors as {'/'.join(unreadable)}, "
                    f"which alpaccaroo cannot read yet - pick a Q4_K_M, Q5_K_M "
                    f"or Q8_0 build of this model instead")

            def meta(key, default=None):
                return gf.get(f"{arch}.{key}", default)

            def meta_required(key) -> int:
                """A core dimension. Missing or non-positive means the file is
                unusable; say so instead of dying in int(None) or NumPy."""
                raw = meta(key)
                try:
                    value = int(raw)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"{path}: required metadata '{arch}.{key}' is "
                        f"{'missing' if raw is None else repr(raw)}") from None
                if value <= 0:
                    raise ValueError(
                        f"{path}: metadata '{arch}.{key}' must be positive, got {value}")
                return value

            n_embd = meta_required("embedding_length")
            n_head = meta_required("attention.head_count")
            n_kv = int(meta("attention.head_count_kv", n_head) or n_head)
            head_dim = int(meta("attention.key_length", n_embd // n_head) or n_embd // n_head)
            n_layer = meta_required("block_count")
            # not `or 10000.0`: that idiom coerces a stored 0.0 to the default
            # and the check below could then never fire for the one value the
            # check exists for. Absent key -> default; present key -> validated.
            _rope_base_md = meta("rope.freq_base", None)
            rope_base = 10000.0 if _rope_base_md is None else float(_rope_base_md)
            if not math.isfinite(rope_base) or rope_base <= 0.0:
                raise ValueError(
                    f"{path}: {arch}.rope.freq_base must be positive and "
                    f"finite, got {rope_base}")
            rope_base_swa = float(meta("rope.freq_base_swa", 0.0) or 0.0)
            sliding_window = int(meta("attention.sliding_window", 0) or 0)
            if arch == "gemma3" and sliding_window > 0 and rope_base_swa <= 0.0:
                rope_base_swa = 10000.0

            rope_scale_legacy = meta("rope.scale_linear", None)
            rope_scaling_type = str(meta("rope.scaling.type", "") or "").lower()
            rope_scaling_factor = float(
                meta("rope.scaling.factor",
                     rope_scale_legacy if rope_scale_legacy is not None else 1.0) or 1.0)
            rope_freq_scale = 1.0
            # llama.cpp treats a bare rope.scaling.factor as linear, so a file
            # that omits the type still scales. Applies to every architecture,
            # which is what upstream does - yarn/longrope are not implemented
            # and are deliberately left unscaled rather than scaled wrongly.
            if rope_scaling_factor > 0.0 and rope_scaling_type in ("", "linear"):
                rope_freq_scale = 1.0 / rope_scaling_factor
            elif rope_scaling_type not in ("", "linear", "none"):
                print(f"warning: {path}: rope scaling type "
                      f"'{rope_scaling_type}' is not implemented; "
                      f"running unscaled", file=sys.stderr)

            fallback_attn_scale = 1.0 / math.sqrt(head_dim)
            # Gemma 3 27B alone scales by the per-head width rather than the
            # key length. llama.cpp keys this on the layer count alone
            # (src/models/gemma3.cpp: case 62 -> LLM_TYPE_27B), so keep the
            # same rule rather than inventing a stricter one.
            if arch == "gemma3" and n_layer == 62:
                fallback_attn_scale = 1.0 / math.sqrt(n_embd / n_head)
            attention_scale = float(meta("attention.scale", fallback_attn_scale)
                                    or fallback_attn_scale)
            # llama.cpp never reads this key - it always computes the scale -
            # so anything writing it is third-party. A real scale is a
            # reciprocal square root and so lies in (0, 1]; a converter using
            # query_pre_attn_scalar semantics would write 256 instead of
            # 0.0625 and every softmax would be catastrophically mis-scaled.
            if not math.isfinite(attention_scale) or not 0.0 < attention_scale <= 1.0:
                print(f"warning: {path}: {arch}.attention.scale is "
                      f"{attention_scale}, which is not a reciprocal square "
                      f"root; using {fallback_attn_scale:.6g} instead",
                      file=sys.stderr)
                attention_scale = fallback_attn_scale

            sliding_layers: tuple[bool, ...] = ()
            full_attention_period = 0
            swa_rule = ""
            if arch == "gemma3":
                pattern = meta("attention.sliding_window_pattern", None)
                if isinstance(pattern, list):
                    if len(pattern) != n_layer:
                        raise ValueError(
                            "gemma3.attention.sliding_window_pattern has "
                            f"{len(pattern)} entries, expected {n_layer}")
                    sliding_layers = tuple(bool(v) for v in pattern)
                    swa_rule = "per-layer pattern"
                elif pattern is not None:
                    # a scalar is a period, llama.cpp's set_swa_pattern:
                    #   is_swa[i] = n == 0 or (i % n < n - 1)
                    # so 1 means "no sliding window at all" and 0 means "every
                    # layer slides". A converter writes 1 for exactly the
                    # models that have no SWA, so do not second-guess it.
                    full_attention_period = int(pattern)
                    swa_rule = "period from metadata"
                else:
                    full_attention_period = int(meta("full_attention_interval", 6) or 6)
                    swa_rule = ("period from full_attention_interval"
                                if meta("full_attention_interval") is not None
                                else "default period 6")

            # a stored 0 here silently makes every logit token-independent,
            # so validate it the same way rope.freq_base is validated above
            if arch in _GEMMA_ARCHES:
                _embed_scale_md = meta("embedding_scale", None)
                embed_scale = (math.sqrt(n_embd) if _embed_scale_md is None
                               else float(_embed_scale_md))
                if not math.isfinite(embed_scale) or embed_scale <= 0.0:
                    raise ValueError(
                        f"{path}: {arch}.embedding_scale must be positive and "
                        f"finite, got {embed_scale}")
            else:
                embed_scale = 1.0

            hp = Hyperparams(
                arch=arch,
                n_layer=n_layer,
                n_embd=n_embd,
                n_head=n_head,
                n_kv=n_kv,
                n_ff=meta_required("feed_forward_length"),
                n_vocab=int(gf.get(f"{arch}.vocab_size",
                                   len(gf.get("tokenizer.ggml.tokens", [])))),
                n_ctx_train=int(meta("context_length", 4096)),
                head_dim=head_dim,
                n_rot=int(meta("rope.dimension_count", head_dim) or head_dim),
                rms_eps=float(meta("attention.layer_norm_rms_epsilon", 1e-5)),
                rope_base=rope_base,
                rope_style=SUPPORTED_ARCHES[arch],
                rope_freq_scale=rope_freq_scale,
                rope_base_swa=rope_base_swa,
                sliding_window=sliding_window,
                sliding_layers=sliding_layers,
                attention_scale=attention_scale,
                final_logit_softcap=float(meta("final_logit_softcapping", 0.0) or 0.0),
                embed_scale=embed_scale,
                full_attention_period=full_attention_period,
                swa_rule=swa_rule,
                attention_scale_from_metadata=meta("attention.scale") is not None,
            )

            tokenizer = Tokenizer.from_gguf(gf.metadata)
            m = cls(hp, tokenizer)
            m.metadata = {k: v for k, v in gf.metadata.items()
                          if not isinstance(v, list) or len(v) < 64}
            dense_matrices = 0
            dense_bytes = 0
            quantized_bytes = 0
            quantized_matrices: dict[str, int] = {}
            fallback_matrices: dict[str, int] = {}
            densified_names: list[str] = []

            # ALPACCAROO_DENSE_WEIGHT_MB: pick which quantizable matrices to
            # expand to dense float32 at load (BLAS-speed decode), spending
            # the budget tier by tier; everything else stays quantized.
            densify_plan: set[str] = set()
            densified_bytes = 0
            budget = 0
            if T.HAS_NUMPY and not os.environ.get("ALPACCAROO_F32"):
                budget = _dense_budget_bytes()
            if budget > 0:
                tied_output = "output.weight" not in gf.tensors
                for tier in _DENSIFY_TIERS:
                    for role in tier:
                        if role == "output":
                            names = ["token_embd.weight" if tied_output
                                     else "output.weight"]
                        else:
                            names = [f"blk.{i}.{role}.weight"
                                     for i in range(hp.n_layer)]
                        for nm in names:
                            info = gf.tensors.get(nm)
                            if info is None or len(info.shape) < 2:
                                continue
                            if not T.can_quantized_matvec(
                                    info.dtype, int(info.shape[0])):
                                continue  # loads dense anyway, costs no budget
                            nbytes = info.n_elements * 4
                            if densified_bytes + nbytes <= budget:
                                densify_plan.add(nm)
                                densified_bytes += nbytes

            # GPU tier: quantized matrices upload their codes-layout arrays
            # to VRAM at load; dispatch is per-matrix, so a partial upload
            # (VRAM exhausted mid-load) is exact mixed placement, not an
            # error. Anything the tier declines falls back to the exact
            # QuantMatrix it would have been. The token embedding never
            # uploads: row gathers are the wrong shape for these kernels.
            gpu_matrices = 0
            gpu_bytes = 0
            _gpu = None
            if T.HAS_NUMPY and not os.environ.get("ALPACCAROO_F32"):
                try:
                    from . import cuda as _gpu_mod
                    if _gpu_mod.available():
                        _gpu = _gpu_mod
                except Exception:
                    _gpu = None

            def tensor_mat(name, rows, cols, required=True):
                nonlocal dense_matrices, dense_bytes, quantized_bytes
                nonlocal gpu_matrices, gpu_bytes
                info = gf.tensors.get(name)
                if info is None:
                    if required:
                        raise ValueError(f"missing tensor {name} in {path}")
                    return None
                if info.n_elements != rows * cols:
                    raise ValueError(
                        f"tensor {name} has {info.n_elements} elements, "
                        f"expected {rows * cols}")
                if (T.HAS_NUMPY and not os.environ.get("ALPACCAROO_F32") and
                        name not in densify_plan and
                        T.can_quantized_matvec(info.dtype, cols)):
                    if _gpu is not None and name != "token_embd.weight":
                        g = _gpu.gpu_matrix(gf.tensor_bytes(name),
                                            info.dtype, rows, cols)
                        if g is not None:
                            # counted as GPU storage only: no QuantMatrix is
                            # built, so these bytes never exist in host RAM
                            gpu_matrices += 1
                            gpu_bytes += g.vram_nbytes
                            return g
                    quantized_matrices[info.dtype] = (
                        quantized_matrices.get(info.dtype, 0) + 1)
                    quantized_bytes += info.n_bytes
                    return T.quantized_matrix(gf.tensor_bytes(name), info.dtype, rows, cols)
                if name in densify_plan:
                    densified_names.append(name)
                elif info.dtype in _KNOWN_QUANT_DTYPES:
                    # counted for describe(); the user-facing warning is
                    # predicted from the header before the layer loop
                    fallback_matrices[info.dtype] = (
                        fallback_matrices.get(info.dtype, 0) + 1)
                dense_matrices += 1
                dense_bytes += info.n_elements * 4
                vals = dequantize(gf.tensor_bytes(name), info.n_elements, info.dtype)
                return T.matrix(vals, rows, cols)

            # Fuse row-adjacent same-dtype quantized pairs (attn_q+attn_k,
            # ffn_gate+ffn_up) into one matrix: GGUF blocks are row-major, so
            # concatenating the raw bytes of two (r, cols) tensors IS a valid
            # (r1+r2, cols) matrix of the same dtype. One kernel launch and
            # one activation quantization instead of two; the per-row math is
            # unchanged. attn_v stays separate - it is Q6_K in Q4_K_M files
            # while q/k are Q4_K, and mixed dtypes cannot share blocks.
            fuse_enabled = (T.HAS_NUMPY and not os.environ.get("ALPACCAROO_F32")
                            and os.environ.get("ALPACCAROO_FUSE", "").strip()
                            .lower() not in ("0", "off", "no"))

            def fused_mat(names, rows_each, cols):
                nonlocal quantized_bytes, gpu_matrices, gpu_bytes
                if not fuse_enabled:
                    return None
                infos = [gf.tensors.get(nm) for nm in names]
                if any(i is None for i in infos):
                    return None
                dt = infos[0].dtype
                if any(i.dtype != dt for i in infos):
                    return None
                if any(nm in densify_plan for nm in names):
                    return None
                if not T.can_quantized_matvec(dt, cols):
                    return None
                for nm, info, r in zip(names, infos, rows_each):
                    if info.n_elements != r * cols:
                        raise ValueError(
                            f"tensor {nm} has {info.n_elements} elements, "
                            f"expected {r * cols}")
                # join accepts the mmap-backed memoryviews directly: one copy
                # into the fused buffer, not a bytes() transient per tensor
                data = b"".join(gf.tensor_bytes(nm) for nm in names)
                if _gpu is not None:
                    g = _gpu.gpu_matrix(data, dt, sum(rows_each), cols)
                    if g is not None:
                        # GPU storage only - see tensor_mat
                        gpu_matrices += 1
                        gpu_bytes += g.vram_nbytes
                        return g
                for info in infos:
                    quantized_matrices[dt] = quantized_matrices.get(dt, 0) + 1
                    quantized_bytes += info.n_bytes
                return T.quantized_matrix(data, dt, sum(rows_each), cols)

            def tensor_vec(name, required=True, size=None):
                info = gf.tensors.get(name)
                if info is None:
                    if required:
                        raise ValueError(f"missing tensor {name} in {path}")
                    return None
                if size is not None and info.n_elements != size:
                    raise ValueError(
                        f"tensor {name} has {info.n_elements} elements, "
                        f"expected {size}")
                vals = dequantize(gf.tensor_bytes(name), info.n_elements, info.dtype)
                return T.vector(vals)

            # A quantization the engine cannot matvec is dequantized to dense
            # float32 at load - Q2_K weights become 8x their file size, which
            # neither budget formula accounts for. The header alone predicts
            # it, so say so BEFORE the RAM goes, not after the layer loop.
            if progress:
                pre_fb: dict[str, int] = {}
                pre_shape: dict[str, int] = {}
                fb_bytes = 0
                quantize_ok = (T.HAS_NUMPY
                               and not os.environ.get("ALPACCAROO_F32"))
                for nm, info in gf.tensors.items():
                    if (len(info.shape) < 2 or nm in densify_plan
                            or info.dtype not in _KNOWN_QUANT_DTYPES):
                        continue
                    cols0 = int(info.shape[0])
                    if quantize_ok and T.can_quantized_matvec(info.dtype, cols0):
                        continue
                    pre_fb[info.dtype] = pre_fb.get(info.dtype, 0) + 1
                    fb_bytes += info.n_elements * 4
                    # distinguish "this format has no kernel" from "this
                    # matrix is the wrong width for one": Q4_K/Q6_K need a
                    # multiple of 256 columns, and a third-party conversion
                    # that ignores that goes fully dense with no other clue
                    if T.can_quantized_matvec(info.dtype, 256):
                        pre_shape[info.dtype] = cols0
                if pre_fb and quantize_ok:
                    fb_mb = fb_bytes / (1024 * 1024)
                    size = (f"{fb_mb / 1024:.1f} GiB" if fb_mb >= 1024
                            else f"{fb_mb:.0f} MiB")
                    # a file can hit both causes at once, so report them apart
                    unsupported = sorted(set(pre_fb) - set(pre_shape))
                    reasons = []
                    if pre_shape:
                        reasons.append(
                            "%s needs a column count that is a multiple of "
                            "its block size and this file has %s"
                            % ("/".join(sorted(pre_shape)),
                               "/".join(str(pre_shape[d])
                                        for d in sorted(pre_shape))))
                    if unsupported:
                        reasons.append("alpaccaroo has no quantized matvec for %s"
                                       % "/".join(unsupported))
                    print(f"warning: {'; '.join(reasons)}, so "
                          f"{sum(pre_fb.values())} matrices load as dense "
                          f"float32 ({size}) - no memory budget accounts for "
                          f"this", file=sys.stderr)

            kv_dim = hp.n_kv * hp.head_dim
            q_dim = hp.n_head * hp.head_dim
            m.tok_embd = tensor_mat("token_embd.weight", hp.n_vocab, hp.n_embd)
            for i in range(hp.n_layer):
                if progress:
                    print(f"\rloading layers {i + 1}/{hp.n_layer}", end="",
                          flush=True, file=sys.stderr)
                p = f"blk.{i}."
                ly = Layer()
                ly.q_norm = None
                ly.k_norm = None
                ly.post_attn_norm = None
                ly.post_ffw_norm = None
                ly.attn_norm = tensor_vec(p + "attn_norm.weight", size=hp.n_embd)
                ly.wqk = fused_mat([p + "attn_q.weight", p + "attn_k.weight"],
                                   [q_dim, kv_dim], hp.n_embd)
                if ly.wqk is not None:
                    ly.wq = ly.wk = None
                else:
                    ly.wq = tensor_mat(p + "attn_q.weight", q_dim, hp.n_embd)
                    ly.wk = tensor_mat(p + "attn_k.weight", kv_dim, hp.n_embd)
                ly.wv = tensor_mat(p + "attn_v.weight", kv_dim, hp.n_embd)
                ly.wo = tensor_mat(p + "attn_output.weight", hp.n_embd, q_dim)
                ly.bq = tensor_vec(p + "attn_q.bias", required=False, size=q_dim)
                ly.bk = tensor_vec(p + "attn_k.bias", required=False, size=kv_dim)
                ly.bv = tensor_vec(p + "attn_v.bias", required=False, size=kv_dim)
                if arch == "gemma3":
                    # q/k-norm are per-head vectors shared across heads
                    ly.q_norm = tensor_vec(p + "attn_q_norm.weight",
                                           size=hp.head_dim)
                    ly.k_norm = tensor_vec(p + "attn_k_norm.weight",
                                           size=hp.head_dim)
                    ly.post_attn_norm = tensor_vec(p + "post_attention_norm.weight",
                                                   size=hp.n_embd)
                ly.ffn_norm = tensor_vec(p + "ffn_norm.weight", size=hp.n_embd)
                ly.wgu = fused_mat([p + "ffn_gate.weight", p + "ffn_up.weight"],
                                   [hp.n_ff, hp.n_ff], hp.n_embd)
                if ly.wgu is not None:
                    ly.w_gate = ly.w_up = None
                else:
                    ly.w_gate = tensor_mat(p + "ffn_gate.weight", hp.n_ff, hp.n_embd)
                    ly.w_up = tensor_mat(p + "ffn_up.weight", hp.n_ff, hp.n_embd)
                ly.w_down = tensor_mat(p + "ffn_down.weight", hp.n_embd, hp.n_ff)
                if arch == "gemma3":
                    ly.post_ffw_norm = tensor_vec(p + "post_ffw_norm.weight",
                                                  size=hp.n_embd)
                m.layers.append(ly)
            if progress:
                print("\r" + " " * 40 + "\r", end="", flush=True, file=sys.stderr)

            m.out_norm = tensor_vec("output_norm.weight", size=hp.n_embd)
            m.output = tensor_mat("output.weight", hp.n_vocab, hp.n_embd, required=False)
            if m.output is None:
                m.output = m.tok_embd  # tied embeddings

            m.weight_storage = {
                "dense": dense_matrices,
                "dense_bytes": dense_bytes,
                "quantized_bytes": quantized_bytes,
                "quantized": dict(sorted(quantized_matrices.items())),
                "fallback": dict(sorted(fallback_matrices.items())),
                "densified": sorted(densified_names),
                "densified_bytes": densified_bytes,
                "gpu": gpu_matrices,
                "gpu_bytes": gpu_bytes,
            }

            m.n_ctx = min(n_ctx, hp.n_ctx_train) if n_ctx else min(hp.n_ctx_train, 4096)
            m._init_cache()
            m.model_path = str(Path(path).resolve())
            m._state_adapter = ConventionalKVStateAdapter(
                m, _standard_state_identity(gf, path, hp),
            )
            # gpu_matrices counts too: attention and rope stay on the CPU in
            # the GPU tier, so a fully-GPU weight placement (possible when the
            # embedding is dense) still wants the JIT attention kernels
            if (quantized_matrices or gpu_matrices) and T.HAS_NUMPY:
                from . import kernels
                if kernels.available():
                    kernels.warmup()  # JIT compile/cache-load counts as load
                    m._use_kernel_attention = True
                    # opt-in only, and a cache READ only: `alpaccaroo tune`
                    # is the sole place a benchmark runs, so a normal load
                    # never pays for one. Env-gated so the import itself
                    # costs nothing when the feature is off.
                    if os.environ.get("ALPACCAROO_AUTOTUNE"):
                        from . import tuning
                        tuning.apply_cached_threads(
                            m, log=lambda s: print(s, file=sys.stderr))
            if gpu_matrices and _gpu is not None:
                _gpu.warmup()  # same rule: JIT compile counts as load time
                # stage-2 gpu paths: prefill batch attention and the
                # device decode chain. Both fall back per call / per
                # instance to the exact paths below, so flipping the
                # flags is safe even for geometries the kernels decline.
                m._use_gpu_batch_attention = True
                m._use_gpu_chain = True
            m.load_seconds = time.time() - t0
            p = _prof.ACTIVE
            if p is not None:
                p.add("model_load", m.load_seconds)
                p.set("model_path", path)
                p.set("model_describe", m.describe())
                p.set("paths", _prof.model_paths(m))
            return m
        finally:
            gf.close()

    # ---- KV cache -------------------------------------------------------

    def _init_cache(self):
        """Allocate the KV cache at the full context for every layer.

        Sliding-window layers only ever read the last `sliding_window` rows, so
        this over-allocates - 1.35 GiB at a 32k context on Gemma 3, though the
        default 4096 clamp bounds it to ~154 MiB and np.zeros is lazily mapped,
        so nothing is touched until it is written.

        A ring buffer indexed by `pos % window` would reclaim that, and it
        would be WRONG here. `_truncate_cache` plus `prefill`'s prefix reuse can
        restart at position 100 after the cache has reached 3000, and slot
        `100 % window` would still hold position 2659's K. Any fix has to keep
        absolute-position semantics or invalidate the cache on truncation. Note
        also that the batch path writes up to `window + chunk - 1` rows, so a
        window-sized buffer is too small at the default chunk of 256.
        """
        hp = self.hp
        if T.HAS_NUMPY:
            self.cache_k = [np.zeros((self.n_ctx, hp.n_kv, hp.head_dim), dtype=np.float32)
                            for _ in range(hp.n_layer)]
            self.cache_v = [np.zeros((self.n_ctx, hp.n_kv, hp.head_dim), dtype=np.float32)
                            for _ in range(hp.n_layer)]
            if self._rope_cos is None or len(self._rope_cos) != self.n_ctx:
                theta = (np.arange(self.n_ctx, dtype=np.float32)[:, None] *
                         self._rope_inv_freq[None, :])
                self._rope_cos = np.cos(theta)
                self._rope_sin = np.sin(theta)
            if self._rope_inv_freq_swa is not None and (
                    self._rope_cos_swa is None or
                    len(self._rope_cos_swa) != self.n_ctx):
                theta = (np.arange(self.n_ctx, dtype=np.float32)[:, None] *
                         self._rope_inv_freq_swa[None, :])
                self._rope_cos_swa = np.cos(theta)
                self._rope_sin_swa = np.sin(theta)
        else:
            self.cache_k = [[] for _ in range(hp.n_layer)]
            self.cache_v = [[] for _ in range(hp.n_layer)]
        self.n_past = 0
        self.cached_ids = []
        self.last_prefill_forwarded = 0
        self._chain_invalidate(0)  # fresh arrays: nothing mirrored survives

    def reset(self, *, clear_prefixes: bool = False):
        """Reset live K/V state; saved prefixes survive unless requested."""

        adapter = getattr(self, "_state_adapter", None)
        if adapter is None:
            self._init_cache()
            if clear_prefixes:
                self.clear_prefix_cache()
            return
        adapter.reset(clear_prefixes=clear_prefixes)

    def _truncate_cache(self, n_tokens: int) -> None:
        """Keep only the first `n_tokens` KV entries."""
        adapter = getattr(self, "_state_adapter", None)
        if adapter is not None:
            adapter.truncate(n_tokens)
            return
        n_tokens = max(0, min(n_tokens, self.n_past))
        if T.HAS_NUMPY:
            self.n_past = n_tokens
        else:
            for li in range(self.hp.n_layer):
                del self.cache_k[li][n_tokens:]
                del self.cache_v[li][n_tokens:]
            self.n_past = n_tokens
        del self.cached_ids[n_tokens:]
        self._chain_invalidate(n_tokens)  # rows past here will be rewritten

    def snapshot(self) -> StateSnapshot:
        """Capture a complete, immutable conventional K/V state snapshot."""

        adapter = getattr(self, "_state_adapter", None)
        if adapter is None:
            raise RuntimeError("state snapshots require a normally loaded model")
        return adapter.snapshot()

    snapshot_state = snapshot

    def restore(self, snapshot: StateSnapshot) -> None:
        """Atomically restore a validated state snapshot."""

        adapter = getattr(self, "_state_adapter", None)
        if adapter is None:
            raise RuntimeError("state restore requires a normally loaded model")
        adapter.restore(snapshot)

    restore_state = restore

    def truncate(self, n_tokens: int) -> None:
        """Public state truncation operation used by context managers."""

        self._truncate_cache(n_tokens)

    def clear_prefix_cache(self) -> None:
        """Release saved prefix snapshots without resetting live K/V state."""

        adapter = getattr(self, "_state_adapter", None)
        if adapter is not None:
            adapter.clear_prefixes()
            return
        if self._prefix_slots is not None:
            self._prefix_slots.clear()
        self._prefix_bytes = 0

    def state_description(self) -> dict:
        adapter = getattr(self, "_state_adapter", None)
        if adapter is None:
            row_values = self.hp.n_kv * self.hp.head_dim
            return {
                "position": self.n_past,
                "layers": self.hp.n_layer,
                "attention_bytes": self.n_past * self.hp.n_layer * row_values * 8,
            }
        return adapter.describe()

    def _chain_invalidate(self, pos: int) -> None:
        """Report a host KV-cache mutation from row `pos` on, so the gpu
        decode chain's device mirror re-uploads exactly what changed.

        The complete mutation set, each of which calls here: _init_cache
        (load and reset), _truncate_cache (prefill's prefix restart),
        forward_batch and _forward_np's cache-row writes, and their
        gemma3 twins. The pure-Python paths append to list caches and
        cannot coexist with a chain: no NumPy means no gpu tier at all.
        A no-op until cuda.chain_forward builds a chain."""
        ch = self._gpu_chain
        if ch is not None:
            ch.invalidate(pos)

    # ---- prefix cache (saved KV snapshots) ------------------------------

    def _prefix_cache_route(self, tokens: list[int], n: int) -> int:
        """Multi-slot prefix reuse for `prefill`: `n` is the incoming
        prompt's longest common prefix (LCP) with the LIVE cache. When a
        saved snapshot shares a sufficiently longer prefix (at least
        _PREFIX_CACHE_MIN_MATCH tokens past the live cache's - the copy
        has to be worth it), snapshot the live context if
        it is worth keeping, restore the winning slot, and return the new
        (longer) trusted prefix; otherwise return `n` unchanged. Either
        way `prefill` proceeds with its normal suffix logic.

        Gated on NumPy: the pure-Python tier keeps list caches, and the
        byte-parity argument here rests on float32 array row copies - so
        that tier keeps the exact single-cache behavior it has today, the
        same guarantee as ALPACCAROO_PREFIX_CACHE_MB=0."""
        budget = _prefix_cache_budget_bytes()
        if budget <= 0 or not T.HAS_NUMPY:
            if self._prefix_slots:
                # the budget dropped to zero mid-run: release the RAM now
                self._prefix_slots.clear()
                self._prefix_bytes = 0
            return n
        store = self._prefix_slots
        if store is None:
            store = self._prefix_slots = {}
        while store and self._prefix_bytes > budget:
            # the budget shrank mid-run: honor the new cap before serving
            # from the store (dict order IS the LRU order: drop the front)
            self._prefix_bytes -= store.pop(next(iter(store)))[-1]
            self._prefix_evictions += 1

        # best saved slot: usable when its ids are a full prefix of the
        # incoming prompt or share at least _PREFIX_CACHE_MIN_MATCH tokens,
        # and a restore must beat the live cache's prefix by a full
        # _PREFIX_CACHE_MIN_MATCH to be worth the all-rows copy. Ties and
        # near-ties keep the live cache (no copy beats a copy).
        ids = tuple(tokens)
        best_key = None
        best_lcp = n + _PREFIX_CACHE_MIN_MATCH - 1
        for key in store:
            m = min(len(key), len(ids))
            if key[:m] == ids[:m]:
                lcp = m       # one side is a prefix of the other
            else:
                lcp = 0
                while lcp < m and key[lcp] == ids[lcp]:
                    lcp += 1
                if lcp < _PREFIX_CACHE_MIN_MATCH:
                    continue  # diverges too early to be worth the copy
            if lcp > best_lcp:
                best_key, best_lcp = key, lcp

        # context-switch detection: a restore - or the heavy truncate that
        # `prefill` is about to do anyway - is about to discard a live tail
        # worth keeping. The test is the ABSOLUTE size of the discarded
        # tail, not its share of the context: interleaved conversations
        # behind a long shared system prompt discard only their unique
        # tails, and those are exactly what the slots exist to bring back.
        # Same-conversation turns extend their own prefix (n == live) and
        # never pass this test, so the hot path stays copy-free; this is
        # the ONLY place snapshots are taken.
        live = len(self.cached_ids)
        if live - n >= _PREFIX_CACHE_MIN_MATCH:
            self._prefix_cache_save(budget, protect=best_key)

        if best_key is None:
            if n < live and n < len(ids):
                # a genuine divergence from the live context that no slot
                # could serve; live-cache extensions and prompts the live
                # cache already fully contains are neither hit nor miss
                self._prefix_misses += 1
            return n
        self._prefix_hits += 1
        slot = store.pop(best_key)
        store[best_key] = slot  # LRU: most recently used moves to the back
        self._prefix_cache_restore(best_key, slot, best_lcp)
        return best_lcp

    def _prefix_cache_save(self, budget: int, protect: tuple | None = None) -> None:
        """Snapshot the live rows [0:n_past) of every layer as float32
        copies, keyed by the exact token ids they were computed from.
        Called only at context-switch time (_prefix_cache_route). `protect`
        is the slot the caller is about to restore: eviction must never
        take it, and when even evicting every OTHER slot cannot make room,
        the save loses - the restore is the prompt actually being served."""
        store = self._prefix_slots
        key = tuple(self.cached_ids)
        if key in store:
            store[key] = store.pop(key)  # same ids, same rows: touch LRU
            return
        n_rows = self.n_past
        if n_rows <= 0 or len(key) != n_rows:
            return  # nothing to keep / caches out of step: save nothing
        adapter = getattr(self, "_state_adapter", None)
        if adapter is not None:
            snapshot = adapter.snapshot()
            row_bytes = snapshot.memory_bytes()
        else:
            snapshot = None
            row_bytes = sum(self.cache_k[li][:n_rows].nbytes +
                            self.cache_v[li][:n_rows].nbytes
                            for li in range(self.hp.n_layer))
        if row_bytes > budget:
            return  # one snapshot larger than the whole budget: never saved
        while self._prefix_bytes + row_bytes > budget:
            # dict order IS the LRU order: evict from the front
            oldest = next((k for k in store if k != protect), None)
            if oldest is None:
                return  # only the protected slot is left: skip the save
            self._prefix_bytes -= store.pop(oldest)[-1]
            self._prefix_evictions += 1
        if snapshot is not None:
            store[key] = (snapshot, row_bytes)
        else:
            ks = [self.cache_k[li][:n_rows].copy() for li in range(self.hp.n_layer)]
            vs = [self.cache_v[li][:n_rows].copy() for li in range(self.hp.n_layer)]
            store[key] = (ks, vs, row_bytes)
        self._prefix_bytes += row_bytes
        self._prefix_saves += 1

    def _prefix_cache_restore(self, key: tuple, slot: tuple, lcp: int) -> None:
        """Copy a saved slot's rows [0:lcp) back into the live cache.

        Goes through _truncate_cache(0) first so every invalidation hook
        fires exactly as it does for any host mutation - the gpu decode
        chain's mirror sees a truncation followed by rewritten rows
        (invalidate-before-write, the same contract as forward_batch) and
        lazily re-uploads them on its next use. All lcp rows are copied,
        never just the part past the live LCP: the slot's rows are the
        bytes a prefill of these ids produced, and re-using live rows
        computed under different chunk boundaries would break the
        byte-parity guarantee."""
        if len(slot) == 2 and isinstance(slot[0], StateSnapshot):
            self.restore(slot[0])
            if lcp != self.n_past:
                self._truncate_cache(lcp)
            return
        ks, vs, _ = slot
        self._truncate_cache(0)
        for li in range(self.hp.n_layer):
            self.cache_k[li][:lcp] = ks[li][:lcp]
            self.cache_v[li][:lcp] = vs[li][:lcp]
        self.n_past = lcp
        self.cached_ids.extend(key[:lcp])

    def prefix_cache_stats(self) -> dict:
        """Counters for the multi-slot prefix cache: hits are restores,
        misses are context switches no slot could serve."""
        return {"slots": len(self._prefix_slots or ()),
                "bytes": self._prefix_bytes,
                "hits": self._prefix_hits,
                "misses": self._prefix_misses,
                "saves": self._prefix_saves,
                "evictions": self._prefix_evictions}

    # ---- rotary embeddings ----------------------------------------------

    def _rope_pure(self, vec: list, n_heads: int, pos: int) -> list:
        return self._rope_pure_base(vec, n_heads, pos, self.hp.rope_base,
                                    self.hp.rope_freq_scale)

    def _rope_pure_base(self, vec: list, n_heads: int, pos: int,
                        rope_base: float, freq_scale: float = 1.0) -> list:
        hp = self.hp
        hd, n_rot = hp.head_dim, hp.n_rot
        out = list(vec)
        half = n_rot // 2
        for h in range(n_heads):
            base = h * hd
            for i in range(half):
                theta = pos * freq_scale * rope_base ** (-2.0 * i / n_rot)
                c, s = math.cos(theta), math.sin(theta)
                if hp.rope_style == "norm":
                    a, b = base + 2 * i, base + 2 * i + 1
                else:  # neox
                    a, b = base + i, base + half + i
                x0, x1 = out[a], out[b]
                out[a] = x0 * c - x1 * s
                out[b] = x0 * s + x1 * c
        return out

    def _rope_np(self, vec, n_heads: int, pos: int):
        hp = self.hp
        hd, n_rot = hp.head_dim, hp.n_rot
        half = n_rot // 2
        if self._use_kernel_attention and n_rot % 2 == 0 and n_rot <= hd:
            # bit-identical JIT rotation (strict FP, no reductions); the
            # NumPy slicing below allocates four temporaries per call and
            # costs ~1.4 ms/token across 64 calls. Degenerate rope
            # dimensions (odd, or wider than the head - unvalidated GGUF
            # metadata) stay on the NumPy path, which fails loudly where
            # the kernel would return uninitialized memory.
            from . import kernels
            return kernels.rope_decode(
                np.ascontiguousarray(vec, dtype=np.float32),
                self._rope_cos[pos], self._rope_sin[pos],
                n_heads, hd, n_rot, hp.rope_style)
        v = vec.reshape(n_heads, hd).copy()
        c, s = self._rope_cos[pos], self._rope_sin[pos]
        if hp.rope_style == "norm":
            x0 = v[:, 0:n_rot:2].copy()
            x1 = v[:, 1:n_rot:2].copy()
            v[:, 0:n_rot:2] = x0 * c - x1 * s
            v[:, 1:n_rot:2] = x0 * s + x1 * c
        else:
            x0 = v[:, :half].copy()
            x1 = v[:, half:n_rot].copy()
            v[:, :half] = x0 * c - x1 * s
            v[:, half:n_rot] = x0 * s + x1 * c
        return v.reshape(-1)

    def _rope_batch_np(self, vecs, n_heads: int, positions):
        hp = self.hp
        hd, n_rot = hp.head_dim, hp.n_rot
        half = n_rot // 2
        v = vecs.reshape(len(vecs), n_heads, hd).copy()
        c = self._rope_cos[positions][:, None, :]
        s = self._rope_sin[positions][:, None, :]
        if hp.rope_style == "norm":
            x0 = v[:, :, 0:n_rot:2].copy()
            x1 = v[:, :, 1:n_rot:2].copy()
            v[:, :, 0:n_rot:2] = x0 * c - x1 * s
            v[:, :, 1:n_rot:2] = x0 * s + x1 * c
        else:
            x0 = v[:, :, :half].copy()
            x1 = v[:, :, half:n_rot].copy()
            v[:, :, :half] = x0 * c - x1 * s
            v[:, :, half:n_rot] = x0 * s + x1 * c
        return v.reshape(len(vecs), -1)

    # ---- forward pass ----------------------------------------------------

    def forward(self, token: int) -> "object":
        """Process one token at the current position; returns logits."""
        if self.n_past >= self.n_ctx:
            raise RuntimeError(f"context window full ({self.n_ctx} tokens)")
        p = _prof.ACTIVE
        t0 = _perf() if p is not None else 0.0
        if self.hp.arch == "gemma3":
            logits = (self._forward_gemma3_np(token) if T.HAS_NUMPY
                      else self._forward_gemma3_pure(token))
        else:
            logits = self._forward_np(token) if T.HAS_NUMPY else self._forward_pure(token)
        if p is not None and not p.in_prefill:
            # one decode token, wall-clock: the sum of the buckets inside is
            # always less, and the difference is what `orchestration` reports
            p.token(_perf() - t0)
        self.cached_ids.append(token)
        return logits

    def _attention_np(self, q, K, V, group: int, inv_sqrt: float):
        hp = self.hp
        # the fused kernel runs on the matvec kernels' own thread pool; the
        # np.matmul path below enters OpenBLAS, whose separate pool fans out
        # per call once the context passes its size threshold (~600 tokens)
        # and thrashes against the kernel threads - measured 124 -> 800+
        # ms/token. The gate is "this model runs quantized kernels", the
        # same condition that warmed the JIT at load: a fully DENSE model
        # decodes single-pool on OpenBLAS, and sending only its attention
        # to numba would create the two-pool thrash here instead of fixing
        # it (measured 75 -> 258 ms/token), plus a mid-token JIT compile
        # that load never warmed.
        if self._use_kernel_attention:
            from . import kernels
            return kernels.attention_decode(q, K, V, group, inv_sqrt)
        qg = q.reshape(hp.n_kv, group, hp.head_dim)
        scores = np.matmul(qg, K.transpose(1, 2, 0)) * inv_sqrt
        scores -= scores.max(axis=2, keepdims=True)
        w = np.exp(scores)
        w /= w.sum(axis=2, keepdims=True)
        att_out = np.matmul(w, V.transpose(1, 0, 2))
        return att_out.reshape(hp.n_head, hp.head_dim)

    def _attention_batch_np(self, q, K, V, positions, group: int, inv_sqrt: float):
        hp = self.hp
        if self._use_gpu_batch_attention:
            # gpu batch attention (alpaccaroo/cuda.py): same math with an
            # online softmax. None (unsupported geometry, tier parked)
            # falls through - the einsum below stays the reference.
            from . import cuda as _gpu
            out = _gpu.attention_batch(q, K, V, positions, group, inv_sqrt)
            if out is not None:
                return out
        qg = q.reshape(len(q), hp.n_kv, group, hp.head_dim)
        scores = np.einsum("tkgh,skh->tkgs", qg, K, optimize=True) * inv_sqrt
        allowed = np.arange(K.shape[0], dtype=np.int32)[None, :] <= positions[:, None]
        scores = np.where(allowed[:, None, None, :], scores, -1.0e30)
        scores -= scores.max(axis=-1, keepdims=True)
        w = np.exp(scores)
        w /= w.sum(axis=-1, keepdims=True)
        out = np.einsum("tkgs,skh->tkgh", w, V, optimize=True)
        return out.reshape(len(q), hp.n_head * hp.head_dim)

    def forward_batch(self, tokens: list[int], want_logits: bool = True):
        """Process a NumPy batch at the current position; returns last-token
        logits, or None when `want_logits` is False - prefill discards every
        chunk's logits but the last, and on a tied 262144-row head that
        projection is a large share of the chunk's work."""
        if not T.HAS_NUMPY:
            raise RuntimeError("forward_batch requires the NumPy backend")
        if self.hp.arch == "gemma3":
            return self._forward_batch_gemma3_np(tokens, want_logits)
        if not tokens:
            return None
        if self.n_past + len(tokens) > self.n_ctx:
            raise RuntimeError(f"context window full ({self.n_ctx} tokens)")

        if self._use_gpu_chain and len(tokens) > 1:
            # device-resident prefill chunk (alpaccaroo/cuda.py): the whole
            # chunk on the GPU with one embedding upload, the K/V rows
            # written straight into the decode mirror and copied back to
            # the host cache. None means unavailable or just failed -
            # the body below recomputes the chunk correctly, and a
            # failed prefill path never activates again; a 1-tuple is
            # success even when no logits were wanted.
            from . import cuda as _gpu
            res = _gpu.chain_prefill(self, tokens, want_logits)
            if res is not None:
                self.n_past += len(tokens)
                self.cached_ids.extend(tokens)
                return res[0]

        hp = self.hp
        pos0 = self.n_past
        self._chain_invalidate(pos0)  # this batch rewrites rows from pos0
        positions = np.arange(pos0, pos0 + len(tokens), dtype=np.int32)
        # matrix_rows returns a fresh float32 array for dense and quantized
        x = T.matrix_rows(self.tok_embd, tokens)
        if hp.embed_scale != 1.0:
            x = x * hp.embed_scale
        inv_sqrt = 1.0 / math.sqrt(hp.head_dim)
        group = hp.n_head // hp.n_kv

        qd = hp.n_head * hp.head_dim
        for li, ly in enumerate(self.layers):
            h = T.rmsnorm(x, ly.attn_norm, hp.rms_eps)
            if ly.wqk is not None:
                qk = T.matmul_t(h, ly.wqk)
                q = qk[:, :qd]
                k = qk[:, qd:]
            else:
                q = T.matmul_t(h, ly.wq)
                k = T.matmul_t(h, ly.wk)
            v = T.matmul_t(h, ly.wv)
            if ly.bq is not None:
                q = q + ly.bq
            if ly.bk is not None:
                k = k + ly.bk
            if ly.bv is not None:
                v = v + ly.bv
            q = self._rope_batch_np(q, hp.n_head, positions).reshape(
                len(tokens), hp.n_head, hp.head_dim)
            k = self._rope_batch_np(k, hp.n_kv, positions).reshape(
                len(tokens), hp.n_kv, hp.head_dim)
            self.cache_k[li][pos0:pos0 + len(tokens)] = k
            self.cache_v[li][pos0:pos0 + len(tokens)] = v.reshape(
                len(tokens), hp.n_kv, hp.head_dim)

            K = self.cache_k[li][:pos0 + len(tokens)]
            V = self.cache_v[li][:pos0 + len(tokens)]
            att_out = self._attention_batch_np(q, K, V, positions, group, inv_sqrt)
            x = x + T.matmul_t(att_out, ly.wo)

            h = T.rmsnorm(x, ly.ffn_norm, hp.rms_eps)
            ffn = None
            if (ly.wgu is not None and self._use_gpu_batch_attention
                    and hp.arch not in _GEMMA_ARCHES):
                # fused device FFN (alpaccaroo/cuda.py): both GEMMs plus the
                # silu*up between them in VRAM, so the (batch, 2*n_ff)
                # gate|up block never travels. None (mixed placement,
                # batch 1, degraded) falls through to the exact host path.
                from . import cuda as _gpu
                ffn = _gpu.ffn_swiglu_batch(ly.wgu, ly.w_down, h)
            if ffn is not None:
                x = x + ffn
            else:
                if ly.wgu is not None:
                    gu = T.matmul_t(h, ly.wgu)
                    gate = gu[:, :hp.n_ff]
                    up = gu[:, hp.n_ff:]
                else:
                    gate = T.matmul_t(h, ly.w_gate)
                    up = T.matmul_t(h, ly.w_up)
                act = (T.gelu_pytorch_tanh(gate) if hp.arch in _GEMMA_ARCHES
                       else gate / (1.0 + np.exp(-gate))) * up
                x = x + T.matmul_t(act, ly.w_down)

        self.n_past += len(tokens)
        self.cached_ids.extend(tokens)
        if not want_logits:
            return None
        return T.matvec(self.output, T.rmsnorm(x[-1], self.out_norm, hp.rms_eps))

    def _forward_np(self, token: int):
        if self._use_gpu_chain:
            # device-resident decode chain (alpaccaroo/cuda.py): the whole
            # token on the GPU, one sync. None means unavailable or just
            # failed - either way the body below recomputes the token
            # correctly, and a failed chain never activates again.
            from . import cuda as _gpu
            logits = _gpu.chain_forward(self, token)
            if logits is not None:
                self.n_past += 1
                return logits
        # One module read per token when profiling is off, and `add` stays
        # None so every timing site below is a single truth test. The
        # arithmetic is untouched either way: profiled and un-profiled runs
        # emit identical logits, which tests/smoke.py pins down.
        p = _prof.ACTIVE
        add = p.add if p is not None else None
        hp = self.hp
        pos = self.n_past
        self._chain_invalidate(pos)  # the cache writes below are host-side
        if add:
            t0 = _perf()
        x = T.matrix_row(self.tok_embd, token)  # fresh float32 copy
        if hp.embed_scale != 1.0:
            x = x * hp.embed_scale
        if add:
            add("embedding", _perf() - t0)
        inv_sqrt = 1.0 / math.sqrt(hp.head_dim)
        group = hp.n_head // hp.n_kv

        qd = hp.n_head * hp.head_dim
        for li, ly in enumerate(self.layers):
            if add:
                t0 = _perf()
            h = T.rmsnorm(x, ly.attn_norm, hp.rms_eps)
            if add:
                t1 = _perf()
                add("norm", t1 - t0)
                t0 = t1
            if ly.wqk is not None:
                qk, v = T.matvec_group([ly.wqk, ly.wv], h)
                q = qk[:qd]
                k = qk[qd:]
            else:
                q, k, v = T.matvec_group([ly.wq, ly.wk, ly.wv], h)
            if ly.bq is not None:
                q = q + ly.bq
            if ly.bk is not None:
                k = k + ly.bk
            if ly.bv is not None:
                v = v + ly.bv
            if add:
                t1 = _perf()
                add("attn_qkv", t1 - t0)
                t0 = t1
            q = self._rope_np(q, hp.n_head, pos).reshape(hp.n_head, hp.head_dim)
            k = self._rope_np(k, hp.n_kv, pos).reshape(hp.n_kv, hp.head_dim)
            self.cache_k[li][pos] = k
            self.cache_v[li][pos] = v.reshape(hp.n_kv, hp.head_dim)
            if add:
                t1 = _perf()
                add("rope", t1 - t0)
                t0 = t1

            K = self.cache_k[li][:pos + 1]            # (t, n_kv, hd)
            V = self.cache_v[li][:pos + 1]
            att_out = self._attention_np(q, K, V, group, inv_sqrt)
            if add:
                t1 = _perf()
                add("attention", t1 - t0)
                t0 = t1
            x = x + T.matvec(ly.wo, att_out.reshape(-1))
            if add:
                t1 = _perf()
                add("attn_out", t1 - t0)
                t0 = t1

            h = T.rmsnorm(x, ly.ffn_norm, hp.rms_eps)
            if add:
                t1 = _perf()
                add("norm", t1 - t0)
                t0 = t1
            if ly.wgu is not None:
                gu = T.matvec(ly.wgu, h)
                gate = gu[:hp.n_ff]
                up = gu[hp.n_ff:]
            else:
                gate = T.matvec(ly.w_gate, h)
                up = T.matvec(ly.w_up, h)
            if add:
                t1 = _perf()
                add("ffn_gate_up", t1 - t0)
                t0 = t1
            act = (T.gelu_pytorch_tanh(gate) if hp.arch in _GEMMA_ARCHES
                   else gate / (1.0 + np.exp(-gate))) * up
            if add:
                t1 = _perf()
                add("ffn_act", t1 - t0)
                t0 = t1
            x = x + T.matvec(ly.w_down, act)
            if add:
                add("ffn_down", _perf() - t0)

        self.n_past += 1
        if add:
            t0 = _perf()
        h = T.rmsnorm(x, self.out_norm, hp.rms_eps)
        if add:
            t1 = _perf()
            add("norm", t1 - t0)
            t0 = t1
        logits = T.matvec(self.output, h)
        if add:
            add("output_proj", _perf() - t0)
        return logits

    def _gemma3_layer_is_sliding(self, layer_index: int) -> bool:
        hp = self.hp
        if hp.sliding_window <= 0:
            return False
        if hp.sliding_layers:
            return hp.sliding_layers[layer_index]
        if hp.full_attention_period <= 0:
            # llama.cpp reads a period of 0 as "every layer slides"
            return hp.arch == "gemma3"
        return (layer_index + 1) % hp.full_attention_period != 0

    def _rope_np_gemma3(self, vec, n_heads: int, pos: int, sliding: bool):
        cos = self._rope_cos_swa if sliding else self._rope_cos
        sin = self._rope_sin_swa if sliding else self._rope_sin
        # a sliding layer only exists when sliding_window > 0, which forces
        # rope_base_swa > 0, which builds this table - so falling back to the
        # global table here would silently rotate sliding layers with the 1e6
        # base and there would be no way to notice
        assert cos is not None and sin is not None, (
            "sliding-window RoPE table missing for a sliding layer")
        hp = self.hp
        hd, n_rot = hp.head_dim, hp.n_rot
        half = n_rot // 2
        if self._use_kernel_attention and n_rot % 2 == 0 and n_rot <= hd:
            # same degenerate-n_rot guard as _rope_np
            from . import kernels
            return kernels.rope_decode(
                np.ascontiguousarray(vec, dtype=np.float32),
                cos[pos], sin[pos], n_heads, hd, n_rot, "neox")
        v = vec.reshape(n_heads, hd).copy()
        c, s = cos[pos], sin[pos]
        x0 = v[:, :half].copy()
        x1 = v[:, half:n_rot].copy()
        v[:, :half] = x0 * c - x1 * s
        v[:, half:n_rot] = x0 * s + x1 * c
        return v.reshape(-1)

    def _rope_batch_np_gemma3(self, vecs, n_heads: int, positions, sliding: bool):
        cos = self._rope_cos_swa if sliding else self._rope_cos
        sin = self._rope_sin_swa if sliding else self._rope_sin
        # a sliding layer only exists when sliding_window > 0, which forces
        # rope_base_swa > 0, which builds this table - so falling back to the
        # global table here would silently rotate sliding layers with the 1e6
        # base and there would be no way to notice
        assert cos is not None and sin is not None, (
            "sliding-window RoPE table missing for a sliding layer")
        hp = self.hp
        hd, n_rot = hp.head_dim, hp.n_rot
        half = n_rot // 2
        v = vecs.reshape(len(vecs), n_heads, hd).copy()
        c = cos[positions][:, None, :]
        s = sin[positions][:, None, :]
        x0 = v[:, :, :half].copy()
        x1 = v[:, :, half:n_rot].copy()
        v[:, :, :half] = x0 * c - x1 * s
        v[:, :, half:n_rot] = x0 * s + x1 * c
        return v.reshape(len(vecs), -1)

    def _rmsnorm_heads_np(self, vec, n_heads: int, weight):
        hp = self.hp
        return T.rmsnorm(vec.reshape(n_heads, hp.head_dim), weight,
                         hp.rms_eps).reshape(-1)

    def _rmsnorm_heads_batch_np(self, vecs, n_heads: int, weight):
        hp = self.hp
        return T.rmsnorm(vecs.reshape(len(vecs), n_heads, hp.head_dim), weight,
                         hp.rms_eps).reshape(len(vecs), -1)

    def _rmsnorm_heads_pure(self, vec: list, n_heads: int, weight) -> list:
        hp = self.hp
        out: list[float] = []
        for h in range(n_heads):
            start = h * hp.head_dim
            out.extend(T.rmsnorm(vec[start:start + hp.head_dim], weight,
                                 hp.rms_eps))
        return out

    def _attention_batch_window_np(self, q, K, V, positions, group: int,
                                   inv_sqrt: float, window: int = 0,
                                   kv_start: int = 0):
        hp = self.hp
        if self._use_gpu_batch_attention:
            # same gpu path as _attention_batch_np; the kernel carries
            # the window mask and kv_start natively. Today's gemma3
            # models decline it anyway (head_dim 256 > the kernel's 128)
            # and keep this exact NumPy path - see cuda._ATT_HD.
            from . import cuda as _gpu
            out = _gpu.attention_batch(q, K, V, positions, group, inv_sqrt,
                                       window, kv_start)
            if out is not None:
                return out
        qg = q.reshape(len(q), hp.n_kv, group, hp.head_dim)
        scores = np.einsum("tkgh,skh->tkgs", qg, K, optimize=True) * inv_sqrt
        kv_pos = np.arange(kv_start, kv_start + K.shape[0], dtype=np.int32)
        allowed = kv_pos[None, :] <= positions[:, None]
        if window > 0:
            allowed &= kv_pos[None, :] > (positions[:, None] - window)
        scores = np.where(allowed[:, None, None, :], scores, -1.0e30)
        scores -= scores.max(axis=-1, keepdims=True)
        w = np.exp(scores)
        w /= w.sum(axis=-1, keepdims=True)
        out = np.einsum("tkgs,skh->tkgh", w, V, optimize=True)
        return out.reshape(len(q), hp.n_head * hp.head_dim)

    def _softcap_logits(self, logits):
        cap = self.hp.final_logit_softcap
        if cap <= 0.0:
            return logits
        if T.HAS_NUMPY:
            return np.tanh(logits / cap) * cap
        return [math.tanh(v / cap) * cap for v in logits]

    def _forward_batch_gemma3_np(self, tokens: list[int], want_logits: bool = True):
        if not tokens:
            return None
        if self.n_past + len(tokens) > self.n_ctx:
            raise RuntimeError(f"context window full ({self.n_ctx} tokens)")

        hp = self.hp
        pos0 = self.n_past
        self._chain_invalidate(pos0)  # same contract as forward_batch
        positions = np.arange(pos0, pos0 + len(tokens), dtype=np.int32)
        x = T.matrix_rows(self.tok_embd, tokens) * hp.embed_scale
        inv_sqrt = hp.attention_scale
        group = hp.n_head // hp.n_kv

        qd = hp.n_head * hp.head_dim
        for li, ly in enumerate(self.layers):
            h = T.rmsnorm(x, ly.attn_norm, hp.rms_eps)
            if ly.wqk is not None:
                qk = T.matmul_t(h, ly.wqk)
                q = qk[:, :qd]
                k = qk[:, qd:]
            else:
                q = T.matmul_t(h, ly.wq)
                k = T.matmul_t(h, ly.wk)
            v = T.matmul_t(h, ly.wv)
            if ly.bq is not None:
                q = q + ly.bq
            if ly.bk is not None:
                k = k + ly.bk
            if ly.bv is not None:
                v = v + ly.bv
            q = self._rmsnorm_heads_batch_np(q, hp.n_head, ly.q_norm)
            k = self._rmsnorm_heads_batch_np(k, hp.n_kv, ly.k_norm)
            sliding = self._gemma3_layer_is_sliding(li)
            q = self._rope_batch_np_gemma3(q, hp.n_head, positions, sliding).reshape(
                len(tokens), hp.n_head, hp.head_dim)
            k = self._rope_batch_np_gemma3(k, hp.n_kv, positions, sliding).reshape(
                len(tokens), hp.n_kv, hp.head_dim)
            self.cache_k[li][pos0:pos0 + len(tokens)] = k
            self.cache_v[li][pos0:pos0 + len(tokens)] = v.reshape(
                len(tokens), hp.n_kv, hp.head_dim)

            window = hp.sliding_window if sliding else 0
            kv_start = max(0, int(positions[0]) - window + 1) if window > 0 else 0
            K = self.cache_k[li][kv_start:pos0 + len(tokens)]
            V = self.cache_v[li][kv_start:pos0 + len(tokens)]
            att_out = self._attention_batch_window_np(q, K, V, positions,
                                                       group, inv_sqrt, window,
                                                       kv_start)
            att_proj = T.matmul_t(att_out, ly.wo)
            x = x + T.rmsnorm(att_proj, ly.post_attn_norm, hp.rms_eps)

            h = T.rmsnorm(x, ly.ffn_norm, hp.rms_eps)
            if ly.wgu is not None:
                gu = T.matmul_t(h, ly.wgu)
                gate = gu[:, :hp.n_ff]
                up = gu[:, hp.n_ff:]
            else:
                gate = T.matmul_t(h, ly.w_gate)
                up = T.matmul_t(h, ly.w_up)
            act = T.gelu_pytorch_tanh(gate) * up
            ffn_out = T.matmul_t(act, ly.w_down)
            x = x + T.rmsnorm(ffn_out, ly.post_ffw_norm, hp.rms_eps)

        self.n_past += len(tokens)
        self.cached_ids.extend(tokens)
        if not want_logits:
            return None
        logits = T.matvec(self.output, T.rmsnorm(x[-1], self.out_norm, hp.rms_eps))
        return self._softcap_logits(logits)

    def _forward_gemma3_np(self, token: int):
        # same bucket contract as _forward_np; see the comment there
        p = _prof.ACTIVE
        add = p.add if p is not None else None
        hp = self.hp
        pos = self.n_past
        self._chain_invalidate(pos)  # same contract as _forward_np
        if add:
            t0 = _perf()
        x = T.matrix_row(self.tok_embd, token) * hp.embed_scale
        if add:
            add("embedding", _perf() - t0)
        inv_sqrt = hp.attention_scale
        group = hp.n_head // hp.n_kv

        qd = hp.n_head * hp.head_dim
        for li, ly in enumerate(self.layers):
            if add:
                t0 = _perf()
            h = T.rmsnorm(x, ly.attn_norm, hp.rms_eps)
            if add:
                t1 = _perf()
                add("norm", t1 - t0)
                t0 = t1
            if ly.wqk is not None:
                qk, v = T.matvec_group([ly.wqk, ly.wv], h)
                q = qk[:qd]
                k = qk[qd:]
            else:
                q, k, v = T.matvec_group([ly.wq, ly.wk, ly.wv], h)
            if ly.bq is not None:
                q = q + ly.bq
            if ly.bk is not None:
                k = k + ly.bk
            if ly.bv is not None:
                v = v + ly.bv
            if add:
                t1 = _perf()
                add("attn_qkv", t1 - t0)
                t0 = t1
            q = self._rmsnorm_heads_np(q, hp.n_head, ly.q_norm)
            k = self._rmsnorm_heads_np(k, hp.n_kv, ly.k_norm)
            if add:
                t1 = _perf()
                add("norm", t1 - t0)   # gemma3's per-head q/k norms
                t0 = t1
            sliding = self._gemma3_layer_is_sliding(li)
            q = self._rope_np_gemma3(q, hp.n_head, pos, sliding).reshape(
                hp.n_head, hp.head_dim)
            k = self._rope_np_gemma3(k, hp.n_kv, pos, sliding).reshape(
                hp.n_kv, hp.head_dim)
            self.cache_k[li][pos] = k
            self.cache_v[li][pos] = v.reshape(hp.n_kv, hp.head_dim)
            if add:
                t1 = _perf()
                add("rope", t1 - t0)
                t0 = t1

            start = max(0, pos - hp.sliding_window + 1) if sliding else 0
            K = self.cache_k[li][start:pos + 1]
            V = self.cache_v[li][start:pos + 1]
            att_out = self._attention_np(q, K, V, group, inv_sqrt)
            if add:
                t1 = _perf()
                add("attention", t1 - t0)
                t0 = t1
            att_proj = T.matvec(ly.wo, att_out.reshape(-1))
            if add:
                t1 = _perf()
                add("attn_out", t1 - t0)
                t0 = t1
            x = x + T.rmsnorm(att_proj, ly.post_attn_norm, hp.rms_eps)

            h = T.rmsnorm(x, ly.ffn_norm, hp.rms_eps)
            if add:
                t1 = _perf()
                add("norm", t1 - t0)
                t0 = t1
            if ly.wgu is not None:
                gu = T.matvec(ly.wgu, h)
                if add:
                    t1 = _perf()
                    add("ffn_gate_up", t1 - t0)
                    t0 = t1
                act = T.gelu_pytorch_tanh(gu[:hp.n_ff]) * gu[hp.n_ff:]
            else:
                gate = T.matvec(ly.w_gate, h)
                up = T.matvec(ly.w_up, h)
                if add:
                    t1 = _perf()
                    add("ffn_gate_up", t1 - t0)
                    t0 = t1
                act = T.gelu_pytorch_tanh(gate) * up
            if add:
                t1 = _perf()
                add("ffn_act", t1 - t0)
                t0 = t1
            ffn_out = T.matvec(ly.w_down, act)
            if add:
                t1 = _perf()
                add("ffn_down", t1 - t0)
                t0 = t1
            x = x + T.rmsnorm(ffn_out, ly.post_ffw_norm, hp.rms_eps)
            if add:
                add("norm", _perf() - t0)

        self.n_past += 1
        if add:
            t0 = _perf()
        h = T.rmsnorm(x, self.out_norm, hp.rms_eps)
        if add:
            t1 = _perf()
            add("norm", t1 - t0)
            t0 = t1
        logits = self._softcap_logits(T.matvec(self.output, h))
        if add:
            add("output_proj", _perf() - t0)
        return logits

    def _forward_gemma3_pure(self, token: int):
        # same bucket and in_prefill contract as _forward_pure
        p = _prof.ACTIVE
        add = p.add if p is not None and not p.in_prefill else None
        hp = self.hp
        pos = self.n_past
        if add:
            t0 = _perf()
        x = T.scale(T.matrix_row(self.tok_embd, token), hp.embed_scale)
        if add:
            add("embedding", _perf() - t0)
        inv_sqrt = hp.attention_scale
        group = hp.n_head // hp.n_kv
        hd = hp.head_dim

        for li, ly in enumerate(self.layers):
            if add:
                t0 = _perf()
            h = T.rmsnorm(x, ly.attn_norm, hp.rms_eps)
            if add:
                t1 = _perf()
                add("norm", t1 - t0)
                t0 = t1
            q = T.matvec(ly.wq, h)
            k = T.matvec(ly.wk, h)
            v = T.matvec(ly.wv, h)
            if ly.bq is not None:
                q = T.add(q, ly.bq)
            if ly.bk is not None:
                k = T.add(k, ly.bk)
            if ly.bv is not None:
                v = T.add(v, ly.bv)
            if add:
                t1 = _perf()
                add("attn_qkv", t1 - t0)
                t0 = t1
            q = self._rmsnorm_heads_pure(q, hp.n_head, ly.q_norm)
            k = self._rmsnorm_heads_pure(k, hp.n_kv, ly.k_norm)
            if add:
                t1 = _perf()
                add("norm", t1 - t0)   # gemma3's per-head q/k norms
                t0 = t1
            sliding = self._gemma3_layer_is_sliding(li)
            base = hp.rope_base_swa if sliding else hp.rope_base
            freq_scale = 1.0 if sliding else hp.rope_freq_scale
            q = self._rope_pure_base(q, hp.n_head, pos, base, freq_scale)
            k = self._rope_pure_base(k, hp.n_kv, pos, base, freq_scale)
            self.cache_k[li].append(k)
            self.cache_v[li].append(v)
            if add:
                t1 = _perf()
                add("rope", t1 - t0)
                t0 = t1

            start = max(0, pos - hp.sliding_window + 1) if sliding else 0
            att_out = [0.0] * (hp.n_head * hd)
            for hh in range(hp.n_head):
                kvh = hh // group
                qh = q[hh * hd:(hh + 1) * hd]
                scores = []
                for t in range(start, pos + 1):
                    kt = self.cache_k[li][t][kvh * hd:(kvh + 1) * hd]
                    scores.append(T.dot(qh, kt) * inv_sqrt)
                w = T.softmax(scores)
                acc = [0.0] * hd
                for offs, wt in enumerate(w):
                    if wt == 0.0:
                        continue
                    vt = self.cache_v[li][start + offs][kvh * hd:(kvh + 1) * hd]
                    for d in range(hd):
                        acc[d] += wt * vt[d]
                att_out[hh * hd:(hh + 1) * hd] = acc
            if add:
                t1 = _perf()
                add("attention", t1 - t0)
                t0 = t1
            att_proj = T.matvec(ly.wo, att_out)
            if add:
                t1 = _perf()
                add("attn_out", t1 - t0)
                t0 = t1
            x = T.add(x, T.rmsnorm(att_proj, ly.post_attn_norm, hp.rms_eps))

            h = T.rmsnorm(x, ly.ffn_norm, hp.rms_eps)
            if add:
                t1 = _perf()
                add("norm", t1 - t0)
                t0 = t1
            gate = T.matvec(ly.w_gate, h)
            up = T.matvec(ly.w_up, h)
            if add:
                t1 = _perf()
                add("ffn_gate_up", t1 - t0)
                t0 = t1
            act = T.mul(T.gelu_pytorch_tanh(gate), up)
            if add:
                t1 = _perf()
                add("ffn_act", t1 - t0)
                t0 = t1
            ffn_out = T.matvec(ly.w_down, act)
            if add:
                t1 = _perf()
                add("ffn_down", t1 - t0)
                t0 = t1
            x = T.add(x, T.rmsnorm(ffn_out, ly.post_ffw_norm, hp.rms_eps))
            if add:
                add("norm", _perf() - t0)

        self.n_past += 1
        if add:
            t0 = _perf()
        h = T.rmsnorm(x, self.out_norm, hp.rms_eps)
        if add:
            t1 = _perf()
            add("norm", t1 - t0)
            t0 = t1
        logits = self._softcap_logits(T.matvec(self.output, h))
        if add:
            add("output_proj", _perf() - t0)
        return logits

    def _forward_pure(self, token: int):
        # Same bucket contract as _forward_np; see the comment there. Unlike
        # the NumPy tier, which prefills through forward_batch, this one is
        # also the prefill path (prefill() loops forward() without NumPy), so
        # the buckets stay off while in_prefill or prompt tokens would be
        # charged to the decode breakdown and it would exceed 100%.
        p = _prof.ACTIVE
        add = p.add if p is not None and not p.in_prefill else None
        hp = self.hp
        pos = self.n_past
        if add:
            t0 = _perf()
        x = T.matrix_row(self.tok_embd, token)
        if hp.embed_scale != 1.0:
            x = T.scale(x, hp.embed_scale)
        if add:
            add("embedding", _perf() - t0)
        inv_sqrt = 1.0 / math.sqrt(hp.head_dim)
        group = hp.n_head // hp.n_kv
        hd = hp.head_dim

        for li, ly in enumerate(self.layers):
            if add:
                t0 = _perf()
            h = T.rmsnorm(x, ly.attn_norm, hp.rms_eps)
            if add:
                t1 = _perf()
                add("norm", t1 - t0)
                t0 = t1
            q = T.matvec(ly.wq, h)
            k = T.matvec(ly.wk, h)
            v = T.matvec(ly.wv, h)
            if ly.bq is not None:
                q = T.add(q, ly.bq)
            if ly.bk is not None:
                k = T.add(k, ly.bk)
            if ly.bv is not None:
                v = T.add(v, ly.bv)
            if add:
                t1 = _perf()
                add("attn_qkv", t1 - t0)
                t0 = t1
            q = self._rope_pure(q, hp.n_head, pos)
            k = self._rope_pure(k, hp.n_kv, pos)
            self.cache_k[li].append(k)
            self.cache_v[li].append(v)
            if add:
                t1 = _perf()
                add("rope", t1 - t0)
                t0 = t1

            att_out = [0.0] * (hp.n_head * hd)
            t_len = pos + 1
            for hh in range(hp.n_head):
                kvh = hh // group
                qh = q[hh * hd:(hh + 1) * hd]
                scores = []
                for t in range(t_len):
                    kt = self.cache_k[li][t][kvh * hd:(kvh + 1) * hd]
                    scores.append(T.dot(qh, kt) * inv_sqrt)
                w = T.softmax(scores)
                acc = [0.0] * hd
                for t, wt in enumerate(w):
                    if wt == 0.0:
                        continue
                    vt = self.cache_v[li][t][kvh * hd:(kvh + 1) * hd]
                    for d in range(hd):
                        acc[d] += wt * vt[d]
                att_out[hh * hd:(hh + 1) * hd] = acc
            if add:
                t1 = _perf()
                add("attention", t1 - t0)
                t0 = t1
            x = T.add(x, T.matvec(ly.wo, att_out))
            if add:
                t1 = _perf()
                add("attn_out", t1 - t0)
                t0 = t1

            h = T.rmsnorm(x, ly.ffn_norm, hp.rms_eps)
            if add:
                t1 = _perf()
                add("norm", t1 - t0)
                t0 = t1
            gate = T.matvec(ly.w_gate, h)
            up = T.matvec(ly.w_up, h)
            if add:
                t1 = _perf()
                add("ffn_gate_up", t1 - t0)
                t0 = t1
            act = T.mul(T.gelu_pytorch_tanh(gate) if hp.arch in _GEMMA_ARCHES
                        else T.silu(gate), up)
            if add:
                t1 = _perf()
                add("ffn_act", t1 - t0)
                t0 = t1
            x = T.add(x, T.matvec(ly.w_down, act))
            if add:
                add("ffn_down", _perf() - t0)

        self.n_past += 1
        if add:
            t0 = _perf()
        h = T.rmsnorm(x, self.out_norm, hp.rms_eps)
        if add:
            t1 = _perf()
            add("norm", t1 - t0)
            t0 = t1
        logits = T.matvec(self.output, h)
        if add:
            add("output_proj", _perf() - t0)
        return logits

    # ---- convenience -----------------------------------------------------

    def prefill(self, tokens: list[int], *, check_cancelled=None):
        """Feed prompt tokens; returns logits of the last one."""
        if check_cancelled is not None:
            check_cancelled()
        self.last_prefill_forwarded = 0
        if not tokens:
            return None
        if len(tokens) > self.n_ctx:
            raise RuntimeError(f"context window full ({self.n_ctx} tokens)")

        n = 0
        max_prefix = min(len(tokens), len(self.cached_ids))
        while n < max_prefix and tokens[n] == self.cached_ids[n]:
            n += 1
        # multi-slot prefix cache: a saved snapshot sharing a longer prefix
        # than the live cache restores here (snapshotting the live context
        # first when it is worth keeping), and `n` grows to match; with the
        # feature disabled this returns `n` untouched
        n = self._prefix_cache_route(tokens, n)
        if n == len(tokens):
            n = max(0, len(tokens) - 1)
        if n != self.n_past:
            self._truncate_cache(n)

        suffix = tokens[n:]
        self.last_prefill_forwarded = len(suffix)
        logits = None
        p = _prof.ACTIVE
        if p is not None:
            p.set("prefill_reused_prefix_tokens", n)
        t0 = _perf() if p is not None else 0.0
        if p is not None:
            p.in_prefill = True
        try:
            if T.HAS_NUMPY:
                raw = os.environ.get("ALPACCAROO_PREFILL_CHUNK", "256")
                try:
                    chunk = max(1, int(raw))
                except ValueError:
                    chunk = 256
                for i in range(0, len(suffix), chunk):
                    if check_cancelled is not None:
                        check_cancelled()
                    logits = self.forward_batch(suffix[i:i + chunk],
                                                want_logits=i + chunk >= len(suffix))
            else:
                for t in suffix:
                    if check_cancelled is not None:
                        check_cancelled()
                    logits = self.forward(t)
        finally:
            if p is not None:
                p.in_prefill = False
        if p is not None:
            # the forwarded suffix only: a prefix hit costs no compute, and
            # charging it here would report a tok/s the machine never did
            p.prefill(len(suffix), _perf() - t0)
        return logits

    def describe(self) -> str:
        hp = self.hp
        params = hp.n_vocab * hp.n_embd
        if self.output is not self.tok_embd:
            params += hp.n_vocab * hp.n_embd  # untied output projection
        params += hp.n_embd  # output_norm
        for ly in range(hp.n_layer):
            params += 2 * hp.n_embd  # norms
            if hp.arch == "gemma3":
                params += 2 * hp.head_dim  # q_norm/k_norm are shared per head
                params += 2 * hp.n_embd  # post-attention/post-ffw norms
            params += hp.n_embd * hp.n_head * hp.head_dim * 2  # wq, wo
            params += hp.n_embd * hp.n_kv * hp.head_dim * 2    # wk, wv
            params += 3 * hp.n_embd * hp.n_ff
        storage = self._storage_description()
        # the default clamps a 32768-token Gemma 3 down to 4096; say so rather
        # than letting the model look like it has a quarter of its real window
        ctx = (f"ctx {self.n_ctx}" if self.n_ctx >= hp.n_ctx_train
               else f"ctx {self.n_ctx} of {hp.n_ctx_train}")
        attn = ""
        if hp.arch == "gemma3" and hp.sliding_window > 0:
            layout = (f"pattern {sum(hp.sliding_layers)}/{hp.n_layer}"
                      if hp.sliding_layers else f"every {hp.full_attention_period}")
            attn = (f" | swa {hp.sliding_window} ({layout}, {hp.swa_rule}) | "
                    f"attn scale {hp.attention_scale:.4g}"
                    f"{' (metadata)' if hp.attention_scale_from_metadata else ''}")
        # only when snapshots exist: an idle store is invisible
        prefix = ""
        if self._prefix_slots:
            prefix = (f" | prefix cache {len(self._prefix_slots)} slots "
                      f"({self._size(self._prefix_bytes)})")
        return (f"{hp.arch} | {hp.n_layer} layers | embd {hp.n_embd} | "
                f"heads {hp.n_head}/{hp.n_kv} | ff {hp.n_ff} | vocab {hp.n_vocab} | "
                f"~{params / 1e6:.0f}M params | {ctx}{attn} | "
                f"backend {T.backend_name()} | {storage}{prefix}")

    @staticmethod
    def _size(nbytes: float) -> str:
        mb = nbytes / (1024 * 1024)
        return f"{mb / 1024:.1f} GiB" if mb >= 1024 else f"{mb:.0f} MiB"

    def _storage_description(self) -> str:
        q = self.weight_storage.get("quantized", {})
        dense = int(self.weight_storage.get("dense", 0) or 0)
        fallback = self.weight_storage.get("fallback", {})
        densified = self.weight_storage.get("densified") or []
        q_bytes = self.weight_storage.get("quantized_bytes", 0)
        d_bytes = self.weight_storage.get("dense_bytes", 0)
        # matrix counts alone say nothing about the RAM this actually costs
        if q:
            q_desc = "/".join(q.keys())
            total_q = sum(q.values())
            parts = [f"weights quantized {q_desc} "
                     f"({total_q} matrices, {self._size(q_bytes)})"]
            if dense:
                parts.append(f"dense {dense} ({self._size(d_bytes)})")
        else:
            parts = [f"weights dense ({dense} matrices, {self._size(d_bytes)})"]
        if fallback:
            fb_desc = "/".join(fallback.keys())
            parts.append(f"dense fallback {fb_desc}")
        if densified:
            size = self._size(self.weight_storage.get("densified_bytes", 0))
            parts.append(f"dense budget {len(densified)} matrices ({size})")
        gpu_n = int(self.weight_storage.get("gpu", 0) or 0)
        if gpu_n:
            g_bytes = self.weight_storage.get("gpu_bytes", 0)
            parts.append(f"gpu {gpu_n} matrices "
                         f"({self._size(g_bytes)} VRAM)")
        return ", ".join(parts)
