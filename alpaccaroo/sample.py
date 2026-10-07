# Alpaccaroo - token sampling: greedy, temperature, top-k, top-p, repeat
# penalty. Deterministic for a given seed. MIT License. See LICENSE.
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from time import perf_counter as _perf

from . import profiling as _prof
from . import tensor as T

if T.HAS_NUMPY:
    import numpy as _np


@dataclass
class SamplerParams:
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    repeat_penalty: float = 1.1
    repeat_last_n: int = 64
    seed: int = -1  # -1 -> random


def _topk_order(arr, k: int):
    """Indices of the k largest entries of a float64 array, descending,
    ties broken by lower index. Returns None when the array contains NaN;
    the caller falls back to the stable Python sort whose NaN behaviour
    the tests pin down. Everything here is O(n) or O(k log k) NumPy - the
    old list-based path converted 128k-262k logits to Python floats every
    token, which cost more than the whole top-k selection."""
    n = arr.size
    if _np.isnan(arr).any():
        # NaN sorts as the largest value for argpartition but compares
        # False against everything, so the masks below would select
        # nothing and hand the caller an empty list. Infinities are fine.
        return None
    if k >= n:
        # full descending order; the secondary arange key reproduces the
        # stable sort's ascending-index tie-break
        return _np.lexsort((_np.arange(n), -arr))
    # argpartition alone is not enough: when logits tie across the cut it
    # keeps an arbitrary one, where the stable sort keeps the lowest index.
    # So take everything strictly above the k-th value, then fill from the
    # tied indices in ascending order.
    thr = arr[_np.argpartition(arr, n - k)[n - k:]].min()
    above = _np.flatnonzero(arr > thr)
    ties = _np.flatnonzero(arr == thr)[:k - above.size]
    sel = _np.concatenate((above, ties))
    order = _np.lexsort((sel, -arr[sel]))  # last key is primary
    return sel[order]


def _topk_indices(logits: list, k: int) -> list[int]:
    """Indices of the k largest logits, descending, ties broken by lower index.

    Identical output to ``sorted(range(n), key=logits.__getitem__,
    reverse=True)[:k]`` - which is a stable sort, so equal logits keep
    ascending index order - but O(n) instead of O(n log n). Gemma 3's 262144
    -entry vocabulary makes the full sort a measurable share of every token.
    """
    n = len(logits)
    if T.HAS_NUMPY:
        order = _topk_order(_np.asarray(logits, dtype=_np.float64), min(k, n))
        if order is None:
            return sorted(range(n), key=logits.__getitem__, reverse=True)[:k]
        return [int(i) for i in order]
    if k >= n:
        return sorted(range(n), key=logits.__getitem__, reverse=True)
    import heapq

    top = heapq.nlargest(k, range(n), key=lambda i: (logits[i], -i))
    return top


@dataclass
class Sampler:
    params: SamplerParams = field(default_factory=SamplerParams)

    def __post_init__(self):
        seed = self.params.seed
        self.rng = random.Random(seed if seed >= 0 else None)
        self.recent: list[int] = []

    def accept(self, token: int) -> None:
        self.recent.append(token)
        if len(self.recent) > max(self.params.repeat_last_n, 1):
            self.recent.pop(0)

    def sample(self, logits, banned=None) -> int:
        """Pick a token id. `banned` is an optional set of token ids to
        exclude (constrained decoding - see jsonform): they are masked to
        -inf on a private copy, so the caller's logits are never touched
        and repeated resampling of one position stays independent."""
        prof = _prof.ACTIVE
        if prof is not None:
            t0 = _perf()
            try:
                return self._sample(logits, banned)
            finally:
                prof.add("sampler", _perf() - t0)
        return self._sample(logits, banned)

    def _greedy_fast(self, arr):
        """argmax of the repeat-penalized logits without ever building the
        penalized vector, or None to fall through to the general path.

        PERFORMANCE_PLAN Track 1. The general path converts the whole
        logits vector to float64 (1.2 MB on a 151936-token vocabulary),
        scans it for NaN, penalizes it and scans it again. Greedy decoding
        needs one index out of all that, and the repeat penalty can only
        touch `repeat_last_n` entries - 64 by default.

        Exact, not approximate:

        * float32 -> float64 is exact and order-preserving, so an argmax
          taken over the float32 values is the argmax of the float64 copy.
        * the <= 64 penalized entries are still evaluated in float64,
          because ``v / 1.1`` generally is not a float32.
        * the winner is the better of "best penalized entry" and "best
          untouched entry", ties broken by lower index - which is what
          np.argmax does on the materialized vector.
        * the untouched maximum is found among the top ``k+1`` values (at
          most k entries can outrank it), then resolved to the FIRST index
          holding that value, so a tie with a lower index outside the
          partition cannot be missed.

        Returns None whenever the general path's finiteness gate would have
        sent this vector to the list implementation, so the NaN behaviour
        the tests pin down is unchanged.
        """
        p = self.params
        if not arr.size:
            return None
        # ONE pass over the vocabulary, and it does double duty. np.argmax
        # returns the first index of the maximum; NaN compares greater than
        # everything for it, so a NaN anywhere and an infinite maximum both
        # surface as a non-finite value here - exactly the two conditions
        # the general path's gate tests. An earlier version reached for
        # argpartition to find the runner-up and measured 2x SLOWER than the
        # path it was replacing: a quickselect over 151936 entries costs
        # more than the float64 copy it saves.
        top = int(_np.argmax(arr))
        top_v = float(arr[top])
        if not math.isfinite(top_v):
            return None
        if not p.repeat_penalty or p.repeat_penalty == 1.0 or not self.recent:
            return top

        touched = set(self.recent)
        idx = _np.fromiter(sorted(touched), dtype=_np.int64)
        vals = arr[idx].astype(_np.float64)
        pen = _np.where(vals > 0.0, vals / p.repeat_penalty,
                        vals * p.repeat_penalty)
        if not bool(_np.isfinite(pen).all()):
            # a degenerate penalty minted a NaN or an infinity the raw
            # logits never had; the general path's gate would have caught it
            return None
        j = int(_np.argmax(pen))          # first max, so lower index wins
        best_pen_i, best_pen_v = int(idx[j]), float(pen[j])

        if top in touched:
            # The best untouched entry is the runner-up behind up to 64
            # penalized ones, and finding it costs another selection pass.
            # Hand this rare shape to the general path instead of paying for
            # the machinery on every token - the fallback IS the reference.
            return None
        # `top` is the first index of the global maximum and it was not
        # penalized, so it is also the first index of the untouched maximum
        if best_pen_v > top_v:
            return best_pen_i
        if best_pen_v < top_v:
            return top
        return min(best_pen_i, top)

    def _sample(self, logits, banned=None) -> int:
        p = self.params
        if (T.HAS_NUMPY and not banned and p.temperature <= 0
                and not math.isnan(p.temperature)
                and isinstance(logits, _np.ndarray)
                and logits.dtype == _np.float32):
            # float32 only: converting a Python-float list to float32 would
            # ROUND it, and the rounded argmax is not always the float64 one
            hit = self._greedy_fast(logits)
            if hit is not None:
                return hit
        if T.HAS_NUMPY:
            # np.array always copies, so the penalty below cannot mutate the
            # caller's logits; float64 matches the Python-float arithmetic of
            # the list path bit for bit
            arr = _np.array(logits, dtype=_np.float64)
            if banned:
                # exp(-inf)=0 keeps the fast path safe (comment below); a
                # mask covering every finite logit turns arr.max() into -inf
                # and falls through the finiteness gate to the list path,
                # which tolerates the all--inf vector without raising
                arr[list(banned)] = -_np.inf
            if p.repeat_penalty and p.repeat_penalty != 1.0 and self.recent:
                for t in set(self.recent):
                    v = arr[t]
                    arr[t] = (v / p.repeat_penalty if v > 0
                              else v * p.repeat_penalty)
            # The finiteness gate must run AFTER the penalty: a degenerate
            # repeat_penalty (NaN, inf against a 0.0 logit, or a subnormal
            # whose division overflows) mints non-finite values the raw
            # logits never had. A NaN anywhere, or a +/-inf *maximum*,
            # drives the softmax through inf-inf=NaN and the two paths'
            # NaN comparisons differ; np.max propagates NaN, so one test
            # covers both. A NaN temperature poisons the softmax the same
            # way. All such vectors keep the historical list behaviour;
            # -inf among finite logits stays on the fast path: exp(-inf)=0.
            if (arr.size and bool(_np.isfinite(arr.max()))
                    and not math.isnan(p.temperature)):
                return self._sample_array(arr)
        return self._sample_list(T.to_list(logits), banned)

    def _sample_array(self, arr) -> int:
        """NumPy sampling path over already-penalized float64 logits. Same
        arithmetic as _sample_list in the same order (cumsum accumulates
        sequentially, like the running Python sums), so a given seed picks
        the same token; measured 0.5 ms vs 6.6 ms per token on a 128256
        vocabulary, and a request asking for top_k=0 costs a lexsort
        instead of a full-vocab Python sort."""
        p = self.params

        if p.temperature <= 0:
            return int(_np.argmax(arr))

        k = p.top_k if p.top_k and p.top_k > 0 else arr.size
        idx = _topk_order(arr, min(k, arr.size))

        maxl = arr[idx[0]]
        weights = _np.exp((arr[idx] - maxl) / p.temperature)
        cum = _np.cumsum(weights)
        probs = weights / cum[-1]

        if 0.0 < p.top_p < 1.0:
            cp = _np.cumsum(probs)
            # first index whose running sum reaches top_p, inclusive - and
            # like the list path, renormalize even when nothing was cut
            cut = min(int(_np.searchsorted(cp, p.top_p, side="left")) + 1,
                      idx.size)
            idx = idx[:cut]
            probs = probs[:cut] / cp[cut - 1]

        r = self.rng.random()
        j = int(_np.searchsorted(_np.cumsum(probs), r, side="left"))
        if j >= idx.size:
            j = idx.size - 1  # r beyond the last running sum: keep last
        return int(idx[j])

    def _sample_list(self, logits: list, banned=None) -> int:
        """Reference implementation over Python lists: the pure-stdlib path,
        and the fallback for NaN logits (whose comparison quirks the tests
        pin down)."""
        p = self.params

        if banned:
            # same masking as the array path; T.to_list always copies, so
            # this never reaches the caller's logits either
            ninf = float("-inf")
            for t in banned:
                logits[t] = ninf

        if p.repeat_penalty and p.repeat_penalty != 1.0 and self.recent:
            for t in set(self.recent):
                v = logits[t]
                logits[t] = v / p.repeat_penalty if v > 0 else v * p.repeat_penalty

        if p.temperature <= 0:
            return max(range(len(logits)), key=logits.__getitem__)

        # work on the top-k slice only (huge speedup for big vocabs)
        k = p.top_k if p.top_k and p.top_k > 0 else len(logits)
        k = min(k, len(logits))
        idx = _topk_indices(logits, k)

        maxl = logits[idx[0]]
        weights = [math.exp((logits[i] - maxl) / p.temperature) for i in idx]
        total = sum(weights)
        probs = [w / total for w in weights]

        if 0.0 < p.top_p < 1.0:
            acc = 0.0
            cut = len(probs)
            for n, pr in enumerate(probs):
                acc += pr
                if acc >= p.top_p:
                    cut = n + 1
                    break
            idx, probs = idx[:cut], probs[:cut]
            total = sum(probs)
            probs = [pr / total for pr in probs]

        r = self.rng.random()
        acc = 0.0
        for i, pr in zip(idx, probs):
            acc += pr
            if r <= acc:
                return i
        return idx[-1]
