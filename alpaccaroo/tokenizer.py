# Alpaccaroo - tokenizers implemented from scratch: SentencePiece-style
# (greedy highest-score-first bigram merge, byte fallback) and byte-level BPE
# with a GPT-2/llama-3 style pre-tokenizer built on unicodedata (no regex deps).
# MIT License. See LICENSE.
from __future__ import annotations

import heapq
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

SPM_SPACE = "\u2581"  # \u2581

# token types stored in tokenizer.ggml.token_type
TT_NORMAL, TT_UNKNOWN, TT_CONTROL, TT_USER_DEFINED, TT_UNUSED, TT_BYTE = 1, 2, 3, 4, 5, 6


def _gpt2_byte_encoder() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("\xa1"), ord("\xac") + 1)) + \
         list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


_BYTE_ENC = _gpt2_byte_encoder()
_BYTE_DEC = {v: k for k, v in _BYTE_ENC.items()}


@lru_cache(maxsize=None)
def _is_letter(ch: str) -> bool:
    # cached: pretokenize asks per character and unicodedata.category is the
    # single hottest call in it; the cache is bounded by distinct codepoints
    return unicodedata.category(ch).startswith("L")


@lru_cache(maxsize=None)
def _is_number(ch: str) -> bool:
    return unicodedata.category(ch).startswith("N")


@lru_cache(maxsize=None)
def _is_mark(ch: str) -> bool:
    return unicodedata.category(ch).startswith("M")


def pretokenize(
    text: str, *, include_marks: bool = False, max_number_digits: int = 3
) -> list[str]:
    """Split text the way GPT-2/llama-3 style BPE expects.

    Hand-rolled equivalent of the usual pattern
    (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\\r\\n L N]? L+ | N{1,3}
    |  ?[^\\s L N]+ [\\r\\n]* | \\s*[\\r\\n]+ | \\s+(?!\\S) | \\s+
    implemented with unicodedata so it needs no third-party regex engine.
    """
    out: list[str] = []
    i, n = 0, len(text)
    contractions = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")

    def word_character(value: str) -> bool:
        return _is_letter(value) or (include_marks and _is_mark(value))

    while i < n:
        ch = text[i]

        # every contraction starts with a literal apostrophe, so the slice +
        # lower + scan only ever matters there - not at every position
        if ch == "'":
            low = text[i:i + 3].lower()
            matched = next((c for c in contractions if low.startswith(c)), None)
            if matched:
                out.append(text[i:i + len(matched)])
                i += len(matched)
                continue

        # optional non-letter/number/newline prefix + letters
        if word_character(ch) or (
            ch not in "\r\n" and not _is_number(ch)
            and i + 1 < n and word_character(text[i + 1])
        ):
            j = i if word_character(ch) else i + 1
            k = j
            while k < n and word_character(text[k]):
                k += 1
            if k > j:
                out.append(text[i:k])
                i = k
                continue

        if _is_number(ch):
            k = i
            while (
                k < n and k - i < max_number_digits and _is_number(text[k])
            ):
                k += 1
            out.append(text[i:k])
            i = k
            continue

        if not ch.isspace():
            # punctuation run, optionally preceded by one space, plus newlines
            k = i
            while k < n and not text[k].isspace() and not word_character(text[k]) \
                    and not _is_number(text[k]):
                k += 1
            while k < n and text[k] in "\r\n":
                k += 1
            out.append(text[i:k])
            i = k
            continue

        if ch == " " and i + 1 < n and not text[i + 1].isspace() \
                and not word_character(text[i + 1]) and not _is_number(text[i + 1]):
            # " ?" prefix of a punctuation run
            k = i + 1
            while k < n and not text[k].isspace() and not word_character(text[k]) \
                    and not _is_number(text[k]):
                k += 1
            while k < n and text[k] in "\r\n":
                k += 1
            out.append(text[i:k])
            i = k
            continue

        # whitespace run
        k = i
        while k < n and text[k].isspace():
            k += 1
        run = text[i:k]
        last_nl = max(run.rfind("\r"), run.rfind("\n"))
        if last_nl >= 0:
            out.append(run[:last_nl + 1])
            i += last_nl + 1
            continue
        if k < n and len(run) > 1:   # leave one space to attach to next word
            out.append(run[:-1])
            i = k - 1
            continue
        out.append(run)
        i = k
    return [t for t in out if t]


@dataclass
class Tokenizer:
    model: str                       # "llama" (SPM) or "gpt2" (BPE)
    pre: str = ""                    # tokenizer.ggml.pre contract
    pieces: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    types: list[int] = field(default_factory=list)
    piece_to_id: dict[str, int] = field(default_factory=dict)
    merge_ranks: dict[tuple[str, str], int] = field(default_factory=dict)
    bos_id: int = -1
    eos_id: int = -1
    pad_id: int = -1
    unk_id: int = -1
    add_bos: bool = True
    add_eos: bool = False
    add_space_prefix: bool = True
    eog_ids: set = field(default_factory=set)
    byte_ids: dict[int, int] = field(default_factory=dict)  # byte -> token id
    # SPM: pieces pulled out of the raw text before the merge pass, longest first
    special_ids: list[int] = field(default_factory=list)
    # lazily-built index: first character of a special piece -> its entries
    # as (position in special_ids, sid). A piece whose first character never
    # occurs in the text cannot occur in any fragment, so the scan only
    # visits candidates whose first character is present - Gemma 3 has 6414
    # special pieces and paid a fixed ~0.2 ms on every encode otherwise.
    _special_by_first: dict | None = field(default=None, repr=False)
    # candidate lists memoized per set of present first-characters: on Gemma
    # 3, 6321 of the 6414 specials start with '<', and re-sorting that list
    # on every encode of '<'-containing text cost more than the old scan
    _special_cand: dict = field(default_factory=dict, repr=False)

    # ---- construction --------------------------------------------------

    @classmethod
    def from_gguf(cls, md: dict) -> "Tokenizer":
        model = str(md.get("tokenizer.ggml.model", "llama"))
        t = cls(model=model, pre=str(md.get("tokenizer.ggml.pre", "")))
        t.pieces = list(md.get("tokenizer.ggml.tokens", []))
        t.scores = list(md.get("tokenizer.ggml.scores", [])) or [0.0] * len(t.pieces)
        t.types = list(md.get("tokenizer.ggml.token_type", [])) or [TT_NORMAL] * len(t.pieces)
        t.piece_to_id = {p: i for i, p in enumerate(t.pieces)}
        t.bos_id = int(md.get("tokenizer.ggml.bos_token_id", -1))
        t.eos_id = int(md.get("tokenizer.ggml.eos_token_id", -1))
        t.pad_id = int(md.get("tokenizer.ggml.padding_token_id", -1))
        t.unk_id = int(md.get("tokenizer.ggml.unknown_token_id", -1))
        t.add_bos = bool(md.get("tokenizer.ggml.add_bos_token", model == "llama"))
        t.add_eos = bool(md.get("tokenizer.ggml.add_eos_token", False))
        t.add_space_prefix = bool(md.get("tokenizer.ggml.add_space_prefix", True))

        for i, (p, tt) in enumerate(zip(t.pieces, t.types)):
            if tt == TT_BYTE and len(p) == 6 and p.startswith("<0x"):
                t.byte_ids[int(p[3:5], 16)] = i

        if t.eos_id >= 0:
            t.eog_ids.add(t.eos_id)
        for key in ("tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"):
            if key in md:
                t.eog_ids.add(int(md[key]))
        for name in ("<|eot_id|>", "<|im_end|>", "<|end|>", "<end_of_turn>",
                     "<|endoftext|>", "<|end_of_text|>", "</s>"):
            tid = t.piece_to_id.get(name)
            # only control tokens end generation: Gemma 3 ships "</s>" as an
            # ordinary USER_DEFINED text piece, and halting on it truncates
            # any answer that happens to contain it
            if tid is not None and t.types[tid] == TT_CONTROL:
                t.eog_ids.add(tid)

        # Both tokenizer families distinguish ordinary text from whole-vocab
        # special pieces.  USER_DEFINED pieces always split; CONTROL and
        # UNKNOWN pieces split only when parse_special is requested.  This is
        # observable in Qwen35's GPT-2 vocabulary (for example, <think> is
        # USER_DEFINED while <|im_start|> is CONTROL).
        t.special_ids = sorted(
            (i for i, tt in enumerate(t.types)
             if tt in (TT_CONTROL, TT_USER_DEFINED, TT_UNKNOWN) and t.pieces[i]),
            key=lambda i: len(t.pieces[i]), reverse=True,
        )

        if model == "gpt2":
            merges = md.get("tokenizer.ggml.merges", []) or []
            t.merge_ranks = {}
            for rank, m in enumerate(merges):
                a, _, b = m.partition(" ")
                t.merge_ranks[(a, b)] = rank
        return t

    @property
    def vocab_size(self) -> int:
        return len(self.pieces)

    def piece(self, token_id: int) -> str:
        return self.pieces[token_id] if 0 <= token_id < len(self.pieces) else ""

    def is_eog(self, token_id: int) -> bool:
        return token_id in self.eog_ids

    def token_id(self, piece: str) -> int:
        return self.piece_to_id.get(piece, -1)

    # ---- encoding -------------------------------------------------------

    def encode(self, text: str, add_bos: bool | None = None,
               parse_special: bool = False) -> list[int]:
        ids = (self._encode_spm(text, parse_special) if self.model == "llama"
               else self._encode_bpe(text, parse_special))
        use_bos = self.add_bos if add_bos is None else add_bos
        if use_bos and self.bos_id >= 0:
            ids = [self.bos_id] + ids
        return ids

    # -- SentencePiece ----------------------------------------------------
    #
    # Greedy highest-score-first bigram merge, matching llama.cpp's
    # llm_tokenizer_spm. This is *not* a unigram Viterbi: SPM `scores` are
    # merge priorities, and for a BPE-trained vocab like Gemma 3's they are
    # plain integer ranks, so maximizing their sum would just minimize the sum
    # of token ids and shatter every long piece into short ones.

    def _encode_spm(self, text: str, parse_special: bool = False) -> list[int]:
        ids: list[int] = []
        if not text:
            return ids
        prev_special = True  # the first fragment gets the space prefix too
        for tid, chunk in self._spm_fragments(text, parse_special):
            if tid >= 0:
                ids.append(tid)
                prev_special = True
                continue
            if self.add_space_prefix and prev_special:
                chunk = " " + chunk
            self._spm_merge(chunk.replace(" ", SPM_SPACE), ids)
            prev_special = False
        return ids

    def _spm_fragments(self, text: str, parse_special: bool) -> list[tuple[int, str]]:
        """Split off whole-vocabulary pieces before merging, longest first.

        User-defined pieces always split (Gemma 3 stores its runs of spaces
        that way); control and unknown pieces only when the caller asks, so a
        user who types `<start_of_turn>` cannot forge a turn boundary.
        """
        candidates = self.special_ids
        if candidates:
            by_first = self._special_by_first
            if by_first is None:
                by_first = {}
                for oi, sid in enumerate(self.special_ids):
                    by_first.setdefault(self.pieces[sid][0], []).append((oi, sid))
                self._special_by_first = by_first
            present = frozenset(set(text) & by_first.keys())
            if not present:
                return [(-1, text)]
            candidates = self._special_cand.get(present)
            if candidates is None:
                if len(self._special_cand) > 64:
                    self._special_cand.clear()  # unbounded text, bounded cache
                # keep the global longest-first order across selected groups
                candidates = [sid for _, sid in
                              sorted(t for fc in present for t in by_first[fc])]
                self._special_cand[present] = candidates
        frags: list[tuple[int, str]] = [(-1, text)]
        for sid in candidates:
            if not parse_special and self.types[sid] in (TT_CONTROL, TT_UNKNOWN):
                continue
            piece = self.pieces[sid]
            out: list[tuple[int, str]] = []
            for tid, chunk in frags:
                if tid >= 0 or piece not in chunk:
                    out.append((tid, chunk))
                    continue
                pos = 0
                while True:
                    hit = chunk.find(piece, pos)
                    if hit < 0:
                        break
                    if hit > pos:
                        out.append((-1, chunk[pos:hit]))
                    out.append((sid, ""))
                    pos = hit + len(piece)
                if pos < len(chunk):
                    out.append((-1, chunk[pos:]))
            frags = out
        return frags

    def _spm_merge(self, text: str, out: list[int]) -> None:
        """One symbol per character, then merge the best-scoring adjacent pair
        until no adjacent pair is a vocabulary piece."""
        n = len(text)
        if n == 0:
            return
        start = list(range(n))
        size = [1] * n
        prev = list(range(-1, n - 1))
        nxt = list(range(1, n + 1))
        nxt[n - 1] = -1
        heap: list[tuple[float, int, int, int]] = []
        get = self.piece_to_id.get
        scores = self.scores

        def add_bigram(left: int, right: int) -> None:
            if left < 0 or right < 0:
                return
            width = size[left] + size[right]
            tid = get(text[start[left]:start[left] + width])
            if tid is not None:
                # max-heap on score; ties go to the leftmost pair, and `width`
                # dates the entry so a stale one is dropped rather than applied
                heapq.heappush(heap, (-scores[tid], left, right, width))

        for i in range(1, n):
            add_bigram(i - 1, i)

        while heap:
            _, left, right, width = heapq.heappop(heap)
            if size[left] == 0 or size[right] == 0 or size[left] + size[right] != width:
                continue  # one side was swallowed by an earlier merge
            size[left] = width
            size[right] = 0
            nxt[left] = nxt[right]
            if nxt[right] >= 0:
                prev[nxt[right]] = left
            add_bigram(prev[left], left)
            add_bigram(left, nxt[left])

        i = 0
        while i >= 0:
            piece = text[start[i]:start[i] + size[i]]
            tid = get(piece)
            if tid is not None:
                out.append(tid)
            else:
                # only ever a single character: anything merged was a piece
                for b in piece.encode("utf-8"):
                    bid = self.byte_ids.get(b, self.unk_id)
                    if bid >= 0:
                        out.append(bid)
            i = nxt[i]

    def _encode_bpe(self, text: str, parse_special: bool = False) -> list[int]:
        ids: list[int] = []
        get_rank = self.merge_ranks.get
        for special_id, fragment in self._spm_fragments(text, parse_special):
            if special_id >= 0:
                ids.append(special_id)
                continue
            for chunk in pretokenize(
                fragment,
                include_marks=self.pre == "qwen35",
                max_number_digits=1 if self.pre == "qwen35" else 3,
            ):
                self._bpe_merge_chunk(chunk, ids, get_rank)
        return ids

    def _bpe_merge_chunk(self, chunk: str, ids: list[int], get_rank) -> None:
        """Merge one pre-tokenized byte-level BPE chunk into *ids*."""
        if chunk:
            word = [_BYTE_ENC[b] for b in chunk.encode("utf-8")]
            n = len(word)
            if 1 < n <= 16:
                # ordinary words: the rescan-per-merge loop's constant beats
                # the heap's on short chunks (measured ~10% on plain prose)
                while len(word) > 1:
                    best_rank = None
                    best_i = -1
                    for i in range(len(word) - 1):
                        r = get_rank((word[i], word[i + 1]))
                        if r is not None and (best_rank is None or r < best_rank):
                            best_rank, best_i = r, i
                    if best_i < 0:
                        break
                    word[best_i:best_i + 2] = [word[best_i] + word[best_i + 1]]
            elif n > 16:
                # lowest-rank pair first, leftmost on rank ties - the same
                # order the rescan loop above produces, but via a lazy heap
                # over a linked list so a long chunk costs O(L log L) instead
                # of O(L^2) (unbroken CJK arrives as ONE chunk, and the old
                # loop took ~5 s on 4000 characters of it)
                sym = word[:]           # live piece at each original slot
                prev = list(range(-1, n - 1))
                nxt = list(range(1, n + 1))
                nxt[n - 1] = -1
                heap: list[tuple[int, int, int, str, str]] = []

                def push(left: int, right: int) -> None:
                    if left < 0 or right < 0:
                        return
                    rank = get_rank((sym[left], sym[right]))
                    if rank is not None:
                        heapq.heappush(heap, (rank, left, right,
                                              sym[left], sym[right]))

                for i in range(n - 1):
                    push(i, i + 1)
                while heap:
                    _, left, right, a, b = heapq.heappop(heap)
                    if nxt[left] != right or sym[left] != a or sym[right] != b:
                        continue  # one side changed since this entry was made
                    sym[left] = a + b
                    sym[right] = ""   # dead: dates any remaining stale entry
                    nxt[left] = nxt[right]
                    if nxt[right] >= 0:
                        prev[nxt[right]] = left
                    push(prev[left], left)
                    push(left, nxt[left])
                word = []
                i = 0
                while i >= 0:
                    word.append(sym[i])
                    i = nxt[i]
            for piece in word:
                tid = self.piece_to_id.get(piece)
                if tid is not None:
                    ids.append(tid)
                else:  # last resort: per-character lookup
                    for ch in piece:
                        tid = self.piece_to_id.get(ch)
                        if tid is not None:
                            ids.append(tid)

    # ---- decoding -------------------------------------------------------

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.token_bytes(i) for i in ids).decode("utf-8", errors="replace")

    def token_bytes(self, token_id: int) -> bytes:
        """Raw bytes a token contributes to output (may be partial UTF-8)."""
        if not (0 <= token_id < len(self.pieces)):
            return b""
        p = self.pieces[token_id]
        tt = self.types[token_id]
        if tt in (TT_CONTROL, TT_UNKNOWN, TT_UNUSED):
            return b""
        if self.model == "llama":
            if tt == TT_BYTE and p.startswith("<0x"):
                return bytes([int(p[3:5], 16)])
            return p.replace(SPM_SPACE, " ").encode("utf-8")
        return bytes(_BYTE_DEC.get(ch, ord(" ")) for ch in p)


class StreamDecoder:
    """Incremental detokenizer that holds back partial UTF-8 sequences."""

    def __init__(self, tokenizer: Tokenizer):
        self.tok = tokenizer
        self.pending = b""

    def feed(self, token_id: int) -> str:
        self.pending += self.tok.token_bytes(token_id)
        out = ""
        while self.pending:
            # emit the longest prefix that is valid UTF-8
            for cut in range(len(self.pending), max(len(self.pending) - 4, -1), -1):
                try:
                    out += self.pending[:cut].decode("utf-8")
                    self.pending = self.pending[cut:]
                    return out
                except UnicodeDecodeError:
                    continue
            # No prefix decoded, so the head is broken rather than merely
            # truncated. Only the last 3 bytes can still be the start of a
            # character; replace the rest instead of holding a poisoned buffer
            # and emitting nothing for the whole rest of the response - which
            # also stopped stop-strings from ever matching again.
            if len(self.pending) <= 3:
                return out
            out += self.pending[:-3].decode("utf-8", errors="replace")
            self.pending = self.pending[-3:]
        return out

    def flush(self) -> str:
        text = self.pending.decode("utf-8", errors="replace")
        self.pending = b""
        return text
