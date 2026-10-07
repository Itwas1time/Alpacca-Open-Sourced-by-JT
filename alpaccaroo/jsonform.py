# Alpaccaroo - guaranteed-valid-JSON generation: a byte-level JSON grammar
# guard plus the candidate-rejection sampling loop chat.generate runs for
# json_only=True. MIT License. See LICENSE.
# (These first two lines must never say "*oding:" - PEP 263 reads that as an
# encoding declaration.)
from __future__ import annotations

# The guard walks raw token BYTES, not decoded text: byte-level BPE tokens
# can split a UTF-8 character across tokens, and a str-based validator would
# have to hold partial sequences exactly like StreamDecoder does. Bytes need
# no holding at all - a partial UTF-8 sequence only ever occurs inside a JSON
# string, where every byte >= 0x20 is legal as-is.
_VALUE = 0        # expecting the start of a value (leading whitespace ok)
_OBJ_FIRST = 1    # just after '{': a key or the closing '}'
_OBJ_KEY = 2      # after ',' in an object: a key only
_OBJ_COLON = 3    # after a key: ':' only
_OBJ_NEXT = 4     # after a member value: ',' or '}'
_ARR_NEXT = 5     # after an array element: ',' or ']'
_STRING = 6
_STR_ESC = 7      # after '\' inside a string
_STR_HEX = 8      # inside \uXXXX, hex_left digits still owed
_NUMBER = 9
_LITERAL = 10     # inside true/false/null
_DONE = 11        # top-level value complete: nothing more is allowed

# nesting-stack frames
_F_OBJ = 0
_F_ARR = 1
_F_ARR_FIRST = 2  # '[' seen, no element yet: ']' is still legal

# Number sub-states. A number has no closing delimiter of its own: it ends at
# the first byte that cannot extend it, which is then re-dispatched to the
# enclosing context. At top level there IS no enclosing context, so a bare
# number never reaches _DONE - EOG while it is terminal ends it instead
# (see at_valid_end), because stopping at the first terminal digit would
# truncate "42" to "4".
_N_SIGN, _N_ZERO, _N_INT, _N_DOT, _N_FRAC, _N_E, _N_ESIGN, _N_EXP = range(8)
_N_TERMINAL = frozenset((_N_ZERO, _N_INT, _N_FRAC, _N_EXP))

_WS = frozenset(b" \t\r\n")
_DIGITS = frozenset(b"0123456789")
_HEX = frozenset(b"0123456789abcdefABCDEF")
_ESCAPES = frozenset(b'"\\/bfnrt')

#: nesting cap; RFC 8259 sets no limit, but an unbounded stack lets a model
#: that latches onto '[' grow memory forever inside one reply
MAX_DEPTH = 256


class JsonGuard:
    """Tracks whether a byte stream is a prefix of one valid JSON value.

    feed() is transactional: it consumes the whole byte string or none of
    it, so the sampling loop probes a candidate token with feed() and simply
    bans it on False. can_accept() answers without advancing, via copy().
    """

    __slots__ = ("state", "stack", "key", "hex_left", "num", "lit", "lit_pos")

    def __init__(self):
        self.state = _VALUE
        self.stack: list[int] = []
        self.key = False       # is the string being read an object key?
        self.hex_left = 0
        self.num = _N_SIGN
        self.lit = b""
        self.lit_pos = 0

    def copy(self) -> "JsonGuard":
        g = JsonGuard.__new__(JsonGuard)
        g.state = self.state
        g.stack = self.stack[:]
        g.key = self.key
        g.hex_left = self.hex_left
        g.num = self.num
        g.lit = self.lit
        g.lit_pos = self.lit_pos
        return g

    @property
    def done(self) -> bool:
        """The top-level value closed; no byte can extend it."""
        return self.state == _DONE

    @property
    def at_valid_end(self) -> bool:
        """The bytes so far already form one complete JSON value: _DONE, or
        a top-level number that is valid if it ends right here."""
        return self.state == _DONE or (self.state == _NUMBER
                                       and not self.stack
                                       and self.num in _N_TERMINAL)

    def feed(self, data: bytes) -> bool:
        """Consume `data` if every byte keeps the stream a valid JSON
        prefix; otherwise change nothing and return False."""
        if not data:
            return True   # zero-width tokens (control pieces) change nothing
        snap = (self.state, tuple(self.stack), self.key, self.hex_left,
                self.num, self.lit, self.lit_pos)
        for b in data:
            if not self._step(b):
                (self.state, stack, self.key, self.hex_left,
                 self.num, self.lit, self.lit_pos) = snap
                self.stack = list(stack)
                return False
        return True

    def can_accept(self, data: bytes) -> bool:
        """Would feed(data) succeed? Never advances the guard."""
        return self.copy().feed(data)

    # ---- byte transitions ----------------------------------------------

    def _step(self, b: int) -> bool:
        s = self.state
        if s == _STRING:
            if b == 0x22:                      # '"'
                if self.key:
                    self.key = False
                    self.state = _OBJ_COLON
                else:
                    self._end_value()
                return True
            if b == 0x5C:                      # '\'
                self.state = _STR_ESC
                return True
            # any byte >= 0x20 is legal string content, including lone UTF-8
            # continuation bytes: StreamDecoder repairs malformed sequences
            # with U+FFFD, which still parses inside a string. Raw control
            # bytes < 0x20 are what RFC 8259 forbids unescaped.
            return b >= 0x20
        if s == _STR_ESC:
            if b in _ESCAPES:
                self.state = _STRING
                return True
            if b == 0x75:                      # 'u'
                self.state = _STR_HEX
                self.hex_left = 4
                return True
            return False
        if s == _STR_HEX:
            if b in _HEX:
                self.hex_left -= 1
                if self.hex_left == 0:
                    self.state = _STRING
                return True
            return False
        if s == _NUMBER:
            nxt = _number_step(self.num, b)
            if nxt >= 0:
                self.num = nxt
                return True
            if self.num in _N_TERMINAL:
                # the number ends at the first byte it cannot absorb; hand
                # that byte to the enclosing context. "1." or "2e" are not
                # terminal, so a delimiter there fails instead.
                self._end_value()
                return self._step(b)
            return False
        if s == _LITERAL:
            if b == self.lit[self.lit_pos]:
                self.lit_pos += 1
                if self.lit_pos == len(self.lit):
                    self._end_value()
                return True
            return False
        if b in _WS:
            # whitespace is legal between any two structural elements but
            # NOT after the top-level value closes: the guard's caller stops
            # at done, so trailing bytes would never be checked again
            return s != _DONE
        if s == _VALUE:
            if self.stack and self.stack[-1] == _F_ARR_FIRST:
                if b == 0x5D:                  # ']' straight after '['
                    self.stack.pop()
                    self._end_value()
                    return True
                self.stack[-1] = _F_ARR        # first element is starting
            return self._begin_value(b)
        if s == _OBJ_FIRST:
            if b == 0x7D:                      # '}': empty object
                self.stack.pop()
                self._end_value()
                return True
            s = _OBJ_KEY                       # otherwise same rule as a key
        if s == _OBJ_KEY:
            if b == 0x22:
                self.state = _STRING
                self.key = True
                return True
            return False
        if s == _OBJ_COLON:
            if b == 0x3A:                      # ':'
                self.state = _VALUE
                return True
            return False
        if s == _OBJ_NEXT:
            if b == 0x2C:                      # ','
                self.state = _OBJ_KEY
                return True
            if b == 0x7D:
                self.stack.pop()
                self._end_value()
                return True
            return False
        if s == _ARR_NEXT:
            if b == 0x2C:
                self.state = _VALUE
                return True
            if b == 0x5D:
                self.stack.pop()
                self._end_value()
                return True
            return False
        return False                           # _DONE: nothing may follow

    def _begin_value(self, b: int) -> bool:
        if b == 0x7B:                          # '{'
            if len(self.stack) >= MAX_DEPTH:
                return False
            self.stack.append(_F_OBJ)
            self.state = _OBJ_FIRST
            return True
        if b == 0x5B:                          # '['
            if len(self.stack) >= MAX_DEPTH:
                return False
            self.stack.append(_F_ARR_FIRST)
            self.state = _VALUE
            return True
        if b == 0x22:                          # '"'
            self.state = _STRING
            self.key = False
            return True
        if b == 0x2D:                          # '-'
            self.state = _NUMBER
            self.num = _N_SIGN
            return True
        if b == 0x30:                          # '0': JSON forbids 01, 007...
            self.state = _NUMBER
            self.num = _N_ZERO
            return True
        if b in _DIGITS:
            self.state = _NUMBER
            self.num = _N_INT
            return True
        if b == 0x74:                          # 't'
            self.state = _LITERAL
            self.lit = b"true"
            self.lit_pos = 1
            return True
        if b == 0x66:                          # 'f'
            self.state = _LITERAL
            self.lit = b"false"
            self.lit_pos = 1
            return True
        if b == 0x6E:                          # 'n'
            self.state = _LITERAL
            self.lit = b"null"
            self.lit_pos = 1
            return True
        return False

    def _end_value(self) -> None:
        if not self.stack:
            self.state = _DONE
        elif self.stack[-1] == _F_OBJ:
            self.state = _OBJ_NEXT
        else:
            self.state = _ARR_NEXT


def _number_step(n: int, b: int) -> int:
    """Next number sub-state for byte `b`, or -1 when it cannot extend."""
    if n == _N_SIGN:
        if b == 0x30:
            return _N_ZERO
        return _N_INT if b in _DIGITS else -1
    if n == _N_ZERO:
        if b == 0x2E:                          # '.'
            return _N_DOT
        return _N_E if b in (0x65, 0x45) else -1
    if n == _N_INT:
        if b in _DIGITS:
            return _N_INT
        if b == 0x2E:
            return _N_DOT
        return _N_E if b in (0x65, 0x45) else -1
    if n == _N_DOT:
        return _N_FRAC if b in _DIGITS else -1
    if n == _N_FRAC:
        if b in _DIGITS:
            return _N_FRAC
        return _N_E if b in (0x65, 0x45) else -1
    if n == _N_E:
        if b in _DIGITS:
            return _N_EXP
        return _N_ESIGN if b in (0x2B, 0x2D) else -1
    # _N_ESIGN and _N_EXP both continue only on digits
    return _N_EXP if b in _DIGITS else -1


# ---- sampler integration ------------------------------------------------

def token_bytes_table(tok) -> list[bytes]:
    """Token id -> the raw bytes it contributes to output, built once per
    tokenizer. Uses the same token_bytes() mapping StreamDecoder decodes
    with, so the guard judges exactly the bytes the caller will see.

    Cached on the tokenizer instance itself (it is an eq-comparing
    dataclass, so it cannot key a WeakKeyDictionary); the cache then lives
    and dies with the model that owns the tokenizer. Building it costs one
    token_bytes() call per vocab entry (measured 83 ms on the 128256-token
    llama3 vocabulary), too much to repeat every request."""
    table = getattr(tok, "_jsonform_bytes", None)
    if table is None:
        table = [tok.token_bytes(i) for i in range(tok.vocab_size)]
        tok._jsonform_bytes = table
    return table


#: resample attempts before switching to the ranked full-vocab scan; the
#: loop all but always ends immediately (measured on 10 llama3.2:1b replies
#: at temperature 0.9: 33 rejects over 807 positions, never more than 4 at
#: one position), so 512 means the distribution is saturated with invalid
#: continuations and one ranked scan beats resampling on
MAX_REJECTS = 512


def sample_json_token(sampler, logits, guard: JsonGuard, table: list[bytes],
                      tok, stop_tokens: set[int] | None = None) -> int:
    """Sample one token whose bytes keep the stream a valid JSON prefix.

    The chosen token's bytes are already fed into `guard` on return. EOG and
    turn-start tokens are allowed only once the value is complete; before
    that they are banned and the position resampled, so a model that gives
    up mid-object is pushed to close it instead. Raises ValueError when no
    token in the vocabulary can continue the JSON - unreachable for any
    tokenizer that covers ASCII, since a structural byte always continues.
    """
    banned: set[int] = set()
    for _ in range(MAX_REJECTS):
        tid = sampler.sample(logits, banned=banned or None)
        if tok.is_eog(tid) or (stop_tokens and tid in stop_tokens):
            if guard.at_valid_end:
                return tid
            banned.add(tid)
            continue
        if guard.feed(table[tid] if 0 <= tid < len(table) else b""):
            return tid
        banned.add(tid)
    # ranked scan: highest logit first, first token the guard accepts. This
    # path is the guarantee, not the fast path - it triggers only when the
    # distribution is saturated with invalid continuations (or the whole
    # vocabulary is smaller than MAX_REJECTS, as the tiny fixtures are).
    lo = list(logits)
    for tid in sorted(range(min(len(lo), len(table))), key=lo.__getitem__,
                      reverse=True):
        if tok.is_eog(tid) or (stop_tokens and tid in stop_tokens):
            if guard.at_valid_end:
                return tid
            continue
        b = table[tid]
        # zero-width tokens make no progress, so the scan skips them: picking
        # one forever would spin the budget without ever finishing the value
        if b and guard.feed(b):
            return tid
    raise ValueError("no tokenizable continuation of valid JSON")
