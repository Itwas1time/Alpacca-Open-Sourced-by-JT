# Alpaccaroo - chat formatting and generation loops.
# MIT License. See LICENSE.
from __future__ import annotations

import hashlib
import sys
import time
from dataclasses import dataclass, field
from time import perf_counter as _perf
from typing import TextIO

from . import profiling as _prof
from .model import Model
from .sample import Sampler, SamplerParams
from .tokenizer import StreamDecoder

# Known chat formats, detected from the model's embedded chat template.
# Each entry: (needle in template, format name)
_FORMAT_NEEDLES = [
    ("<|start_header_id|>", "llama3"),
    ("<|im_start|>", "chatml"),
    ("<start_of_turn>", "gemma"),
    ("[INST]", "llama2"),
    ("<|user|>", "zephyr"),
]

_QWEN35_XHIGH_INSTRUCTION = (
    "Reasoning effort is set to xhigh. Please think carefully through the task, "
    "validate key assumptions, consider plausible alternatives, and prioritize "
    "correctness, consistency, and clarity in the final answer."
)
_QWEN35_LOW_INSTRUCTION = (
    "Reasoning effort is set to low. Keep your thinking brief and focused, "
    "moving directly to the conclusion without unnecessary elaboration."
)
_QWEN35_SMALL_TEMPLATE_SHA256 = (
    "7f0e529032c25183bcd66c7f238da2d377f43be754a94e2725a58c4e16d2ed67"
)
_QWEN35_PRIMARY_TEMPLATE_SHA256 = (
    "12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce"
)

# Pieces that open a turn. If the model emits one it has started speaking as
# somebody else and its own reply is over. They are control tokens, so they
# decode to empty text and a stop *string* can never catch them - without this
# a runaway turn silently role-plays the user with no visible separator.
_FORMAT_TURN_STARTS = {
    "llama3": ("<|start_header_id|>",),
    "chatml": ("<|im_start|>",),
    "gemma": ("<start_of_turn>",),
    "zephyr": ("<|user|>", "<|system|>"),
    "qwen35": ("<|im_start|>",),
}


def detect_format(metadata: dict) -> str:
    if metadata.get("tokenizer.ggml.pre") == "qwen35":
        return "qwen35"
    template = str(metadata.get("tokenizer.chat_template", ""))
    for needle, name in _FORMAT_NEEDLES:
        if needle in template:
            return name
    return "raw" if not template else "chatml"


def render_qwen35_text(
    messages: list[dict],
    template: str,
    *,
    add_generation_prompt: bool = True,
    enable_thinking: bool | None = None,
    reasoning_effort: str | None = None,
    preserve_thinking: bool | None = None,
) -> str:
    """Render the pinned Qwen35 text-only template contract exactly.

    The 0.8B/2B and 27B artifacts carry different template revisions.  This
    implements their shared text/chat surface and deliberately rejects tools
    and multimodal content until those deferred features have their own
    runtime contract.  ``None`` means the template variable is undefined,
    which matters: the small template defaults thinking off while the pinned
    27B template defaults to xhigh thinking.
    """
    if not messages:
        raise ValueError("Qwen35 chat template requires at least one message")
    template_sha256 = hashlib.sha256(template.encode("utf-8")).hexdigest()
    if template_sha256 == _QWEN35_PRIMARY_TEMPLATE_SHA256:
        primary_template = True
    elif template_sha256 == _QWEN35_SMALL_TEMPLATE_SHA256:
        primary_template = False
    else:
        raise ValueError(
            "unsupported Qwen35 chat-template revision: "
            f"sha256={template_sha256}"
        )

    def content(message: dict) -> str:
        value = message.get("content")
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ValueError(
                "Qwen35 vision/multimodal chat content is unsupported"
            )
        return value.strip()

    out: list[str] = []
    first_conversation_index = 0
    if primary_template:
        merged_system: list[str] = []
        for index, message in enumerate(messages):
            if index != first_conversation_index or message.get("role") not in (
                "system", "developer"
            ):
                break
            value = content(message)
            if value:
                merged_system.append(value)
            first_conversation_index += 1

        instruction = ""
        if enable_thinking is not False:
            effort = reasoning_effort or "xhigh"
            if effort == "high":
                effort = "xhigh"
            if effort not in ("xhigh", "medium", "low"):
                raise ValueError(
                    "Qwen35 reasoning_effort must be xhigh, high, medium, or low"
                )
            if effort == "xhigh":
                instruction = _QWEN35_XHIGH_INSTRUCTION
            elif effort == "low":
                instruction = _QWEN35_LOW_INSTRUCTION
        system = "\n".join(merged_system)
        if system or instruction:
            body = instruction + ("\n\n" if instruction and system else "") + system
            out.append(f"<|im_start|>system\n{body}<|im_end|>\n")
    elif messages[0].get("role") == "system":
        out.append(
            f"<|im_start|>system\n{content(messages[0])}<|im_end|>\n"
        )
        first_conversation_index = 1

    last_query_index = -1
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            last_query_index = index
            break
    if last_query_index < 0:
        raise ValueError("Qwen35 chat template requires a user query")

    for index, message in enumerate(messages[first_conversation_index:],
                                    first_conversation_index):
        role = message.get("role")
        body = content(message)
        if role in ("system", "developer"):
            raise ValueError("Qwen35 system/developer messages must be first")
        if role == "user":
            out.append(f"<|im_start|>user\n{body}<|im_end|>\n")
            continue
        if role != "assistant":
            raise ValueError(
                f"Qwen35 chat role {role!r} is unsupported; tools are deferred"
            )

        reasoning = message.get("reasoning_content")
        if not isinstance(reasoning, str):
            reasoning = ""
            if not primary_template and "</think>" in body:
                reasoning = body.split("</think>", 1)[0].rstrip("\n")
                reasoning = reasoning.split("<think>")[-1].lstrip("\n")
                body = body.split("</think>")[-1].lstrip("\n")
        reasoning = reasoning.strip()
        body = body.strip()
        include_reasoning = (
            (primary_template and preserve_thinking is not False)
            or index > last_query_index
        )
        if include_reasoning:
            out.append(
                f"<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{body}"
            )
        else:
            out.append(f"<|im_start|>assistant\n{body}")
        out.append("<|im_end|>\n")

    if add_generation_prompt:
        out.append("<|im_start|>assistant\n")
        thinking_on = (
            enable_thinking is True
            if not primary_template
            else enable_thinking is not False
        )
        out.append("<think>\n" if thinking_on else "<think>\n\n</think>\n\n")
    return "".join(out)


@dataclass
class ChatFormat:
    """Renders a conversation into token ids for a given format."""
    model: Model
    name: str
    _id_cache: dict = field(default_factory=dict, init=False, repr=False)

    def _ids(self, text: str, add_bos: bool = False) -> list[int]:
        # role names and separators recur every message; on a tokenizer with
        # thousands of special pieces (Gemma 3) each encode of even "\n"
        # pays a fixed scan, so memoize the short constants. Copies out so a
        # caller can never mutate a cached entry.
        if len(text) <= 32:
            key = (text, add_bos)
            hit = self._id_cache.get(key)
            if hit is None:
                hit = self.model.tok.encode(text, add_bos=add_bos)
                self._id_cache[key] = hit
            return list(hit)
        return self.model.tok.encode(text, add_bos=add_bos)

    def _special(self, piece: str) -> list[int]:
        tid = self.model.tok.token_id(piece)
        return [tid] if tid >= 0 else self._ids(piece)

    def stop_tokens(self) -> set[int]:
        """Token ids that end the assistant's turn on top of the EOG set."""
        out = set()
        for piece in _FORMAT_TURN_STARTS.get(self.name, ()):
            tid = self.model.tok.token_id(piece)
            if tid >= 0:
                out.add(tid)
        return out

    def render(
        self, messages: list[dict], add_generation_prompt: bool = True,
        *, enable_thinking: bool | None = None,
        reasoning_effort: str | None = None,
        preserve_thinking: bool | None = None,
    ) -> list[int]:
        tok = self.model.tok
        ids: list[int] = []
        if self.name == "qwen35":
            rendered = render_qwen35_text(
                messages,
                str(self.model.metadata.get("tokenizer.chat_template", "")),
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
                reasoning_effort=reasoning_effort,
                preserve_thinking=preserve_thinking,
            )
            return tok.encode(rendered, add_bos=False, parse_special=True)
        if self.name == "llama3":
            if tok.bos_id >= 0:
                ids.append(tok.bos_id)
            for m in messages:
                ids += self._special("<|start_header_id|>")
                ids += self._ids(m["role"])
                ids += self._special("<|end_header_id|>")
                ids += self._ids("\n\n" + m["content"])
                ids += self._special("<|eot_id|>")
            if add_generation_prompt:
                ids += self._special("<|start_header_id|>")
                ids += self._ids("assistant")
                ids += self._special("<|end_header_id|>")
                ids += self._ids("\n\n")
            return ids

        if self.name == "chatml":
            for m in messages:
                ids += self._special("<|im_start|>")
                ids += self._ids(m["role"] + "\n")
                ids += self._ids(m["content"])
                ids += self._special("<|im_end|>")
                ids += self._ids("\n")
            if add_generation_prompt:
                ids += self._special("<|im_start|>")
                ids += self._ids("assistant\n")
            return ids

        if self.name == "gemma":
            # the gemma template opens with {{ bos_token }} and the models are
            # trained with <bos> at position 0; without it they answer empty
            if tok.bos_id >= 0:
                ids.append(tok.bos_id)
            # Gemma has no system turn: its template folds a leading system
            # message into the first user turn as a prefix, and raises rather
            # than emit two user turns in a row. Rendering it as its own turn
            # produced a sequence the model was never trained on.
            prefix = ""
            turns = messages
            if messages and messages[0].get("role") == "system":
                prefix = messages[0]["content"] + "\n\n"
                turns = messages[1:]
            for i, m in enumerate(turns):
                role = "model" if m["role"] == "assistant" else "user"
                ids += self._special("<start_of_turn>")
                # one encode call: the template concatenates these before the
                # tokenizer sees them, and splitting the call would tokenize
                # across a boundary the model never saw
                ids += self._ids(role + "\n" + (prefix if i == 0 else "") +
                                 m["content"].strip())
                ids += self._special("<end_of_turn>")
                ids += self._ids("\n")
            if add_generation_prompt:
                ids += self._special("<start_of_turn>")
                ids += self._ids("model\n")
            return ids

        if self.name == "llama2":
            system = ""
            convo = []
            for m in messages:
                if m["role"] == "system":
                    system = m["content"]
                else:
                    convo.append(m)
            text = ""
            for i, m in enumerate(convo):
                if m["role"] == "user":
                    content = m["content"]
                    if system and i == 0:
                        content = f"<<SYS>>\n{system}\n<</SYS>>\n\n{content}"
                    text += f"[INST] {content} [/INST]"
                else:
                    text += f" {m['content']} "
            return self._ids(text, add_bos=True)

        if self.name == "zephyr":
            for m in messages:
                ids += self._special(f"<|{m['role']}|>")
                ids += self._ids("\n" + m["content"])
                if tok.eos_id >= 0:
                    ids.append(tok.eos_id)
            if add_generation_prompt:
                ids += self._special("<|assistant|>")
                ids += self._ids("\n")
            return ids

        # raw: plain completion with a simple convention
        text = ""
        for m in messages:
            prefix = {"system": "", "user": "User: ", "assistant": "Assistant: "}.get(m["role"], "")
            text += prefix + m["content"] + "\n"
        if add_generation_prompt:
            text += "Assistant:"
        return self._ids(text, add_bos=True)


@dataclass
class GenerationResult:
    text: str
    tokens: int
    seconds: float
    prompt_tokens: int = 0
    # why generation ended: "eog" (the model finished), "stop" (a stop string
    # matched), "length" (the n_predict budget ran out) or "context" (no room
    # left in the context window). "context" with tokens == 0 is the caller's
    # signal that the prompt itself left nothing to generate into.
    stop_reason: str = "eog"

    @property
    def tok_per_sec(self) -> float:
        return self.tokens / self.seconds if self.seconds > 0 else 0.0


def generate(model: Model, prompt_ids: list[int], params: SamplerParams,
             n_predict: int = -1, stream=None, stop_strings: list[str] | None = None,
             stop_tokens: set[int] | None = None, *,
             json_only: bool = False, check_cancelled=None) -> GenerationResult:
    """Generate until EOG / n_predict / a stop string. `stream` is an
    optional callable receiving text fragments as they decode.

    With json_only=True every emitted fragment is a prefix of one
    syntactically valid JSON value, on any backend at any temperature, and
    generation stops with reason "stop" the moment the value completes."""
    if check_cancelled is not None:
        check_cancelled()
    if not prompt_ids:
        if model.tok.bos_id < 0:
            raise ValueError("prompt produced no tokens and the tokenizer has no BOS token")
        prompt_ids = [model.tok.bos_id]
    guard = table = None
    if json_only:
        from .jsonform import JsonGuard, sample_json_token, token_bytes_table
        guard = JsonGuard()
        table = token_bytes_table(model.tok)
    sampler = Sampler(params)
    _profiler = _prof.ACTIVE
    # only the last repeat_last_n tokens can ever remain in the penalty
    # window, so skip the rest rather than walk a 16k prompt to fill 64 slots
    for t in prompt_ids[-max(params.repeat_last_n, 1):]:
        sampler.accept(t)
    logits = (model.prefill(prompt_ids, check_cancelled=check_cancelled)
              if check_cancelled is not None else model.prefill(prompt_ids))

    dec = StreamDecoder(model.tok)
    emitted = 0
    n_tokens = 0
    t0 = time.time()
    # 0 means "no new tokens" (prefill only, matching Ollama/llama.cpp);
    # only negative values mean "until EOG or the context fills"
    if n_predict >= 0:
        budget = n_predict
    else:
        budget = model.n_ctx - model.n_past

    text = ""
    reason = "eog"
    truncated = False
    # a stop string is only detectable once its last character arrives, so
    # hold back that much of the tail or the caller sees the beginning of it
    hold = max((len(s) for s in stop_strings or [] if s), default=1) - 1
    while True:
        if check_cancelled is not None:
            check_cancelled()
        if model.n_past >= model.n_ctx:
            reason = "context"   # checked first: no room beats no budget
            break
        if n_tokens >= budget:
            reason = "length"
            break
        if guard is None:
            tid = sampler.sample(logits)
        else:
            # rejection loop: only a token whose bytes extend valid JSON can
            # come back, and its bytes are already fed into the guard. EOG
            # only comes back once the value is complete.
            tid = sample_json_token(sampler, logits, guard, table,
                                    model.tok, stop_tokens)
        sampler.accept(tid)
        if model.tok.is_eog(tid):
            break  # never forwarded into the cache, so never counted either
        if stop_tokens and tid in stop_tokens:
            reason = "stop"   # a turn-start token: the reply is complete
            break
        n_tokens += 1
        if _profiler is not None:
            _t = _perf()
            piece = dec.feed(tid)
            _profiler.add("tokenizer_stream", _perf() - _t)
        else:
            piece = dec.feed(tid)
        text += piece
        if guard is not None and guard.done:
            # the top-level value just closed; checked before stop strings so
            # the closing token itself cannot be truncated. Caller-supplied
            # stop strings still win at every EARLIER token - a stop that
            # matches inside a JSON string value cuts the reply mid-value, so
            # callers wanting the completeness guarantee must not combine
            # stop_strings with json_only.
            reason = "stop"
            break
        if stop_strings and piece:
            # a match must involve the newly decoded piece - anything fully
            # inside older text was found on an earlier token - so search
            # only the tail the piece could participate in: a stop of length
            # L ending inside the piece starts at most L-1 (= hold) before it
            scan = max(0, len(text) - len(piece) - hold)
            tail = text[scan:]
            hit = next((s for s in stop_strings if s and s in tail), None)
            if hit:
                text = text[:scan + tail.index(hit)]
                reason = "stop"
                truncated = True
                break
        if stream is not None and len(text) - hold > emitted:
            stream(text[emitted:len(text) - hold])
            emitted = len(text) - hold
        if n_tokens >= budget:
            reason = "length"
            break
        logits = model.forward(tid)
    if truncated:
        # only a stop STRING invalidates the tail: those bytes are part of the
        # match. A stop token breaks before decoding, so its pending bytes are
        # ordinary text and still belong in the answer.
        dec.pending = b""
    else:
        text += dec.flush()
    if stream is not None and len(text) > emitted:
        stream(text[emitted:])
    return GenerationResult(text=text, tokens=n_tokens, seconds=time.time() - t0,
                            prompt_tokens=len(prompt_ids), stop_reason=reason)


#: tokens held back for the reply when trimming a conversation to fit
REPLY_RESERVE = 128


def fit_to_context(fmt: "ChatFormat", messages: list[dict], n_ctx: int,
                   reserve: int = REPLY_RESERVE, *,
                   render_options: dict | None = None) -> tuple[list[int], int]:
    """Render `messages`, dropping the oldest exchanges until the prompt leaves
    `reserve` tokens to answer in. Trims `messages` in place and returns the
    rendered ids plus how many exchanges were dropped.

    A leading system message is never dropped, and neither is the newest turn -
    if that alone does not fit there is nothing left to give up.
    """
    keep = 1 if messages and messages[0].get("role") == "system" else 0
    dropped = 0
    options = dict(render_options or {})
    while True:
        ids = fmt.render(messages, **options)
        if len(ids) + reserve <= n_ctx or len(messages) - keep <= 1:
            return ids, dropped
        del messages[keep]
        dropped += 1
        # a user turn and the reply it drew go together
        while len(messages) - keep > 1 and messages[keep].get("role") == "assistant":
            del messages[keep]


def _compact_text_to_token_budget(tokenizer, text: str, target: int) -> tuple[str, bool]:
    """Keep the useful edges of one prompt section inside a token budget.

    API callers commonly put a whole evolving game snapshot into one system
    message, so dropping old message pairs is not enough.  This bounded,
    deterministic compactor preserves the beginning (identity/instructions)
    and the end (the newest truth/contract sections) and marks the omission.
    """
    text = str(text or "")
    try:
        target = max(1, int(target))
    except (TypeError, ValueError):
        target = 1
    ids = tokenizer.encode(text, add_bos=False)
    if len(ids) <= target:
        return text, False

    marker = "\n[older prompt context compacted]\n"
    marker_ids = tokenizer.encode(marker, add_bos=False)
    if target <= len(marker_ids) + 2:
        # Tiny synthetic contexts (and defensive callers) cannot afford the
        # marker; return the largest prefix that re-encodes inside the budget.
        low, high, best = 0, min(target, len(ids)), ""
        while low <= high:
            middle = (low + high) // 2
            value = tokenizer.decode(ids[:middle])
            if len(tokenizer.encode(value, add_bos=False)) <= target:
                best = value
                low = middle + 1
            else:
                high = middle - 1
        return best, True

    usable = max(2, target - len(marker_ids))
    head = max(1, int(usable * 0.64))
    tail = max(1, usable - head)

    def candidate() -> str:
        return tokenizer.decode(ids[:head]) + marker + tokenizer.decode(ids[-tail:])

    value = candidate()
    for _ in range(32):
        if len(tokenizer.encode(value, add_bos=False)) <= target:
            return value, True
        excess = len(tokenizer.encode(value, add_bos=False)) - target
        if head <= 1 and tail <= 1:
            break
        head = max(1, head - max(1, (excess + 1) // 2))
        tail = max(1, tail - max(1, excess // 2))
        value = candidate()

    # The token boundary can re-tokenize after decoding.  Find the largest
    # prefix that still fits rather than returning another over-budget value.
    low, high, best = 0, min(target, len(ids)), ""
    while low <= high:
        middle = (low + high) // 2
        candidate = tokenizer.decode(ids[:middle])
        if len(tokenizer.encode(candidate, add_bos=False)) <= target:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best, True


def fit_messages_for_request(fmt: "ChatFormat", messages: list[dict], n_ctx: int,
                             reserve: int = REPLY_RESERVE, *,
                             render_options: dict | None = None
                             ) -> tuple[list[dict], list[int], dict]:
    """Fit API chat messages without allowing an avoidable context 400.

    ``fit_to_context`` handles ordinary multi-turn chats.  Game clients often
    send one large system snapshot plus one newest user turn, however, so this
    helper additionally compacts the largest content section until the fully
    rendered chat template leaves room for a reply.
    """
    work = [dict(message) for message in list(messages or [])]
    try:
        n_ctx = max(1, int(n_ctx))
    except (TypeError, ValueError):
        n_ctx = 1
    try:
        reserve = max(1, min(n_ctx - 1 if n_ctx > 1 else 1, int(reserve)))
    except (TypeError, ValueError):
        reserve = min(REPLY_RESERVE, max(1, n_ctx - 1))

    options = dict(render_options or {})
    ids, dropped = fit_to_context(
        fmt, work, n_ctx, reserve=reserve, render_options=options,
    )
    prompt_limit = max(1, n_ctx - reserve)
    compacted = 0
    tok = fmt.model.tok

    for _ in range(max(8, len(work) * 8)):
        if len(ids) <= prompt_limit:
            break
        candidates = []
        newest_index = len(work) - 1
        for index, message in enumerate(work):
            content = str(message.get("content", "") or "")
            content_tokens = len(tok.encode(content, add_bos=False))
            if content_tokens <= 0:
                continue
            role = str(message.get("role", "") or "")
            # Preserve the newest user turn longer than older/system context,
            # but never permit it to make the request unrenderable.
            preserve_bias = 64 if index == newest_index and role == "user" else 0
            candidates.append((content_tokens, index, preserve_bias))
        if not candidates:
            break
        _, index, preserve_bias = max(
            candidates,
            key=lambda row: (row[0] - row[2], row[0], -row[1]),
        )
        current = str(work[index].get("content", "") or "")
        current_tokens = len(tok.encode(current, add_bos=False))
        overflow = max(1, len(ids) - prompt_limit)
        target = max(1, current_tokens - max(1, overflow + 16))
        compacted_text, changed = _compact_text_to_token_budget(tok, current, target)
        if not changed or compacted_text == current:
            # Lower the target for a tokenizer boundary or a very small
            # context where the first candidate still cannot be reduced.
            target = max(1, min(current_tokens - 1, target - 8))
            compacted_text, changed = _compact_text_to_token_budget(tok, current, target)
        if not changed or compacted_text == current:
            break
        work[index]["content"] = compacted_text
        compacted += 1
        ids = fmt.render(work, **options)

    return work, ids, {
        "dropped_messages": int(dropped),
        "compaction_passes": int(compacted),
        "prompt_tokens": len(ids),
        "reply_reserve": int(reserve),
        "fits": bool(len(ids) + reserve <= n_ctx),
    }


def chat_once(model: Model, messages: list[dict], params: SamplerParams,
              n_predict: int = -1, stream=None,
              stop_strings: list[str] | None = None, *,
              json_only: bool = False,
              enable_thinking: bool | None = None,
              reasoning_effort: str | None = None,
              preserve_thinking: bool | None = None,
              check_cancelled=None) -> GenerationResult:
    fmt = ChatFormat(model, detect_format(model.metadata))
    p = _prof.ACTIVE
    t0 = _perf() if p is not None else 0.0
    ids = fmt.render(
        messages,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        preserve_thinking=preserve_thinking,
    )
    if p is not None:
        p.add("prompt_render", _perf() - t0)
        p.set("chat_format", fmt.name)
    return generate(model, ids, params, n_predict, stream, stop_strings,
                    stop_tokens=fmt.stop_tokens(), json_only=json_only,
                    check_cancelled=check_cancelled)


def _read_chat_line(prompt: str = "> ", stdin: TextIO | None = None,
                    stdout: TextIO | None = None) -> str | None:
    """Read one chat line. Return None when the user presses bare Escape."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    if stdin is not sys.stdin or stdout is not sys.stdout or not stdin.isatty():
        stdout.write(prompt)
        stdout.flush()
        line = stdin.readline()
        if line == "":
            raise EOFError
        line = line.rstrip("\r\n")
        return None if "\x1b" in line else line
    if sys.platform == "win32":
        return _read_chat_line_windows(prompt, stdout)
    return _read_chat_line_posix(prompt, stdin, stdout)


def _read_chat_line_windows(prompt: str, stdout: TextIO) -> str | None:
    import msvcrt

    stdout.write(prompt)
    stdout.flush()
    chars: list[str] = []
    while True:
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            msvcrt.getwch()
            continue
        if ch == "\x1b":
            stdout.write("\n")
            stdout.flush()
            return None
        if ch in ("\r", "\n"):
            stdout.write("\n")
            stdout.flush()
            return "".join(chars)
        if ch == "\x03":
            raise KeyboardInterrupt
        if ch == "\x04":
            raise EOFError
        if ch in ("\b", "\x7f"):
            if chars:
                chars.pop()
                stdout.write("\b \b")
                stdout.flush()
            continue
        if ch == "\t" or ch >= " ":
            chars.append(ch)
            stdout.write(ch)
            stdout.flush()


def _read_chat_line_posix(prompt: str, stdin: TextIO,
                          stdout: TextIO) -> str | None:
    import select
    import termios
    import tty

    fd = stdin.fileno()
    old = termios.tcgetattr(fd)
    stdout.write(prompt)
    stdout.flush()
    chars: list[str] = []
    try:
        tty.setcbreak(fd)
        while True:
            ch = stdin.read(1)
            if ch == "\x1b":
                if select.select([stdin], [], [], 0.05)[0]:
                    stdin.read(1)
                    while select.select([stdin], [], [], 0.001)[0]:
                        stdin.read(1)
                    continue
                stdout.write("\n")
                stdout.flush()
                return None
            if ch in ("\r", "\n"):
                stdout.write("\n")
                stdout.flush()
                return "".join(chars)
            if ch == "\x04" and not chars:
                raise EOFError
            if ch == "\x03":
                raise KeyboardInterrupt
            if ch in ("\b", "\x7f"):
                if chars:
                    chars.pop()
                    stdout.write("\b \b")
                    stdout.flush()
                continue
            if ch == "\t" or ch >= " ":
                chars.append(ch)
                stdout.write(ch)
                stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def interactive(model: Model, params: SamplerParams, system: str = "",
                n_predict: int = -1, model_name: str = "",
                model_path: str = "") -> None:
    fmt = ChatFormat(model, detect_format(model.metadata))
    description = model.describe()
    print(f"alpaccaroo chat - {description}", file=sys.stderr)
    print("press Esc or type /exit to return, /clear to reset the conversation\n",
          file=sys.stderr)
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    from .history import start_session
    history = start_session(model_name or description, model_path, system)
    history_disabled = False

    def record_history(method: str, *args, **kwargs) -> None:
        nonlocal history_disabled
        if history_disabled:
            return
        try:
            getattr(history, method)(*args, **kwargs)
        except OSError as e:
            history_disabled = True
            print(f"(chat history disabled: {e})", file=sys.stderr)

    try:
        while True:
            try:
                user = _read_chat_line("> ")
            except (EOFError, KeyboardInterrupt):
                print("", file=sys.stderr)
                return
            if user is None:
                print("(returning to main menu)", file=sys.stderr)
                return
            if user.strip() in ("/exit", "/quit", "/bye"):
                return
            if user.strip() == "/clear":
                messages = messages[:1] if system else []
                model.reset()
                record_history("append_event", "clear")
                print("(cleared)", file=sys.stderr)
                continue
            if not user.strip():
                continue
            messages.append({"role": "user", "content": user})
            record_history("append_message", "user", user)
            ids, dropped = fit_to_context(fmt, messages, model.n_ctx)
            if dropped:
                print(f"(dropped {dropped} earlier turn"
                      f"{'s' if dropped != 1 else ''} to fit the context window)",
                      file=sys.stderr)
            try:
                res = generate(model, ids, params, n_predict,
                               stream=lambda s: print(s, end="", flush=True),
                               stop_tokens=fmt.stop_tokens())
            except RuntimeError as e:
                # the prompt does not fit even on its own - stay in the REPL
                messages.pop()
                print(f"({e}; that message was too long, so it was not sent - "
                      f"/clear resets the conversation)", file=sys.stderr)
                continue
            print()
            if res.stop_reason == "context":
                print("(the context window is full - /clear resets the conversation)",
                      file=sys.stderr)
            print(f"[{res.tokens} tokens, {res.tok_per_sec:.1f} tok/s]", file=sys.stderr)
            messages.append({"role": "assistant", "content": res.text})
            record_history("append_message", "assistant", res.text,
                           tokens=res.tokens, seconds=res.seconds)
    finally:
        record_history("close")
