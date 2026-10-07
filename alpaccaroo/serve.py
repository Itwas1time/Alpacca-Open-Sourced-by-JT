# Alpaccaroo - OpenAI-compatible HTTP API on the standard library only.
# Endpoints: /health, /v1/models, /v1/chat/completions (incl. streaming),
# a llama.cpp-style /completion, and the Ollama-native surface (/api/chat,
# /api/generate, /api/tags, /api/show, /api/ps, /api/version) so the official
# ollama client works unmodified against this port. MIT License. See LICENSE.
from __future__ import annotations

import inspect
import json
import queue
import select
import socket
import socketserver
import sys
import threading
import time
import uuid
from concurrent.futures import Future
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__, chat
from .model import Model
from .sample import SamplerParams


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def server_bind(self):  # skip socket.getfqdn(), which can stall for seconds
        socketserver.TCPServer.server_bind(self)
        self.server_name = "localhost"
        self.server_port = self.server_address[1]


class _ClientDisconnected(ConnectionAbortedError):
    """The requesting client closed its cancellable inference connection."""


class _GenerationWorker:
    """One persistent inference owner with the HTTP server's daemon lifetime."""

    def __init__(self, initializer, before_request=None):
        self._requests = queue.SimpleQueue()
        self._closed = False
        self._thread = threading.Thread(target=self._work, args=(initializer, before_request),
                                        name="alpaccaroo-generation", daemon=True)
        self._thread.start()

    def _work(self, initializer, before_request):
        error = None
        try:
            initializer()
        except BaseException as exc:
            error = exc
        while (request := self._requests.get()) is not None:
            result, fn, args, kwargs = request
            try:
                if error is not None:
                    raise error
                if before_request is not None:
                    before_request()
                result.set_result(fn(*args, **kwargs))
            except BaseException as exc:
                result.set_exception(exc)
            finally:
                # An idle worker must not retain a failed inference traceback
                # (and its temporary arrays) through its completed Future.
                del request, result, fn, args, kwargs

    def run(self, fn, *args, **kwargs):
        if self._closed:
            raise RuntimeError("generation server is shutting down")
        result = Future()
        self._requests.put((result, fn, args, kwargs))
        return result.result()

    def close(self):
        self._closed = True
        self._requests.put(None)


def _body_value(body: dict, key: str, default):
    value = body.get(key, default)
    return default if value is None else value


def _float_param(body: dict, key: str, default: float) -> float:
    try:
        return float(_body_value(body, key, default))
    except (TypeError, ValueError) as e:
        raise ValueError(f"{key} must be a number") from e


def _int_param(body: dict, keys: tuple[str, ...], default: int) -> int:
    key_used = keys[0]
    for key in keys:
        if key in body and body[key] is not None:
            key_used = key
            value = body[key]
            break
    else:
        value = default
    try:
        return int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{key_used} must be an integer") from e


def _params_from(body: dict, defaults: SamplerParams) -> SamplerParams:
    return SamplerParams(
        temperature=_float_param(body, "temperature", defaults.temperature),
        top_k=_int_param(body, ("top_k",), defaults.top_k),
        top_p=_float_param(body, "top_p", defaults.top_p),
        repeat_penalty=_float_param(body, "repeat_penalty", defaults.repeat_penalty),
        seed=_int_param(body, ("seed",), defaults.seed),
    )


def _finish_reason(res) -> str:
    """OpenAI semantics: "length" when the answer was cut short, else "stop".

    Both budgets count as "length": the n_predict one and the context window.
    Inferring this from the token count alone gets both edges wrong - a stop
    string landing on the budget-th token is a clean stop, and running out of
    context is a truncation no matter how few tokens were asked for.
    """
    return "length" if res.stop_reason in ("length", "context") else "stop"


def _messages_from(body: dict) -> list[dict]:
    """Validate before rendering: ChatFormat indexes role/content directly, and
    a KeyError escaping the handler drops the socket with no response."""
    messages = body.get("messages") or []
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages required")
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            raise ValueError(f"messages[{i}] must be an object")
        for key in ("role", "content"):
            if not isinstance(m.get(key), str):
                raise ValueError(f"messages[{i}].{key} must be a string")
    return messages


def _stop_from(body: dict) -> list[str]:
    stop = body.get("stop") or []
    if isinstance(stop, str):
        return [stop]
    if isinstance(stop, list):
        return [str(s) for s in stop if s is not None]
    return []


def _chat_reply_reserve(n_predict: int) -> int:
    """Leave a bounded answer budget when fitting an API prompt."""
    try:
        requested = int(n_predict)
    except (TypeError, ValueError):
        requested = 0
    if requested <= 0:
        return 256
    return max(128, min(512, requested))


def _chat_render_options(body: dict) -> dict:
    raw = body.get("chat_template_kwargs")
    if raw is None:
        template_options = {}
    elif isinstance(raw, dict):
        template_options = raw
    else:
        raise ValueError("chat_template_kwargs must be an object")
    result = {}
    for key in ("enable_thinking", "preserve_thinking"):
        value = template_options.get(key, _body_value(body, key, None))
        if value is not None:
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be a boolean")
            result[key] = value
    effort = body.get("reasoning_effort", template_options.get("reasoning_effort"))
    if effort is not None:
        if not isinstance(effort, str):
            raise ValueError("reasoning_effort must be a string")
        result["reasoning_effort"] = effort
    return result


def _fit_api_messages(
    model: Model, messages: list[dict], n_predict: int,
    render_options: dict | None = None,
) -> tuple[list[dict], dict]:
    """Fit a chat request to the resident model before rendering it."""
    fmt = chat.ChatFormat(model, chat.detect_format(model.metadata))
    fitted, _ids, info = chat.fit_messages_for_request(
        fmt,
        messages,
        model.n_ctx,
        reserve=_chat_reply_reserve(n_predict),
        render_options=render_options,
    )
    return fitted, info


# -- Ollama-native API helpers ----------------------------------------------

def _accepts_json_only(fn) -> bool:
    """The json_only contract (emitted text is a prefix of valid JSON and
    generation stops when the top-level value closes) may not be in chat.py
    yet. Detect it once at import: retrying a rejected keyword after pieces
    already streamed would replay the whole answer, so format=json has to
    degrade to plain generation up front, never mid-stream."""
    try:
        return "json_only" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


_CHAT_JSON_ONLY = _accepts_json_only(chat.chat_once)
_GEN_JSON_ONLY = _accepts_json_only(chat.generate)


def _iso_now() -> str:
    # RFC3339 UTC. Ollama emits fractional timestamps and the client's
    # pydantic models parse created_at/modified_at into datetimes, which
    # accept this form with or without the fraction.
    t = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + \
        f".{int(t % 1 * 1e6):06d}Z"


def _same_model(a: str, b: str) -> bool:
    """Ollama treats "name" and "name:latest" as the same model. Slashes are
    unified because a file ref's display() is Path-normalised on Windows -
    the client that launched `serve C:/x/m.gguf` must not 404 on that same
    string; registry refs never contain a backslash, so they are unaffected."""
    def base(s: str) -> str:
        s = s.strip().replace("\\", "/")
        return s[:-len(":latest")] if s.endswith(":latest") else s
    return base(a) == base(b)


def _wants_json(body: dict) -> bool:
    # format="json" asks for constrained output; a dict is a JSON schema,
    # honoured here as plain "json" - the output is valid JSON but the
    # schema itself is not enforced (Ollama-lite)
    fmt = body.get("format")
    return fmt == "json" or isinstance(fmt, dict)


def _ollama_options(body: dict, defaults: SamplerParams) -> tuple[SamplerParams, int, list[str]]:
    """Map Ollama's options block onto the sampler. num_ctx is accepted and
    ignored: the window was fixed when the model loaded, so a larger ask just
    proceeds at model.n_ctx rather than erroring. keep_alive is ignored the
    same way - the one model stays resident for the life of the process."""
    options = body.get("options")
    if not isinstance(options, dict):
        options = {}
    params = SamplerParams(
        temperature=_float_param(options, "temperature", defaults.temperature),
        top_k=_int_param(options, ("top_k",), defaults.top_k),
        top_p=_float_param(options, "top_p", defaults.top_p),
        repeat_penalty=_float_param(options, "repeat_penalty", defaults.repeat_penalty),
        repeat_last_n=_int_param(options, ("repeat_last_n",), defaults.repeat_last_n),
        seed=_int_param(options, ("seed",), defaults.seed),
    )
    # Ollama's num_predict default is -1: generate until EOG or the window
    # fills, which is exactly what generate() does with a non-positive budget
    n_predict = _int_param(options, ("num_predict",), -1)
    return params, n_predict, _stop_from(options)


def _param_count(model: Model) -> int:
    """Mirror of Model.describe()'s arithmetic (embeddings, norms, attention,
    FFN) so /api/tags and /api/show report the same size the CLI prints."""
    hp = model.hp
    params = hp.n_vocab * hp.n_embd
    if model.output is not model.tok_embd:
        params += hp.n_vocab * hp.n_embd  # untied output projection
    params += hp.n_embd  # output_norm
    for _ in range(hp.n_layer):
        params += 2 * hp.n_embd  # norms
        if hp.arch == "gemma3":
            params += 2 * hp.head_dim  # q_norm/k_norm are shared per head
            params += 2 * hp.n_embd    # post-attention/post-ffw norms
        params += hp.n_embd * hp.n_head * hp.head_dim * 2  # wq, wo
        params += hp.n_embd * hp.n_kv * hp.head_dim * 2    # wk, wv
        params += 3 * hp.n_embd * hp.n_ff
    return params


def _param_size_label(n: int) -> str:
    return f"{n / 1e9:.1f}B" if n >= 1e9 else f"{n / 1e6:.0f}M"


def _served_details(model: Model) -> dict:
    """Ollama's details block, filled from the loaded model's real state."""
    q = model.weight_storage.get("quantized") or {}
    # the dominant stored type stands in for llama.cpp's file-level label;
    # an all-dense load (an F16/F32 GGUF) has no meaningful level to report
    level = max(q, key=q.get) if q else "unknown"
    return {"format": "gguf", "family": model.hp.arch,
            "families": [model.hp.arch],
            "parameter_size": _param_size_label(_param_count(model)),
            "quantization_level": level}


def _model_info(model: Model) -> dict:
    """Flat metadata map for /api/show: scalars only. Long strings (the chat
    template) and arrays (the tokenizer vocab) are dropped - the template has
    its own response field and the vocab would be megabytes of JSON."""
    info = {}
    for k, v in model.metadata.items():
        if isinstance(v, (bool, int, float)) or (isinstance(v, str) and len(v) <= 200):
            info[k] = v
    hp = model.hp
    # guaranteed present even if the loader stripped the metadata
    info.setdefault("general.architecture", hp.arch)
    info.setdefault(f"{hp.arch}.context_length", hp.n_ctx_train)
    return info


def serve(model: Model, model_name: str, host: str = "127.0.0.1", port: int = 8080,
          defaults: SamplerParams | None = None, ready_callback=None) -> None:
    defaults = defaults or SamplerParams()
    lock = threading.Lock()  # one generation at a time

    def prepare_generation_thread():
        # Each old connection thread started with Numba's default mask. Keep
        # that request-entry behavior on this persistent owner: kernel dispatch
        # can narrow the mask later, and its existing bookkeeping is unchanged.
        from . import kernels
        if kernels.available():
            import numba
            numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)

    # HTTP connection threads keep the health/API surface responsive, but all
    # large prefill/decode allocations have one persistent thread owner. A
    # lock alone serialized work on a succession of connection threads whose
    # CPU allocator arenas retained their freed temporary buffers.
    generation = _GenerationWorker(lambda: None, before_request=prepare_generation_thread)

    def run_generation(fn, *args, **kwargs):
        return generation.run(fn, *args, **kwargs)

    class Handler(BaseHTTPRequestHandler):
        def check_cancelled(self):
            # An ordinary HTTP client may half-close its write side and still
            # expect a response. The client explicitly opts in when it will close
            # the entire connection on deadline or preemption.
            if self.headers.get("X-Alpaccaroo-Cancel-On-Disconnect") != "1":
                return
            try:
                readable, _, _ = select.select([self.connection], [], [], 0)
                closed = readable and not self.connection.recv(1, socket.MSG_PEEK)
            except OSError:
                closed = True
            if closed:
                raise _ClientDisconnected("requesting client disconnected")

        @contextmanager
        def generation_slot(self):
            while not lock.acquire(timeout=.05):
                self.check_cancelled()
            try:
                self.check_cancelled()
                yield
            finally:
                lock.release()

        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            print(f"[serve] {self.address_string()} {fmt % args}", file=sys.stderr)

        def handle_one_request(self):
            """Treat a client closing an idle keep-alive socket as normal.

            A caller watchdog and short-lived health probes can close an HTTP
            connection after receiving a response.  BaseHTTPRequestHandler
            otherwise prints a full traceback while reading the next request,
            even though no generation failed and the server remains healthy.
            Route/generation exceptions are still handled by ``do_POST`` and
            are intentionally not swallowed here.
            """
            try:
                return super().handle_one_request()
            except (ConnectionResetError, ConnectionAbortedError):
                self.close_connection = True
                self.log_message("client closed idle keep-alive connection")

        # -- helpers -----------------------------------------------------

        def send_json(self, obj, status=200):
            data = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def read_body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if not n:
                return {}
            try:
                return json.loads(self.rfile.read(n).decode("utf-8"))
            except json.JSONDecodeError:
                return {}

        def log_generation(self, path: str, result, total_ns: int) -> None:
            """Record model timing without exposing prompt or response text.

            A caller's correspondence watchdog observes the client-side request
            lifecycle, while this server knows the split between prompt
            evaluation and decode.  Keeping both measurements lets future
            playtests distinguish a long prefill from a slow/blocked handoff.
            """
            try:
                total_ms = max(0.0, float(total_ns or 0) / 1_000_000.0)
                decode_ms = max(0.0, float(getattr(result, "seconds", 0.0) or 0.0) * 1000.0)
                prompt_ms = max(0.0, total_ms - decode_ms)
                self.log_message(
                    "generation path=%s prompt_tokens=%s completion_tokens=%s "
                    "stop=%s total_ms=%.1f prompt_ms=%.1f decode_ms=%.1f",
                    path,
                    int(getattr(result, "prompt_tokens", 0) or 0),
                    int(getattr(result, "tokens", 0) or 0),
                    str(getattr(result, "stop_reason", "") or ""),
                    total_ms,
                    prompt_ms,
                    decode_ms,
                )
            except Exception:
                # Timing is diagnostic only; never change an otherwise valid
                # generation response because logging failed.
                pass

        # -- routes ------------------------------------------------------

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/health":
                return self.send_json({"status": "ok"})
            if path == "/v1/models":
                return self.send_json({"object": "list", "data": [
                    {"id": model_name, "object": "model", "owned_by": "alpaccaroo"}]})
            if path == "/api/tags":
                return self.api_tags()
            if path == "/api/version":
                return self.send_json({"version": __version__})
            if path == "/api/ps":
                # one resident model that never unloads, hence the far-future
                # expiry; sizes are reported 0 rather than guessed
                return self.send_json({"models": [{
                    "name": model_name, "model": model_name, "size": 0,
                    "size_vram": 0, "expires_at": "2099-01-01T00:00:00Z",
                    "details": _served_details(model)}]})
            self.send_json({"error": "not found"}, 404)

        def do_POST(self):
            path = self.path.split("?")[0]
            try:
                # inside the try: a bad Content-Length or a body that is not
                # UTF-8 both raise ValueError, and outside it they would take
                # the socket down without any HTTP response at all
                body = self.read_body()
                if path == "/v1/chat/completions":
                    return self.chat_completions(body)
                if path == "/completion":
                    return self.completion(body)
                if path == "/api/chat":
                    return self.api_chat(body)
                if path == "/api/generate":
                    return self.api_generate(body)
                if path == "/api/show":
                    return self.api_show(body)
                self.send_json({"error": "not found"}, 404)
            except (_ClientDisconnected, BrokenPipeError, ConnectionResetError):
                self.log_message("request cancelled: client disconnected")
            except (TypeError, ValueError) as e:
                self.send_json({"error": str(e)}, 400)
            except RuntimeError as e:
                # Model.prefill/forward raise this when the prompt does not fit
                # the context window - ordinary input, not a server fault
                self.send_json({"error": str(e)}, 400)

        def completion(self, body: dict):
            prompt = str(body.get("prompt", ""))
            params = _params_from(body, defaults)
            n_predict = _int_param(body, ("n_predict", "max_tokens"), 256)
            stop = _stop_from(body)
            t0 = time.perf_counter_ns()
            with self.generation_slot():
                ids = model.tok.encode(prompt)
                res = run_generation(chat.generate, model, ids, params, n_predict, stop_strings=stop,
                                     check_cancelled=self.check_cancelled)
            self.log_generation("/completion", res, time.perf_counter_ns() - t0)
            self.send_json({
                "content": res.text,
                "tokens_predicted": res.tokens,
                "timings": {"predicted_per_second": res.tok_per_sec},
                "model": model_name,
            })

        def chat_completions(self, body: dict):
            messages = _messages_from(body)
            params = _params_from(body, defaults)
            n_predict = _int_param(body, ("max_tokens", "max_completion_tokens"), 512)
            stop = _stop_from(body)
            render_options = _chat_render_options(body)
            messages, fit_info = _fit_api_messages(
                model, messages, n_predict, render_options,
            )
            if fit_info.get("dropped_messages") or fit_info.get("compaction_passes"):
                self.log_message(
                    "chat prompt fitted: dropped=%s compacted=%s prompt_tokens=%s reserve=%s",
                    fit_info.get("dropped_messages", 0),
                    fit_info.get("compaction_passes", 0),
                    fit_info.get("prompt_tokens", 0),
                    fit_info.get("reply_reserve", 0),
                )
            rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            created = int(time.time())
            t0 = time.perf_counter_ns()

            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                def chunk(obj):
                    payload = b"data: " + json.dumps(obj).encode("utf-8") + b"\n\n"
                    self.wfile.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")

                def piece(delta, finish=None):
                    chunk({"id": rid, "object": "chat.completion.chunk",
                           "created": created, "model": model_name,
                           "choices": [{"index": 0, "delta": delta,
                                        "finish_reason": finish}]})

                piece({"role": "assistant"})
                try:
                    try:
                        with self.generation_slot():
                            res = run_generation(chat.chat_once, model, messages, params, n_predict,
                                                 stream=lambda s: piece({"content": s}),
                                                 stop_strings=stop,
                                                 check_cancelled=self.check_cancelled,
                                                 **render_options)
                        self.log_generation("/v1/chat/completions", res,
                                            time.perf_counter_ns() - t0)
                        piece({}, finish=_finish_reason(res))
                    except Exception as e:
                        # the 200 and the first delta are already on the wire, so
                        # any failure has to close the stream rather than drop the
                        # socket and leave the client on an unterminated body
                        self.log_message("stream failed: %r", e)
                        piece({"content": ""}, finish="length")
                        chunk({"error": {"message": str(e),
                                         "type": "invalid_request_error"}})
                finally:
                    tail = b"data: [DONE]\n\n"
                    self.wfile.write(f"{len(tail):x}\r\n".encode() + tail + b"\r\n")
                    self.wfile.write(b"0\r\n\r\n")
                return

            with self.generation_slot():
                res = run_generation(chat.chat_once,
                    model, messages, params, n_predict, stop_strings=stop,
                    check_cancelled=self.check_cancelled,
                    **render_options,
                )
            self.log_generation("/v1/chat/completions", res,
                                time.perf_counter_ns() - t0)
            self.send_json({
                "id": rid, "object": "chat.completion", "created": created,
                "model": model_name,
                "choices": [{"index": 0,
                             "finish_reason": _finish_reason(res),
                             "message": {"role": "assistant", "content": res.text}}],
                # counted by generate(), not inferred from n_past: the last
                # sampled token is never forwarded into the cache, so n_past
                # under-reports the prompt by one on every truncated answer
                "usage": {"prompt_tokens": res.prompt_tokens,
                          "completion_tokens": res.tokens,
                          "total_tokens": res.prompt_tokens + res.tokens},
            })

        # -- Ollama-native API (/api/*) ----------------------------------

        def check_served_model(self, body: dict) -> bool:
            """404 in Ollama's error shape when the request names a model this
            process is not serving. An absent name means the served model, and
            "name" matches "name:latest" - the client normalises both ways."""
            requested = str(body.get("model") or "").strip()
            if not requested or _same_model(requested, model_name):
                return True
            self.send_json({"error": f"model '{requested}' not found"}, 404)
            return False

        def ollama_usage(self, res, total_ns: int) -> dict:
            """Final-response bookkeeping fields. All durations nanoseconds.
            Counts are exact (counted by generate()); total is this handler's
            wall clock, eval is generate()'s decode clock (its timer starts
            after prefill), so the prompt_eval remainder genuinely is prefill
            plus render/lock overhead - nothing here is fabricated."""
            eval_ns = int(res.seconds * 1e9)
            return {
                "done": True,
                "done_reason": _finish_reason(res),
                "total_duration": total_ns,
                "load_duration": int(getattr(model, "load_seconds", 0.0) * 1e9),
                "prompt_eval_count": res.prompt_tokens,
                "prompt_eval_duration": max(0, total_ns - eval_ns),
                "eval_count": res.tokens,
                "eval_duration": eval_ns,
            }

        def start_ndjson(self):
            # Ollama streams newline-delimited JSON objects, not SSE
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

        def ndline(self, obj):
            payload = json.dumps(obj).encode("utf-8") + b"\n"
            self.wfile.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")

        def probe_reply(self, body: dict, extra: dict | None = None) -> None:
            """Answer a model load/unload probe. Ollama's CLI and clients send
            /api/chat with no messages and /api/generate with no prompt to
            (pre)load a model, and keep_alive 0 to unload it; both expect an
            immediate done response, never a generation - without this special
            case an `ollama stop` triggers a full unprompted decode from BOS
            while holding the generation lock."""
            keep = _body_value(body, "keep_alive", None)
            unload = keep in (0, "0", "0s", "0m")
            out = {"model": model_name, "created_at": _iso_now(),
                   "done": True,
                   "done_reason": "unload" if unload else "load"}
            if extra:
                out.update(extra)
            self.send_json(out)

        def api_chat(self, body: dict):
            if not self.check_served_model(body):
                return
            if not body.get("messages"):
                return self.probe_reply(
                    body, {"message": {"role": "assistant", "content": ""}})
            messages = _messages_from(body)
            params, n_predict, stop = _ollama_options(body, defaults)
            render_options = _chat_render_options(body)
            messages, fit_info = _fit_api_messages(
                model, messages, n_predict, render_options,
            )
            if fit_info.get("dropped_messages") or fit_info.get("compaction_passes"):
                self.log_message(
                    "chat prompt fitted: dropped=%s compacted=%s prompt_tokens=%s reserve=%s",
                    fit_info.get("dropped_messages", 0),
                    fit_info.get("compaction_passes", 0),
                    fit_info.get("prompt_tokens", 0),
                    fit_info.get("reply_reserve", 0),
                )
            kwargs = dict(render_options)
            if _wants_json(body) and _CHAT_JSON_ONLY:
                kwargs["json_only"] = True
            t0 = time.perf_counter_ns()

            def line_for(content: str, extra: dict | None = None) -> dict:
                out = {"model": model_name, "created_at": _iso_now(),
                       "message": {"role": "assistant", "content": content},
                       "done": False}
                if extra:
                    out.update(extra)
                return out

            if _body_value(body, "stream", True):  # ollama streams by default
                self.start_ndjson()
                try:
                    try:
                        with self.generation_slot():
                            res = run_generation(chat.chat_once,
                                model, messages, params, n_predict,
                                stream=lambda s: self.ndline(line_for(s)),
                                stop_strings=stop, check_cancelled=self.check_cancelled, **kwargs)
                        self.log_generation("/api/chat", res,
                                            time.perf_counter_ns() - t0)
                        self.ndline(line_for("", self.ollama_usage(
                            res, time.perf_counter_ns() - t0)))
                    except Exception as e:
                        # the 200 and earlier pieces may be on the wire; an
                        # error line is Ollama's mid-stream failure shape and
                        # the official client raises ResponseError on it
                        self.log_message("stream failed: %r", e)
                        self.ndline({"error": str(e)})
                finally:
                    self.wfile.write(b"0\r\n\r\n")
                return

            with self.generation_slot():
                res = run_generation(chat.chat_once, model, messages, params, n_predict,
                                     stop_strings=stop, check_cancelled=self.check_cancelled, **kwargs)
            self.log_generation("/api/chat", res,
                                time.perf_counter_ns() - t0)
            self.send_json(line_for(res.text, self.ollama_usage(
                res, time.perf_counter_ns() - t0)))

        def api_generate(self, body: dict):
            if not self.check_served_model(body):
                return
            prompt = str(_body_value(body, "prompt", ""))
            if not prompt:
                return self.probe_reply(body, {"response": "", "context": []})
            params, n_predict, stop = _ollama_options(body, defaults)
            render_options = _chat_render_options(body)
            kwargs = {"json_only": True} if _wants_json(body) and _GEN_JSON_ONLY else {}
            # Ollama renders /api/generate prompts through the model's chat
            # template unless raw is set; a bare encode hands an instruct
            # model an untemplated string and it produces continuations, not
            # answers. The template's turn-end tokens come along as stop
            # tokens, exactly as chat_once wires them.
            stop_tokens = None
            if _body_value(body, "raw", False):
                def encode() -> list[int]:
                    return model.tok.encode(prompt)
            else:
                fmt = chat.ChatFormat(model, chat.detect_format(model.metadata))
                msgs = [{"role": "user", "content": prompt}]
                system = str(_body_value(body, "system", "") or "")
                if system:
                    msgs.insert(0, {"role": "system", "content": system})
                stop_tokens = fmt.stop_tokens()

                def encode() -> list[int]:
                    return fmt.render(msgs, **render_options)
            t0 = time.perf_counter_ns()

            def line_for(piece: str, extra: dict | None = None) -> dict:
                out = {"model": model_name, "created_at": _iso_now(),
                       "response": piece, "done": False}
                if extra:
                    out.update(extra)
                return out

            if _body_value(body, "stream", True):
                self.start_ndjson()
                try:
                    try:
                        with self.generation_slot():
                            res = run_generation(chat.generate,
                                model, encode(), params, n_predict,
                                stream=lambda s: self.ndline(line_for(s)),
                                stop_strings=stop, stop_tokens=stop_tokens,
                                check_cancelled=self.check_cancelled,
                                **kwargs)
                        self.log_generation("/api/generate", res,
                                            time.perf_counter_ns() - t0)
                        final = line_for("", self.ollama_usage(
                            res, time.perf_counter_ns() - t0))
                        final["context"] = []  # no per-request state is kept
                        self.ndline(final)
                    except Exception as e:
                        self.log_message("stream failed: %r", e)
                        self.ndline({"error": str(e)})
                finally:
                    self.wfile.write(b"0\r\n\r\n")
                return

            with self.generation_slot():
                res = run_generation(chat.generate, model, encode(), params, n_predict,
                                    stop_strings=stop, stop_tokens=stop_tokens,
                                    check_cancelled=self.check_cancelled,
                                    **kwargs)
            self.log_generation("/api/generate", res,
                                time.perf_counter_ns() - t0)
            out = line_for(res.text, self.ollama_usage(
                res, time.perf_counter_ns() - t0))
            out["context"] = []
            self.send_json(out)

        def api_tags(self):
            """Installed models from the store, with the served model always
            present even when the store is unreadable or empty."""
            entries = []
            try:
                from .store import list_models
                for m in list_models():
                    entries.append({
                        "name": m["name"], "model": m["name"],
                        "modified_at": m.get("pulled_at") or _iso_now(),
                        "size": int(m.get("size", 0) or 0), "digest": "",
                        # manifests do not record arch/quant; only the loaded
                        # model has real values, filled in below
                        "details": {"format": "gguf", "family": "",
                                    "families": [], "parameter_size": "",
                                    "quantization_level": ""}})
            except Exception as e:  # a broken store must not take /api/tags down
                self.log_message("store listing failed: %r", e)
                entries = []
            served = next((e for e in entries
                           if _same_model(e["name"], model_name)), None)
            if served is None:
                served = {"name": model_name, "model": model_name,
                          "modified_at": _iso_now(), "size": 0, "digest": ""}
                entries.insert(0, served)
            served["details"] = _served_details(model)
            self.send_json({"models": entries})

        def api_show(self, body: dict):
            if not self.check_served_model(body):
                return
            self.send_json({
                "modelfile": "", "parameters": "",
                "template": str(model.metadata.get("tokenizer.chat_template", "")),
                "details": _served_details(model),
                "model_info": _model_info(model),
            })

    httpd = _Server((host, port), Handler)
    actual_port = httpd.server_address[1]
    print(f"alpaccaroo serving {model_name} on http://{host}:{actual_port} "
          f"(OpenAI-compatible: POST /v1/chat/completions; "
          f"Ollama-native: POST /api/chat)", file=sys.stderr)
    if ready_callback:
        ready_callback(actual_port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", file=sys.stderr)
    finally:
        generation.close()
