# Alpaccaroo - talk to a resident alpaccaroo server instead of loading a
# model. MIT License. See LICENSE.
"""Fast reconnect from the CLI to a local `alpaccaroo serve` (Track 6).

Steady decode speed is not the whole of perceived speed. A one-shot
`alpaccaroo run` pays process startup, imports, GGUF open and unpack, JIT
cache-load and prompt setup before the first token - measured at 30-49 s
for a 3B Q4_K_M against roughly 1 s per token on the same machine, so more
than half the wall clock of a short answer is load.

A server already holds that model. This is the client that uses it:
standard library only, the same OpenAI-compatible streaming endpoint the
server already serves, and never mandatory - `--connect` is opt-in and
falls back to loading locally when nothing is listening.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_TIMEOUT = 5.0


def default_url() -> str:
    """Where a local server would be, from the same env the server reads."""
    host = os.environ.get("ALPACCAROO_HOST", "127.0.0.1")
    port = os.environ.get("ALPACCAROO_PORT", "8080")
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"          # the wildcard is a bind address, not a peer
    return f"http://{host}:{port}"


def probe(url: str, timeout: float = DEFAULT_TIMEOUT) -> "dict | None":
    """{"model": name} when a server answers at `url`, else None.

    Deliberately quiet and quick: this runs on the way to a fallback, and a
    machine with nothing listening must not wait on it.
    """
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/health",
                                    timeout=timeout) as r:
            if r.status != 200:
                return None
        with urllib.request.urlopen(f"{url.rstrip('/')}/v1/models",
                                    timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return None
    models = data.get("data") or []
    return {"model": (models[0].get("id") if models else "") or "unknown"}


def stream_chat(url: str, model: str, messages: list, params,
                n_predict: int = -1, timeout: float = 600.0,
                on_text=None) -> dict:
    """POST /v1/chat/completions with stream=true; return the full reply.

    Returns {"text", "tokens", "finish_reason"}. `on_text` receives each
    delta as it arrives, so the caller streams to the terminal exactly as
    the in-process path does.
    """
    body = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "top_k": params.top_k,
        "repeat_penalty": params.repeat_penalty,
    }
    if n_predict >= 0:
        body["max_tokens"] = n_predict
    if params.seed >= 0:
        body["seed"] = params.seed
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")

    text = ""
    chunks = 0
    finish = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if "error" in obj:
                raise RuntimeError(obj["error"].get("message", "server error"))
            for choice in obj.get("choices") or []:
                piece = (choice.get("delta") or {}).get("content") or ""
                if piece:
                    text += piece
                    chunks += 1
                    if on_text is not None:
                        on_text(piece)
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    return {"text": text, "chunks": chunks, "finish_reason": finish}


def run_connected(url: str, info: dict, args, params) -> int:
    """One-shot or interactive chat against a resident server."""
    import time

    model_name = info["model"]
    print(f"connected to {url} ({model_name}) - no model load",
          file=sys.stderr)

    def once(messages) -> None:
        t0 = time.time()
        res = stream_chat(url, model_name, messages, params, args.n_predict,
                          on_text=lambda s: print(s, end="", flush=True))
        dt = time.time() - t0
        print()
        # the server streams text, not token counts, so this reports
        # deltas per second and says so rather than implying tok/s
        print(f"[{res['chunks']} chunks in {dt:.1f}s"
              + (f", {res['finish_reason']}" if res["finish_reason"] else "")
              + "]", file=sys.stderr)

    if args.prompt:
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": " ".join(args.prompt)})
        once(messages)
        return 0

    history: list = []
    if args.system:
        history.append({"role": "system", "content": args.system})
    print("type /exit to leave, /clear to start over", file=sys.stderr)
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return 0
        if line in ("/exit", "/quit"):
            return 0
        if line == "/clear":
            history = [m for m in history if m["role"] == "system"]
            print("conversation cleared", file=sys.stderr)
            continue
        if not line:
            continue
        history.append({"role": "user", "content": line})
        try:
            t0 = time.time()
            res = stream_chat(url, model_name, history, params,
                              args.n_predict,
                              on_text=lambda s: print(s, end="", flush=True))
        except (urllib.error.URLError, OSError, RuntimeError) as e:
            print(f"\nalpaccaroo: server error: {e}", file=sys.stderr)
            history.pop()
            continue
        print()
        print(f"[{res['chunks']} chunks in {time.time() - t0:.1f}s]",
              file=sys.stderr)
        history.append({"role": "assistant", "content": res["text"]})
