# Alpaccaroo - lightweight JSON chat history.
# MIT License. See LICENSE.
from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .store import alpaccaroo_home, list_models, now_iso8601


def history_root() -> Path:
    return alpaccaroo_home() / "history"


def _history_id() -> str:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _short_title(text: str, limit: int = 72) -> str:
    title = " ".join(text.split())
    if len(title) <= limit:
        return title
    return title[:limit - 1].rstrip() + "..."


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def _load_chat(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not data.get("id"):
        return None
    data["_path"] = path
    return data


def _text_field(data: dict, name: str) -> str:
    value = data.get(name, "")
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def message_dicts(data: dict) -> list[dict]:
    messages = data.get("messages", [])
    if not isinstance(messages, list):
        return []
    return [m for m in messages if isinstance(m, dict)]


def _valid_token_count(value) -> bool:
    return type(value) is int and value >= 0


def _valid_seconds(value) -> bool:
    return (type(value) in (int, float) and
            math.isfinite(float(value)) and float(value) > 0.0)


@dataclass
class ChatHistorySession:
    model: str
    model_path: str = ""
    system: str = ""
    id: str = field(default_factory=_history_id)
    data: dict = field(init=False)
    path: Path = field(init=False)
    _saved: bool = False

    def __post_init__(self) -> None:
        started = now_iso8601()
        self.path = history_root() / f"{self.id}.json"
        self.data = {
            "id": self.id,
            "started_at": started,
            "updated_at": started,
            "ended_at": "",
            "model": self.model,
            "model_path": self.model_path,
            "title": "",
            "messages": [],
        }
        if self.system:
            self.data["messages"].append({
                "role": "system",
                "content": self.system,
                "created_at": started,
            })

    def _has_loggable_content(self) -> bool:
        return any(m.get("role") in ("user", "assistant")
                   for m in self.data.get("messages", []))

    def save(self) -> None:
        if not self._has_loggable_content():
            return
        if self._saved and not self.path.exists():
            return
        _atomic_write_json(self.path, self.data)
        self._saved = True

    def append_message(self, role: str, content: str, *,
                       tokens: int | None = None,
                       seconds: float | None = None) -> None:
        now = now_iso8601()
        msg = {"role": role, "content": content, "created_at": now}
        if tokens is not None:
            msg["tokens"] = tokens
        if seconds is not None:
            msg["seconds"] = round(seconds, 6)
        self.data["messages"].append(msg)
        if role == "user" and not self.data.get("title"):
            self.data["title"] = _short_title(content)
        self.data["updated_at"] = now
        self.save()

    def append_event(self, event: str) -> None:
        now = now_iso8601()
        self.data["messages"].append({
            "role": "event",
            "event": event,
            "created_at": now,
        })
        self.data["updated_at"] = now
        self.save()

    def close(self) -> None:
        if not self._has_loggable_content():
            return
        self.data["ended_at"] = now_iso8601()
        self.data["updated_at"] = self.data["ended_at"]
        self.save()


def start_session(model: str, model_path: str = "",
                  system: str = "") -> ChatHistorySession:
    return ChatHistorySession(model=model, model_path=model_path, system=system)


def list_chats() -> list[dict]:
    root = history_root()
    if not root.exists():
        return []
    out = []
    for path in sorted(root.glob("*.json")):
        data = _load_chat(path)
        if data is None:
            continue
        messages = message_dicts(data)
        turns = sum(1 for m in messages if m.get("role") == "user")
        out.append({
            "id": _text_field(data, "id"),
            "started_at": _text_field(data, "started_at"),
            "updated_at": _text_field(data, "updated_at"),
            "ended_at": _text_field(data, "ended_at"),
            "model": _text_field(data, "model"),
            "title": _text_field(data, "title"),
            "turns": turns,
            "path": path,
        })
    out.sort(key=lambda c: (c["started_at"], c["id"]), reverse=True)
    return out


def resolve_chat_entry(selector: str, chats: list[dict] | None = None) -> dict:
    sel = selector.strip()
    if not sel:
        raise ValueError("empty chat id")
    if chats is None:
        chats = list_chats()
    if sel.isdigit():
        idx = int(sel)
        if 1 <= idx <= len(chats):
            return chats[idx - 1]
    matches = [c for c in chats if c["id"] == sel or c["id"].startswith(sel)]
    if not matches:
        raise ValueError(f"chat not found: {selector}")
    if len(matches) > 1:
        raise ValueError(f"chat id is ambiguous: {selector}")
    return matches[0]


def resolve_chat(selector: str) -> Path:
    return resolve_chat_entry(selector)["path"]


def read_chat(selector: str) -> dict:
    path = resolve_chat(selector)
    data = _load_chat(path)
    if data is None:
        raise ValueError(f"could not read chat: {selector}")
    return data


def delete_chat(selector: str) -> dict:
    data = dict(resolve_chat_entry(selector))
    path = data["path"]
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    data["_path"] = path
    return data


def clear_history() -> int:
    root = history_root()
    if not root.exists():
        return 0
    count = 0
    for path in root.glob("*.json"):
        try:
            path.unlink()
            count += 1
        except FileNotFoundError:
            pass
    for path in root.glob("*.json.tmp"):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    return count


def model_stats() -> list[dict]:
    rows: dict[str, dict] = {}

    def row_for(model: str) -> dict:
        key = model or "(unknown)"
        if key not in rows:
            rows[key] = {
                "model": key,
                "installed": False,
                "chats": 0,
                "responses": 0,
                "tokens": 0,
                "seconds": 0.0,
                "tok_per_sec": 0.0,
            }
        return rows[key]

    for model in list_models():
        row_for(model["name"])["installed"] = True

    for chat in list_chats():
        data = _load_chat(chat["path"])
        if data is None:
            continue
        row = row_for(_text_field(data, "model"))
        row["chats"] += 1
        for msg in message_dicts(data):
            if msg.get("role") != "assistant":
                continue
            tokens = msg.get("tokens")
            seconds = msg.get("seconds")
            if not _valid_token_count(tokens):
                continue
            if not _valid_seconds(seconds):
                continue
            row["responses"] += 1
            row["tokens"] += tokens
            row["seconds"] += float(seconds)

    for row in rows.values():
        if row["seconds"] > 0:
            row["tok_per_sec"] = row["tokens"] / row["seconds"]
    out = list(rows.values())
    out.sort(key=lambda r: (not r["installed"], r["model"].lower()))
    return out
