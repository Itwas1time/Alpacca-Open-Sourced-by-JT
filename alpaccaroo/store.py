# Alpaccaroo - local model store (~/.alpaccaroo/models) and model references.
# MIT License. See LICENSE.
from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import tempfile
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path


def alpaccaroo_home() -> Path:
    env = os.environ.get("ALPACCAROO_HOME")
    if env:
        return Path(env)
    return Path.home() / ".alpaccaroo"


def models_root() -> Path:
    return alpaccaroo_home() / "models"


# Homes written by a pre-rebrand install. The alpacca -> alpaccaroo rename
# moved models_root(), which makes an existing user's pulled models
# invisible: `list` shows nothing and `run` silently re-downloads gigabytes.
# These are consulted READ-ONLY - `pull` always writes to the current home -
# so an upgrade keeps working without moving anyone's files behind them.
LEGACY_HOME_NAMES = (".alpacca",)


def legacy_models_roots() -> list[Path]:
    if os.environ.get("ALPACCAROO_HOME"):
        return []        # an explicit home means exactly that home
    home = Path.home()
    roots = []
    for name in LEGACY_HOME_NAMES:
        root = home / name / "models"
        if root != models_root() and root.is_dir():
            roots.append(root)
    return roots


def _nicknames_file() -> Path:
    return alpaccaroo_home() / "model-nicknames.json"


def legacy_nickname_files() -> list[Path]:
    """Return pre-rebrand nickname files that may still contain aliases.

    The model store already reads pre-rebrand model directories, but the
    nickname map is data too.  Ignoring ``~/.alpacca/model-nicknames.json``
    makes an installed model such as ``qwenmed3b`` look like an unknown
    Ollama name after the package rename.  Legacy nickname files are read
    only; writes always target the current home through ``_write_nicknames``.
    An explicit ``ALPACCAROO_HOME`` remains an isolated store by design.
    """
    if os.environ.get("ALPACCAROO_HOME"):
        return []
    home = Path.home()
    current = _nicknames_file()
    paths = []
    for name in LEGACY_HOME_NAMES:
        path = home / name / "model-nicknames.json"
        if path != current and path.is_file():
            paths.append(path)
    return paths


def _sanitize(part: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._+-]", "_", part)
    return out if out not in ("", ".", "..") else "_"


@dataclass
class ModelRef:
    """Parsed model reference.

    "llama3.2:1b"           -> ollama registry, library/llama3.2:1b
    "ollama:user/name:tag"  -> ollama registry, user namespace
    "hf:org/repo:Q4_K_M"    -> Hugging Face repo + quant/file selector
    "org/repo"              -> Hugging Face
    "./model.gguf"          -> local file
    """
    source: str   # "ollama" | "hf" | "file"
    ns: str = ""
    name: str = ""
    tag: str = ""
    path: Path | None = None

    def display(self) -> str:
        if self.source == "file":
            return str(self.path)
        if self.source == "hf":
            s = f"hf:{self.ns}/{self.name}"
            return f"{s}:{self.tag}" if self.tag else s
        # must re-parse to the same ref: a bare "ns/name" is read as Hugging
        # Face, so a non-library namespace has to keep its disambiguator
        s = self.name if self.ns == "library" else f"ollama:{self.ns}/{self.name}"
        return f"{s}:{self.tag}" if self.tag != "latest" else s

    def store_dir(self, root: "Path | None" = None) -> Path:
        root = models_root() if root is None else root
        if self.source == "hf":
            return root / "hf" / _sanitize(self.ns) / _sanitize(self.name) / \
                   _sanitize(self.tag or "default")
        if self.source == "ollama":
            return root / "ollama" / _sanitize(self.ns) / _sanitize(self.name) / \
                   _sanitize(self.tag)
        raise ValueError("file references have no store directory")


def parse_model_ref(raw: str) -> ModelRef:
    s = raw.strip()
    if not s:
        raise ValueError("empty model name")

    looks_path = (s.startswith(("/", "./", "../", "~/")) or
                  (os.name == "nt" and (re.match(r"^[A-Za-z]:", s) or
                                        s.startswith((".\\", "..\\", "\\\\")))) or
                  (s.lower().endswith(".gguf") and Path(s).expanduser().exists()))
    if looks_path:
        return ModelRef(source="file", path=Path(s).expanduser())

    forced_hf = False
    for p in ("hf:", "hf.co/", "huggingface.co/", "https://huggingface.co/"):
        if s.startswith(p):
            s = s[len(p):]
            forced_hf = True
            break
    forced_ollama = False
    if not forced_hf and s.startswith("ollama:"):
        s = s[len("ollama:"):]
        forced_ollama = True

    # split a trailing :tag (no '/' after the colon)
    name_part, tag = s, ""
    colon = s.rfind(":")
    if colon != -1 and "/" not in s[colon:]:
        name_part, tag = s[:colon], s[colon + 1:]

    parts = name_part.split("/")
    if any(not p.strip() for p in parts):
        raise ValueError(f"invalid model reference: '{raw}'")

    if forced_hf or (not forced_ollama and len(parts) >= 2):
        if len(parts) != 2:
            raise ValueError(f"Hugging Face references look like org/repo[:quant] (got '{raw}')")
        return ModelRef(source="hf", ns=parts[0], name=parts[1], tag=tag)

    if len(parts) == 1:
        return ModelRef(source="ollama", ns="library", name=parts[0], tag=tag or "latest")
    if len(parts) == 2:
        return ModelRef(source="ollama", ns=parts[0], name=parts[1], tag=tag or "latest")
    raise ValueError(f"Ollama references look like [user/]name[:tag] (got '{raw}')")


@dataclass
class LocalModel:
    model_path: Path
    dir: Path | None = None
    manifest: dict = field(default_factory=dict)


def installed_dir(ref: ModelRef) -> "Path | None":
    """The directory actually holding this model: the current store first,
    then any pre-rebrand one. Returns None when it is installed nowhere."""
    for root in [models_root(), *legacy_models_roots()]:
        d = ref.store_dir(root)
        if (d / "manifest.json").exists():
            return d
    return None


def find_local(ref: ModelRef) -> LocalModel | None:
    if ref.source == "file":
        assert ref.path is not None
        return LocalModel(model_path=ref.path) if ref.path.exists() else None
    d = installed_dir(ref)
    if d is None:
        return None
    try:
        manifest = json.loads((d / "manifest.json").read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    model_file = manifest.get("model_file", "")
    if not model_file or not (d / model_file).exists():
        return None
    return LocalModel(model_path=d / model_file, dir=d, manifest=manifest)


def write_manifest(d: Path, manifest: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", "utf-8")
    tmp.replace(d / "manifest.json")


def _clean_nickname(nickname: str) -> str:
    # nicknames are echoed to a terminal and written into JSON, so replace
    # anything invisible or direction-changing with a space before it can be
    # stored: Cc for ESC (which would emit ANSI escape sequences), Cf for the
    # bidi overrides that reorder the rest of the line and for zero-width
    # joiners, and Cs for lone surrogates arriving from argv
    text = "".join(" " if unicodedata.category(ch) in ("Cc", "Cf", "Cs") else ch
                   for ch in str(nickname))
    return " ".join(text.strip().split())


def _quarantine_nicknames(path: Path, err: Exception) -> None:
    """Move an unreadable nicknames file aside and say so, once."""
    spoiled = path.with_suffix(path.suffix + ".corrupt")
    try:
        os.replace(path, spoiled)
    except OSError:
        print(f"alpaccaroo: warning: {path} is unreadable ({err}); "
              f"model nicknames are being ignored", file=sys.stderr)
        return
    print(f"alpaccaroo: warning: {path} is unreadable ({err}); "
          f"moved it to {spoiled.name} and starting a new nickname file",
          file=sys.stderr)


def _read_nickname_file(path: Path, *, quarantine: bool) -> dict[str, str]:
    try:
        data = json.loads(path.read_text("utf-8"))
    except OSError:   # missing, unreadable, a directory - nothing to quarantine
        return {}
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as e:
        # RecursionError is a RuntimeError, not a ValueError: deeply nested
        # JSON would otherwise fail every command that touches the map.
        # The file is readable but unusable - the next write would replace it,
        # so put it out of harm's way rather than destroying the only copy.
        if quarantine:
            _quarantine_nicknames(path, e)
        else:
            print(f"alpaccaroo: warning: {path} is unreadable ({e}); "
                  f"model nicknames are being ignored", file=sys.stderr)
        return {}
    if isinstance(data, dict) and isinstance(data.get("nicknames"), dict):
        data = data["nicknames"]
    if not isinstance(data, dict):
        return {}
    out: dict[str, str] = {}
    for nickname, target in data.items():
        nickname = _clean_nickname(str(nickname))
        target = str(target).strip()
        if not nickname or not target:
            continue
        try:
            out[nickname] = parse_model_ref(target).display()
        except ValueError:
            continue
    return out


def _read_nicknames() -> dict[str, str]:
    """Read current aliases, falling back to pre-rebrand aliases.

    Legacy entries are loaded first so the current file wins on collisions.
    The current file keeps the existing quarantine behavior; legacy files are
    never moved or overwritten because they are migration sources of truth.
    """
    current = _nicknames_file()
    merged = {}
    for path in [*legacy_nickname_files(), current]:
        merged.update(_read_nickname_file(path, quarantine=path == current))
    return merged


def _write_nicknames(nicknames: dict[str, str]) -> None:
    path = _nicknames_file()
    if not nicknames:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = {k: v for k, v in sorted(nicknames.items(), key=lambda kv: kv[0].lower())}
    payload = json.dumps({"nicknames": cleaned}, indent=2) + "\n"
    # a temp name shared between processes is worse than no temp file at all:
    # two writers interleave into it and replace() then publishes the garbage
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent),
                                    prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


@contextlib.contextmanager
def _nicknames_lock():
    """Serialise the nickname read-modify-write across processes.

    Best-effort: an atomic write alone still loses updates, because two
    processes each read the old map and write back only their own entry.
    If locking is unavailable the mutation still proceeds - a lost alias is
    better than a command that refuses to run.
    """
    path = _nicknames_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(path.with_name(path.name + ".lock"), "a+b")
    except OSError:
        yield
        return
    locked = False
    try:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            locked = True
        except (OSError, ImportError, ValueError):
            pass
        yield
    finally:
        if locked:
            try:
                if os.name == "nt":
                    import msvcrt
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except (OSError, ImportError, ValueError):
                pass
        fh.close()


def list_nicknames() -> dict[str, str]:
    """Return nickname -> canonical model reference."""
    return dict(_read_nicknames())


def nickname_for_model(model_name: str) -> str:
    try:
        canonical = parse_model_ref(model_name).display()
    except ValueError:
        canonical = model_name.strip()
    for nickname, target in _read_nicknames().items():
        if target == canonical:
            return nickname
    return ""


def resolve_model_input(raw: str) -> str:
    """Resolve a CLI/menu model input to a canonical model reference.

    Installed canonical names win over nicknames so adding a nickname cannot
    shadow an existing model. If no installed model matches, an exact or
    unique case-insensitive nickname is accepted before falling back to the
    normal model-reference parser.

    File references are returned unchanged. Their display() form is a
    pathlib-normalised string ("./tiny" -> "tiny"), and callers re-parse what
    they get back - so returning display() here would silently reclassify a
    local path as a registry name.
    """
    s = raw.strip()
    if not s:
        raise ValueError("empty model name")
    ref: ModelRef | None
    try:
        ref = parse_model_ref(s)
    except ValueError:
        ref = None
    if ref is not None and ref.source == "file":
        return s
    if ref is not None and find_local(ref) is not None:
        return ref.display()

    cleaned = _clean_nickname(s)
    nicknames = _read_nicknames()
    if cleaned in nicknames:
        return nicknames[cleaned]
    matches = [target for nickname, target in nicknames.items()
               if nickname.lower() == cleaned.lower()]
    if len(set(matches)) == 1:
        return matches[0]
    if matches:
        raise ValueError(f"ambiguous model nickname: {raw}")
    if ref is not None:
        return ref.display()
    return parse_model_ref(s).display()


def set_model_nickname(model_name: str, nickname: str) -> tuple[str, str]:
    target = resolve_model_input(model_name)
    ref = parse_model_ref(target)
    if ref.source == "file" or find_local(ref) is None:
        raise ValueError(f"{target} is not an installed model")
    nickname = _clean_nickname(nickname)
    if not nickname:
        raise ValueError("empty model nickname")

    try:
        nick_ref = parse_model_ref(nickname)
    except ValueError:
        nick_ref = None  # not a parseable reference, so it cannot collide
    if nick_ref is not None and nick_ref.source == "file":
        raise ValueError(
            f"nickname '{nickname}' looks like a file path; "
            f"it could never be resolved back to a model")
    if nick_ref is not None and find_local(nick_ref) is not None \
            and nick_ref.display() != target:
        raise ValueError(
            f"nickname '{nickname}' conflicts with installed model {nick_ref.display()}")

    with _nicknames_lock():
        nicknames = _read_nicknames()
        for existing, existing_target in list(nicknames.items()):
            if existing_target == target:
                del nicknames[existing]
            elif existing.lower() == nickname.lower():
                raise ValueError(
                    f"nickname '{nickname}' already points to {existing_target}")
        nicknames[nickname] = target
        _write_nicknames(nicknames)
    return nickname, target


def clear_model_nickname(model_name: str) -> str:
    target = resolve_model_input(model_name)
    with _nicknames_lock():
        nicknames = _read_nicknames()
        removed = ""
        for nickname, existing_target in list(nicknames.items()):
            if existing_target == target:
                removed = nickname
                del nicknames[nickname]
        if removed:
            _write_nicknames(nicknames)
    return removed


def _remove_nicknames_for_model(model_name: str) -> None:
    with _nicknames_lock():
        nicknames = _read_nicknames()
        changed = False
        for nickname, target in list(nicknames.items()):
            if target == model_name:
                del nicknames[nickname]
                changed = True
        if changed:
            _write_nicknames(nicknames)


def now_iso8601() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def list_models() -> list[dict]:
    out = []
    roots = [r for r in [models_root(), *legacy_models_roots()] if r.exists()]
    if not roots:
        return out
    nicknames = _read_nicknames()
    model_to_nickname = {target: nickname for nickname, target in nicknames.items()}
    seen: set[str] = set()
    manifests = [(root, mf) for root in roots
                 for mf in sorted(root.rglob("manifest.json"))]
    for root, mf in manifests:
        d = mf.parent
        try:
            manifest = json.loads(mf.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        size = manifest.get("size", 0)
        if not size:
            size = sum(f.stat().st_size for f in d.glob("*.gguf"))
        name = manifest.get("name", d.name)
        # nickname_for_model canonicalises through parse_model_ref before
        # comparing; match on the same form so the two never disagree about
        # whether a model has an alias
        try:
            canonical = parse_model_ref(name).display()
        except ValueError:
            canonical = name
        # the current store wins: a model present in both is listed once,
        # from the copy find_local would actually open
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append({
            "name": name,
            "nickname": (model_to_nickname.get(canonical)
                         or model_to_nickname.get(name, "")),
            "source": manifest.get("source", "?"),
            "size": int(size),
            "pulled_at": manifest.get("pulled_at", ""),
            "dir": d,
            "legacy_store": root != models_root(),
        })
    out.sort(key=lambda m: m["name"])
    return out


def remove_model(ref: ModelRef) -> bool:
    if ref.source == "file":
        raise ValueError("refusing to delete a raw file path; remove it yourself if intended")
    d = installed_dir(ref)
    if d is None:
        return False
    # prune back to whichever store the model was actually found in
    root = next((r for r in [models_root(), *legacy_models_roots()]
                 if d.is_relative_to(r)), models_root())
    name = ref.display()
    for f in sorted(d.rglob("*"), reverse=True):
        f.unlink() if f.is_file() else f.rmdir()
    d.rmdir()
    # the model is already gone, so nothing below here may report rm as failed:
    # not the alias bookkeeping, and not pruning the now-empty parents
    try:
        _remove_nicknames_for_model(name)
    except OSError:
        pass
    try:
        parent = d.parent
        while parent != root and parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
    except OSError:
        pass
    return True
