# Alpaccaroo - model downloads from the Ollama registry and Hugging Face,
# using only the Python standard library (urllib + hashlib).
#
# Protocol notes: Ollama models live in an OCI-style registry
# (https://registry.ollama.ai) - a JSON manifest lists content-addressed
# layers; the GGUF weights layer has media type
# "application/vnd.ollama.image.model". This is an independent
# implementation of that protocol. Hugging Face models are plain files
# under /<org>/<repo>/resolve/main/<file>, listed via the
# /api/models/.../tree endpoint.
#
# Downloads resume via HTTP Range requests and are verified against the
# publisher's SHA-256 digests. Nothing here runs during inference.
# MIT License. See LICENSE.
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

from .store import (LocalModel, ModelRef, find_local, human_size,
                    now_iso8601, write_manifest)

UA = "alpaccaroo/0.2 (+https://github.com/Itwas1time/Alpacca)"

OLLAMA_MEDIA = {
    "application/vnd.ollama.image.model": "model.gguf",
    "application/vnd.ollama.image.params": "params.json",
    "application/vnd.ollama.image.system": "system.txt",
    "application/vnd.ollama.image.template": "template.txt",
    "application/vnd.ollama.image.license": "license.txt",
}


def _registry() -> str:
    return os.environ.get("ALPACCAROO_OLLAMA_REGISTRY", "https://registry.ollama.ai")


def _hf_endpoint() -> str:
    return os.environ.get("ALPACCAROO_HF_ENDPOINT", "https://huggingface.co")


def _hf_headers() -> dict:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _get_json(url: str, headers: dict | None = None):
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download(url: str, dest: Path, expected_size: int = 0, sha256: str = "",
              headers: dict | None = None, verify: bool = True,
              label: str = "") -> None:
    """Resumable download with progress and digest verification."""
    if dest.exists():
        if (not expected_size or dest.stat().st_size == expected_size) and \
           (not verify or not sha256 or _sha256_file(dest) == sha256):
            print(f"already present: {dest.name}", file=sys.stderr)
            return
        dest.unlink()

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    done = partial.stat().st_size if partial.exists() else 0
    if expected_size and done > expected_size:
        partial.unlink()
        done = 0

    hdrs = {"User-Agent": UA, **(headers or {})}
    if done:
        hdrs["Range"] = f"bytes={done}-"

    mode = "ab" if done else "wb"
    label = label or dest.name
    try:
        req = urllib.request.Request(url, headers=hdrs)
        with urllib.request.urlopen(req, timeout=60) as resp:
            if done and resp.status != 206:  # server ignored the Range header
                done = 0
                mode = "wb"
            total = expected_size or done + int(resp.headers.get("Content-Length") or 0)
            with open(partial, mode) as f:
                last_pct = -1
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = int(done * 100 / total)
                        if pct != last_pct:
                            print(f"\r{label}: {pct:3d}% of {human_size(total)}",
                                  end="", file=sys.stderr, flush=True)
                            last_pct = pct
        print("", file=sys.stderr)
    except urllib.error.HTTPError as e:
        if e.code == 416 and expected_size and partial.exists() and \
                partial.stat().st_size == expected_size:
            pass  # partial file already complete
        else:
            raise RuntimeError(
                f"download failed (HTTP {e.code}): {url}\n"
                f"  partial data kept at {partial} - rerun to resume") from e
    except OSError as e:
        raise RuntimeError(
            f"download failed: {url} ({e})\n"
            f"  partial data kept at {partial} - rerun to resume") from e

    got = partial.stat().st_size if partial.exists() else 0
    if expected_size and got != expected_size:
        raise RuntimeError(
            f"size mismatch for {dest.name}: expected {expected_size}, got {got}\n"
            f"  partial data kept at {partial} - rerun to resume")
    if verify and sha256:
        print("verifying sha256... ", end="", file=sys.stderr, flush=True)
        actual = _sha256_file(partial)
        if actual != sha256:
            partial.unlink()
            raise RuntimeError(f"sha256 mismatch for {dest.name} "
                               f"(expected {sha256}, got {actual}); removed corrupt download")
        print("ok", file=sys.stderr)
    partial.replace(dest)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- Ollama registry ------------------------------------------------------

def _pull_ollama(ref: ModelRef, force: bool, verify: bool) -> LocalModel:
    base = f"{_registry()}/v2/{ref.ns}/{ref.name}"
    print(f"pulling manifest {ref.display()} from {_registry()}", file=sys.stderr)
    man = _get_json(f"{base}/manifests/{ref.tag}",
                    {"Accept": "application/vnd.docker.distribution.manifest.v2+json"})
    layers = man.get("layers")
    if not isinstance(layers, list):
        raise RuntimeError(f"unexpected registry manifest for {ref.display()} (no layers)")

    d = ref.store_dir()
    out = {"name": ref.display(), "source": "ollama",
           "registry_ref": f"{_registry()}/{ref.ns}/{ref.name}:{ref.tag}"}
    model_size = 0
    todo = []
    for layer in layers:
        media = layer.get("mediaType", "")
        digest = layer.get("digest", "")
        size = int(layer.get("size", 0))
        fname = OLLAMA_MEDIA.get(media)
        if not fname or not digest.startswith("sha256:"):
            if media.endswith((".projector", ".adapter")):
                print(f"note: skipping {media.rsplit('.', 1)[-1]} layer "
                      f"(not supported by the python engine yet)", file=sys.stderr)
            continue
        todo.append((fname, digest[7:], size))
        if fname == "model.gguf":
            out["digest"] = digest
            model_size = size

    if not any(f == "model.gguf" for f, _, _ in todo):
        raise RuntimeError(f"{ref.display()} has no GGUF weights layer; alpaccaroo cannot run it")

    d.mkdir(parents=True, exist_ok=True)
    for fname, digest_hex, size in todo:
        print(f"pulling {fname} ({human_size(size)})", file=sys.stderr)
        target = d / fname
        if force and target.exists():
            target.unlink()
        _download(f"{base}/blobs/sha256:{digest_hex}", target, size, digest_hex,
                  verify=verify, label=fname)

    params_file = d / "params.json"
    if params_file.exists():
        try:
            params = json.loads(params_file.read_text("utf-8"))
            if isinstance(params, dict):
                out["params"] = params
        except (OSError, json.JSONDecodeError):
            pass
    system_file = d / "system.txt"
    if system_file.exists():
        text = system_file.read_text("utf-8", errors="replace").strip()
        if text:
            out["system"] = text

    out["model_file"] = "model.gguf"
    out["size"] = model_size
    out["pulled_at"] = now_iso8601()
    write_manifest(d, out)
    local = find_local(ref)
    if local is None:
        raise RuntimeError("internal error: model not found after pull")
    print(f"success: {ref.display()} ready ({human_size(model_size)})", file=sys.stderr)
    return local


# ---- Hugging Face ----------------------------------------------------------

_QUANT_PREFERENCE = ("q4_k_m", "q4_k_s", "q5_k_m", "q5_k_s", "q4_0", "q8_0",
                     "q6_k", "f16", "bf16", "f32")


def _hf_list_gguf(org: str, repo: str) -> list[dict]:
    url = f"{_hf_endpoint()}/api/models/{org}/{repo}/tree/main?recursive=true"
    tree = _get_json(url, _hf_headers())
    files = []
    for e in tree:
        path = e.get("path", "")
        if not path.lower().endswith(".gguf"):
            continue
        lfs = e.get("lfs") or {}
        files.append({"path": path,
                      "size": int(e.get("size") or lfs.get("size") or 0),
                      "sha256": lfs.get("oid", "")})
    return files


def _hf_list_or_empty(org: str, repo: str) -> list[dict]:
    """List a repo, treating only a missing repo as an empty result.

    A previous version swallowed every HTTP/network error here and later
    reported the misleading message "no .gguf files".  That made an offline
    or blocked Hugging Face connection look like a bad model repository.
    """
    try:
        return _hf_list_gguf(org, repo)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return []
        raise RuntimeError(
            f"Hugging Face returned HTTP {e.code} while listing "
            f"{org}/{repo} at {_hf_endpoint()}") from e
    except urllib.error.URLError as e:
        reason = getattr(e, "reason", e)
        raise RuntimeError(
            f"cannot reach Hugging Face at {_hf_endpoint()} while listing "
            f"{org}/{repo}: {reason}") from e
    except OSError as e:
        raise RuntimeError(
            f"cannot access Hugging Face at {_hf_endpoint()} while listing "
            f"{org}/{repo}: {e}") from e


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _is_mmproj(path: str) -> bool:
    return _basename(path).lower().startswith("mmproj")


def _is_split_gguf(path: str) -> bool:
    return re.match(r"^.*-\d{5}-of-\d{5}\.gguf$", path, re.IGNORECASE) is not None


# IQ-family quantizations have no decoder in the engine yet; choosing one
# would download gigabytes that then fail to load
_UNSUPPORTED_QUANT = re.compile(r"iq\d", re.IGNORECASE)


def _hf_choose(files: list[dict], selector: str) -> dict | None:
    weights = [f for f in files if not _is_mmproj(f["path"])]
    if not weights:
        return None
    single_file_weights = [f for f in weights if not _is_split_gguf(f["path"])]
    candidates = single_file_weights or weights
    if selector:
        chosen = None
        for f in candidates:
            if f["path"] == selector or _basename(f["path"]) == selector:
                chosen = f
                break
        if chosen is None:
            sel = selector.lower()
            matches = [f for f in candidates
                       if sel in _basename(f["path"]).lower()]
            chosen = (min(matches, key=lambda f: len(f["path"]))
                      if matches else None)
        if chosen is not None and \
                _UNSUPPORTED_QUANT.search(_basename(chosen["path"])):
            raise ValueError(
                f"{_basename(chosen['path'])} is an IQ quantization, which "
                f"alpaccaroo cannot read yet - pick a Q4_K_M, Q5_K_M or Q8_0 "
                f"file from the repo instead")
        return chosen
    loadable = [f for f in candidates
                if not _UNSUPPORTED_QUANT.search(_basename(f["path"]))]
    if not loadable:
        names = ", ".join(sorted(_basename(f["path"]) for f in candidates)[:5])
        raise ValueError(
            f"this repo only ships IQ quantizations ({names}), which "
            f"alpaccaroo cannot read yet")
    for q in _QUANT_PREFERENCE:
        for f in loadable:
            if q in _basename(f["path"]).lower():
                return f
    return loadable[0]


def _hf_collect_parts(files: list[dict], chosen: dict) -> list[dict]:
    m = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", chosen["path"], re.IGNORECASE)
    if not m:
        return [chosen]
    prefix, total = m.group(1), m.group(3)
    parts = [f for f in files
             if (mm := re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", f["path"], re.IGNORECASE))
             and mm.group(1) == prefix and mm.group(3) == total]
    return sorted(parts, key=lambda f: f["path"])


def _pull_hf(ref: ModelRef, force: bool, verify: bool) -> LocalModel:
    print(f"fetching file list for {ref.ns}/{ref.name}", file=sys.stderr)
    repo = ref.name
    files = _hf_list_or_empty(ref.ns, repo)
    if not files and not repo.lower().endswith("-gguf"):
        # safetensors repos usually have a "<repo>-GGUF" sibling
        alt = repo + "-GGUF"
        print(f"no GGUF files in {ref.ns}/{repo} - trying {ref.ns}/{alt}", file=sys.stderr)
        files = _hf_list_or_empty(ref.ns, alt)
        if files:
            repo = alt
    if not files:
        raise RuntimeError(
            f"no .gguf files in {ref.ns}/{ref.name} - alpaccaroo runs GGUF models "
            f"(try a -GGUF repo, e.g. from ggml-org or bartowski)")

    chosen = _hf_choose(files, ref.tag)
    if chosen is None:
        raise RuntimeError(f"no GGUF in {ref.ns}/{repo} matches '{ref.tag}'")
    parts = _hf_collect_parts(files, chosen)
    if len(parts) > 1:
        raise RuntimeError(
            f"{chosen['path']} is a multi-part GGUF ({len(parts)} parts), but "
            "the python engine can only load single-file GGUFs right now; "
            "choose a single-file quantization or a smaller model")
    total_size = sum(p["size"] for p in parts)
    extra = f", {len(parts)} parts" if len(parts) > 1 else ""
    print(f"selected {chosen['path']} ({human_size(total_size)}{extra})", file=sys.stderr)

    d = ref.store_dir()
    d.mkdir(parents=True, exist_ok=True)
    resolve = f"{_hf_endpoint()}/{ref.ns}/{repo}/resolve/main/"
    for i, p in enumerate(parts, 1):
        print(f"downloading {i}/{len(parts)} {_basename(p['path'])} "
              f"({human_size(p['size'])})", file=sys.stderr)
        target = d / _basename(p["path"])
        if force and target.exists():
            target.unlink()
        _download(resolve + p["path"], target, p["size"], p["sha256"],
                  headers=_hf_headers(), verify=verify, label=_basename(p["path"]))

    manifest = {
        "name": ref.display(),
        "source": "hf",
        "registry_ref": f"{_hf_endpoint()}/{ref.ns}/{repo}",
        "model_file": _basename(parts[0]["path"]),
        "size": total_size,
        "pulled_at": now_iso8601(),
    }
    if chosen["sha256"]:
        manifest["digest"] = "sha256:" + chosen["sha256"]
    write_manifest(d, manifest)
    local = find_local(ref)
    if local is None:
        raise RuntimeError("internal error: model not found after pull")
    print(f"success: {ref.display()} ready ({human_size(total_size)})", file=sys.stderr)
    return local


# ---- entry point ------------------------------------------------------------

def pull_model(ref: ModelRef, force: bool = False, verify: bool = True) -> LocalModel:
    if ref.source == "file":
        raise ValueError(f"'{ref.path}' is a file path; nothing to pull")
    if not force:
        existing = find_local(ref)
        if existing is not None:
            print(f"{ref.display()} is already installed (use --force to re-pull)",
                  file=sys.stderr)
            return existing
    return _pull_ollama(ref, force, verify) if ref.source == "ollama" \
        else _pull_hf(ref, force, verify)
