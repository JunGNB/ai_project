"""Dependency-free HTTP download helpers shared by dataset adapters.

The benchmark keeps its dataset handling deliberately boring: raw bytes are
fetched over plain HTTP/S (HuggingFace mirror or GitHub raw), stored under
``data/raw/<dataset>/`` and normalised into JSONL under ``data/processed/`` by
the adapters.  No ``datasets`` / ``huggingface_hub`` dependency is involved.

Design goals
------------

* **Idempotent** — an existing file whose SHA-256 matches the recorded
  manifest is reused without touching the network.
* **Resumable** — interrupted transfers continue with an HTTP ``Range``
  request (``.part`` file), falling back to a full download when the server
  ignores the range.
* **Auditable** — size, SHA-256, source URL and timestamp of every file are
  recorded in ``<raw_root>/download_manifest.json`` and surfaced by
  ``modelbenchmark download`` and ``docs/DATASETS.md``.
* **Clear failures** — exhausted retries raise :class:`DownloadError` with the
  URL, target path and the last transport error, so the CLI can report which
  dataset/file is broken instead of silently loading empty data.

A manifest lives next to the dataset files, e.g.::

    data/raw/mmlu/download_manifest.json
    {
      "schema_version": 1,
      "dataset": "mmlu",
      "files": {
        "all/test-00000-of-00001.parquet": {
          "url": "https://hf-mirror.com/...",
          "size": 3504718,
          "sha256": "…",
          "downloaded_at": "2026-11-02T11:43:00+00:00"
        }
      }
    }
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

#: HuggingFace mirror used because the origin is not reachable from the
#: benchmark host; ``<repo>/resolve/main/<path>`` resolves to raw bytes.
HF_MIRROR = "https://hf-mirror.com"

DEFAULT_RETRIES = 5
DEFAULT_RETRY_DELAY_S = 2.0
DEFAULT_TIMEOUT_S = 180.0
MAX_RETRY_DELAY_S = 30.0

MANIFEST_NAME = "download_manifest.json"
MANIFEST_SCHEMA_VERSION = 1
USER_AGENT = "modelbenchmark-dataset-downloader/0.1"

_CHUNK = 1 << 20


class DownloadError(RuntimeError):
    """A required dataset file could not be downloaded or failed verification."""


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #


def sha256_file(path: Path, chunk_size: int = _CHUNK) -> str:
    """Return the hex SHA-256 digest of ``path``."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(raw_root: Path) -> Path:
    return Path(raw_root) / MANIFEST_NAME


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _manifest_key(dest: Path, raw_root: Path) -> str:
    try:
        return dest.resolve().relative_to(raw_root.resolve()).as_posix()
    except ValueError:
        return dest.name


def read_manifest(raw_root: Path) -> Dict[str, Any]:
    """Read ``<raw_root>/download_manifest.json``; return an empty shell if absent.

    A corrupt manifest is treated as absent (the files are re-hashed and the
    manifest rewritten) rather than failing the whole dataset.
    """

    path = manifest_path(raw_root)
    if not path.is_file():
        return {"schema_version": MANIFEST_SCHEMA_VERSION, "dataset": Path(raw_root).name, "files": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"schema_version": MANIFEST_SCHEMA_VERSION, "dataset": Path(raw_root).name, "files": {}}
    if not isinstance(data, dict) or not isinstance(data.get("files"), dict):
        return {"schema_version": MANIFEST_SCHEMA_VERSION, "dataset": Path(raw_root).name, "files": {}}
    return data


def write_manifest(raw_root: Path, manifest: Dict[str, Any]) -> Path:
    """Atomically write a manifest under ``raw_root``."""

    path = manifest_path(raw_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.setdefault("schema_version", MANIFEST_SCHEMA_VERSION)
    manifest.setdefault("dataset", Path(raw_root).name)
    manifest.setdefault("files", {})
    manifest["updated_at"] = _now()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


def _record(
    manifest: Dict[str, Any],
    key: str,
    *,
    url: str,
    dest: Path,
    size: int,
    sha256: str,
) -> None:
    files = manifest.setdefault("files", {})
    previous = files.get(key) or {}
    files[key] = {
        "url": url,
        "size": int(size),
        "sha256": sha256,
        # preserve the original download timestamp when only re-verifying
        "downloaded_at": previous.get("downloaded_at") or _now(),
    }


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #


def _retry_delay(exc: BaseException, attempt: int, base_delay: float) -> float:
    delay = min(MAX_RETRY_DELAY_S, base_delay * (2 ** max(0, attempt - 1)))
    headers = getattr(exc, "headers", None)
    if headers is not None:
        retry_after = headers.get("Retry-After") if hasattr(headers, "get") else None
        if retry_after:
            try:
                delay = max(delay, min(MAX_RETRY_DELAY_S, float(retry_after)))
            except (TypeError, ValueError):
                pass
    return delay


def _fetch(
    url: str,
    dest: Path,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    retries: int = DEFAULT_RETRIES,
    retry_delay_s: float = DEFAULT_RETRY_DELAY_S,
) -> Path:
    """Download ``url`` to ``dest`` (resumable, retried, atomic).

    Kept as a separate function so tests can monkeypatch it without opening a
    socket.
    """

    dest = Path(dest)
    part = dest.with_name(dest.name + ".part")
    attempt = 0
    while True:
        start = part.stat().st_size if part.is_file() else 0
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            if start:
                request.add_header("Range", f"bytes={start}-")
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                status = getattr(response, "status", 200)
                if start and status == 200:
                    # Server ignored the Range header: restart from scratch.
                    _log(f"  server ignored resume for {url}; restarting download")
                    start = 0
                mode = "ab" if start else "wb"
                with part.open(mode) as out:
                    shutil.copyfileobj(response, out, _CHUNK)
            part.replace(dest)
            return dest
        except urllib.error.HTTPError as exc:  # subclass of URLError
            attempt += 1
            if attempt > retries:
                raise DownloadError(
                    f"HTTP {exc.code} while downloading {url} -> {dest} "
                    f"(retried {retries} times; last error: {exc.reason})"
                ) from exc
            delay = _retry_delay(exc, attempt, retry_delay_s)
            _log(f"  HTTP {exc.code} for {url}; retry {attempt}/{retries} in {delay:.1f}s")
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
            attempt += 1
            if attempt > retries:
                raise DownloadError(
                    f"network error while downloading {url} -> {dest} "
                    f"(retried {retries} times; last error: {exc})"
                ) from exc
            delay = _retry_delay(exc, attempt, retry_delay_s)
            _log(f"  network error for {url}: {exc}; retry {attempt}/{retries} in {delay:.1f}s")
            time.sleep(delay)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def download_file(
    url: str,
    dest: Path,
    *,
    raw_root: Optional[Path] = None,
    expected_sha256: Optional[str] = None,
    expected_size: Optional[int] = None,
    min_bytes: int = 1,
    retries: int = DEFAULT_RETRIES,
    retry_delay_s: float = DEFAULT_RETRY_DELAY_S,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    force: bool = False,
) -> Path:
    """Download ``url`` to ``dest`` unless a verified copy already exists.

    Parameters
    ----------
    raw_root:
        Dataset root that owns the manifest.  Defaults to ``dest.parent``.
    expected_sha256 / expected_size:
        Optional pinned values.  When supplied they are verified after the
        download (and used for reuse checks when no manifest entry exists).
    min_bytes:
        Lower bound used to reject truncated files / HTML error pages when the
        checksum is not known yet.
    force:
        Ignore an existing file and download again.

    Returns the local path (``dest``).
    """

    dest = Path(dest)
    root = Path(raw_root) if raw_root is not None else dest.parent
    root.mkdir(parents=True, exist_ok=True)
    dest.parent.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(root)
    key = _manifest_key(dest, root)
    entry = (manifest.get("files") or {}).get(key) or {}

    if not force and dest.is_file():
        size = dest.stat().st_size
        if size >= min_bytes:
            known_sha = expected_sha256 or entry.get("sha256")
            if known_sha:
                actual = sha256_file(dest)
                if actual == known_sha:
                    _record(manifest, key, url=url, dest=dest, size=size, sha256=actual)
                    write_manifest(root, manifest)
                    return dest
                _log(
                    f"  checksum mismatch for {dest} (expected {known_sha[:12]}…, "
                    f"got {actual[:12]}…); re-downloading"
                )
            elif expected_size is None or size == expected_size:
                actual = sha256_file(dest)
                _record(manifest, key, url=url, dest=dest, size=size, sha256=actual)
                write_manifest(root, manifest)
                return dest
            else:
                _log(f"  size mismatch for {dest} ({size} != {expected_size}); re-downloading")

    if dest.exists():
        dest.unlink()
    part = dest.with_name(dest.name + ".part")
    if part.exists():
        part.unlink()

    _log(f"  downloading {url}")
    _fetch(
        url,
        dest,
        timeout_s=timeout_s,
        retries=retries,
        retry_delay_s=retry_delay_s,
    )

    size = dest.stat().st_size
    if size < min_bytes:
        dest.unlink(missing_ok=True)
        raise DownloadError(
            f"downloaded file {dest} is too small ({size} < {min_bytes} bytes); "
            f"source may be an error page: {url}"
        )
    if expected_size is not None and size != expected_size:
        dest.unlink(missing_ok=True)
        raise DownloadError(
            f"downloaded file {dest} has unexpected size {size} (pinned {expected_size}): {url}"
        )
    actual = sha256_file(dest)
    if expected_sha256 and actual != expected_sha256:
        dest.unlink(missing_ok=True)
        raise DownloadError(
            f"checksum mismatch for {dest}: expected {expected_sha256}, got {actual} ({url})"
        )
    _record(manifest, key, url=url, dest=dest, size=size, sha256=actual)
    write_manifest(root, manifest)
    return dest


def list_raw_files(raw_root: Path) -> List[Dict[str, Any]]:
    """Describe every non-transient file under ``raw_root`` (path/size/sha256)."""

    root = Path(raw_root)
    rows: List[Dict[str, Any]] = []
    if not root.is_dir():
        return rows
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name == MANIFEST_NAME or path.name.endswith((".part", ".tmp")):
            continue
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:  # pragma: no cover - rglob always yields children
            relative = path.name
        rows.append(
            {
                "path": str(path),
                "relative_path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


__all__ = [
    "HF_MIRROR",
    "MANIFEST_NAME",
    "DownloadError",
    "download_file",
    "list_raw_files",
    "manifest_path",
    "read_manifest",
    "sha256_file",
    "write_manifest",
]
