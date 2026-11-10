#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fetch / verify the raw assets for the accuracy test set (``dataset/raw/``).

Standard library only (``urllib``, ``zipfile``), so it works on a fresh clone
before ``.venv`` is installed.

    python tool/dataset/accuracy/fetch.py                 # download missing files
    python tool/dataset/accuracy/fetch.py --verify        # read-only size+SHA-256 check
    python tool/dataset/accuracy/fetch.py --dry-run       # print plan, no writes
    python tool/dataset/accuracy/fetch.py --only tokenizer,nltk
    python tool/dataset/accuracy/fetch.py --mirror https://huggingface.co

Behaviour:
* reads the committed ``raw_spec.json`` (path/url/bytes/sha256 + extract rules);
* streams into ``<dest>.part`` then verifies size + SHA-256 before atomically
  replacing the destination (a failed download never leaves a corrupt file);
* extracts LongBench v1 ``data.zip`` and the NLTK punkt archives when the
  declared sentinel file is absent (or ``--force``);
* writes the generated empty package marker ``ifeval/.../__init__.py``;
* ``--verify`` is read-only.

Exit codes: 0 = all selected assets OK; 1 = missing/corrupt/download failure;
2 = usage/spec error.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tool.common.fs import RAW_DIR  # noqa: E402
from tool.dataset.accuracy import sources  # noqa: E402

DEFAULT_RETRIES = 3
DEFAULT_TIMEOUT = 60.0
CHUNK = 1 << 20
PART_SUFFIX = ".part"
UA = "modelbench-accuracy-fetch/1.0"

OK, SKIP, DOWNLOADED, EXTRACTED = "ok", "skip", "downloaded", "extracted"
FAIL = "fail"


def _rewrite_url(url: str, mirror: str) -> str:
    split = urlsplit(url)
    if split.hostname not in {"huggingface.co", "hf-mirror.com"}:
        return url
    m = urlsplit(mirror)
    return urlunsplit((m.scheme or split.scheme, m.netloc, split.path, split.query, split.fragment))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_file(path: Path, expected_bytes: int, expected_sha: str) -> Tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    size = path.stat().st_size
    if size != expected_bytes:
        return False, f"size {size} != {expected_bytes}"
    digest = _sha256(path)
    if digest != expected_sha:
        return False, f"sha256 {digest[:12]} != {expected_sha[:12]}"
    return True, "ok"


def _download(url: str, dest: Path, expected_bytes: int, expected_sha: str,
              retries: int, timeout: float) -> Tuple[bool, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + PART_SUFFIX)
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(part, "wb") as out:
                shutil.copyfileobj(resp, out, CHUNK)
            size = part.stat().st_size
            if size != expected_bytes:
                raise ValueError(f"size {size} != {expected_bytes}")
            digest = _sha256(part)
            if digest != expected_sha:
                raise ValueError(f"sha256 {digest[:12]} != {expected_sha[:12]}")
            os.replace(part, dest)
            return True, "downloaded"
        except Exception as exc:  # noqa: BLE001 - one retry loop for all network errors
            last_error = f"{type(exc).__name__}: {exc}"
            part.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2 ** attempt, 10))
    return False, last_error


def _safe_extract(archive: Path, dest: Path) -> List[str]:
    wrote: List[str] = []
    dest = dest.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            target = (dest / member.filename).resolve()
            if not str(target).startswith(str(dest) + os.sep):
                raise ValueError(f"zip path escape: {member.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out, CHUNK)
            wrote.append(member.filename)
    return wrote


def main(argv: Iterable[str] | None = None) -> int:
    spec = sources.load_spec()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mirror", default=spec["mirror_default"],
                    help="HF host mirror (huggingface.co URLs are rewritten); default %(default)s")
    ap.add_argument("--raw-dir", default=str(RAW_DIR), help="destination root (default dataset/raw)")
    ap.add_argument("--only", default="", help="comma-separated dataset names to process")
    ap.add_argument("--verify", action="store_true", help="read-only size+SHA-256 verification")
    ap.add_argument("--force", action="store_true", help="re-download / re-extract even if OK")
    ap.add_argument("--dry-run", action="store_true", help="print the plan without writing")
    ap.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    args = ap.parse_args(list(argv) if argv is not None else None)

    raw_dir = Path(args.raw_dir)
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    files = [f for f in spec["files"] if not only or f["dataset"] in only]
    extracts = [e for e in spec["extract"] if not only or e["dataset"] in only]
    generated = [g for g in spec["generated"] if not only or g.get("dataset", "ifeval") in only]
    unknown = only - {f["dataset"] for f in spec["files"]}
    if unknown:
        print(f"unknown --only dataset(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2

    print(f"raw_dir={raw_dir}  files={len(files)}  extract={len(extracts)}  mirror={args.mirror}  mode={'verify' if args.verify else 'fetch'}")
    counts: Dict[str, int] = {}
    failed: List[str] = []

    for item in files:
        dest = raw_dir / item["path"]
        ok, detail = _check_file(dest, item["bytes"], item["sha256"])
        if ok and not args.force:
            counts[SKIP] = counts.get(SKIP, 0) + 1
            print(f"  [ok]    {item['path']}")
            continue
        if args.verify:
            counts[FAIL] = counts.get(FAIL, 0) + 1
            failed.append(f"{item['path']} ({detail})")
            print(f"  [FAIL]  {item['path']}: {detail}")
            continue
        url = _rewrite_url(item["url"], args.mirror)
        if args.dry_run:
            counts[DOWNLOADED] = counts.get(DOWNLOADED, 0) + 1
            print(f"  [plan]  download {item['path']} <- {url}")
            continue
        ok, detail = _download(url, dest, item["bytes"], item["sha256"], args.retries, args.timeout)
        if ok:
            counts[DOWNLOADED] = counts.get(DOWNLOADED, 0) + 1
            print(f"  [down]  {item['path']}")
        else:
            counts[FAIL] = counts.get(FAIL, 0) + 1
            failed.append(f"{item['path']} ({detail})")
            print(f"  [FAIL]  {item['path']}: {detail}")

    for item in extracts:
        sentinel = raw_dir / item["sentinel"]
        if sentinel.exists() and not args.force:
            counts[SKIP] = counts.get(SKIP, 0) + 1
            print(f"  [ok]    {item['archive']} (already extracted)")
            continue
        archive = raw_dir / item["archive"]
        if not archive.is_file():
            counts[FAIL] = counts.get(FAIL, 0) + 1
            failed.append(f"{item['archive']} (archive missing)")
            print(f"  [FAIL]  extract {item['archive']}: archive missing")
            continue
        if args.verify:
            counts[FAIL] = counts.get(FAIL, 0) + 1
            failed.append(f"{item['archive']} (sentinel missing: {item['sentinel']})")
            print(f"  [FAIL]  extract {item['archive']}: sentinel missing")
            continue
        if args.dry_run:
            counts[EXTRACTED] = counts.get(EXTRACTED, 0) + 1
            print(f"  [plan]  extract {item['archive']} -> {item['dest']}")
            continue
        wrote = _safe_extract(archive, raw_dir / item["dest"])
        counts[EXTRACTED] = counts.get(EXTRACTED, 0) + 1
        print(f"  [unzip] {item['archive']} -> {item['dest']} ({len(wrote)} files)")

    for item in generated:
        dest = raw_dir / item["path"]
        if dest.exists() and not args.force:
            counts[SKIP] = counts.get(SKIP, 0) + 1
            continue
        if args.verify:
            if not dest.exists():
                counts[FAIL] = counts.get(FAIL, 0) + 1
                failed.append(f"{item['path']} (missing generated file)")
            continue
        if args.dry_run:
            print(f"  [plan]  write {item['path']}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(item.get("content", ""), encoding="utf-8")
        counts[EXTRACTED] = counts.get(EXTRACTED, 0) + 1

    print("summary:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if failed:
        print("FAILED:", file=sys.stderr)
        for item in failed:
            print(f"  - {item}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
