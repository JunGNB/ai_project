"""Shared helpers for building the local evaluation sample sets.

Layout produced by the tooling in ``tool/``:

    dataset/raw/<dataset>/...            upstream files, untouched
    dataset/test/performance/<scenario>_<bucket>.jsonl
    dataset/test/accuracy/<dataset>.jsonl
    dataset/test/{performance,accuracy}/manifest.json
    dataset/.cache/tokenize.json         token-count cache (sha256 -> count)

Token counts are measured with the local oMLX tokenizer (``/v1/tokenize``) so that
the recorded ``input_tokens`` is the model-side view, not a character estimate.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset"
RAW = DATASET / "raw"
TEST_PERF = DATASET / "test" / "performance"
TEST_ACC = DATASET / "test" / "accuracy"
CACHE = DATASET / ".cache"
REPO_RAW = ROOT / "data" / "raw"

MIRROR = "https://hf-mirror.com"
OMLX = "http://127.0.0.1:8000"
TOKEN_MODEL = os.environ.get("MB_TOKEN_MODEL", "MiniCPM5-1B")

UA = {"User-Agent": "modelbench-dataset-builder/1.0"}


# --------------------------------------------------------------------------- #
# filesystem / json
# --------------------------------------------------------------------------- #
def ensure_dirs() -> None:
    for p in (RAW, TEST_PERF, TEST_ACC, CACHE):
        p.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --------------------------------------------------------------------------- #
# downloads (hf-mirror / raw github / direct url)
# --------------------------------------------------------------------------- #
def _download(url: str, dest: Path, retries: int = 4) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        have = tmp.stat().st_size if tmp.exists() else 0
        headers = dict(UA)
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=120.0) as r:
                if r.status_code == 416:  # already complete
                    r.close()
                    break
                r.raise_for_status()
                mode = "ab" if have and r.status_code == 206 else "wb"
                if mode == "wb":
                    have = 0
                with tmp.open(mode) as fh:
                    for chunk in r.iter_bytes(1 << 20):
                        fh.write(chunk)
            break
        except Exception as exc:  # noqa: BLE001 - network retries are intentional
            if attempt == retries:
                raise RuntimeError(f"download failed after {retries} tries: {url}: {exc}") from exc
            time.sleep(2 * attempt)
    tmp.replace(dest)
    return dest


def download_hf(repo: str, path: str, dest: Path, retries: int = 4) -> Path:
    return _download(f"{MIRROR}/datasets/{repo}/resolve/main/{path}", dest, retries=retries)


def download_url(url: str, dest: Path, retries: int = 4) -> Path:
    return _download(url, dest, retries=retries)


def download_range(url: str, dest: Path, max_bytes: int) -> Path:
    """Fetch only the first ``max_bytes`` of a large JSONL (for sampling)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = dict(UA)
    headers["Range"] = f"bytes=0-{max_bytes - 1}"
    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=180.0) as r:
        r.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
    return dest


def copy_cached(name: str, dest_name: Optional[str] = None) -> Path:
    """Copy a dataset already cached under the repo's data/raw into dataset/raw."""
    src = REPO_RAW / name
    if not src.exists():
        raise FileNotFoundError(f"cached dataset missing: {src}")
    dst = RAW / (dest_name or name)
    if dst.exists() and any(dst.iterdir()):
        return dst
    subprocess.run(["cp", "-R", str(src), str(dst)], check=True)
    return dst


# --------------------------------------------------------------------------- #
# oMLX tokenizer
# --------------------------------------------------------------------------- #
#: Interpreter bundled with oMLX.app; it ships the same ``tokenizers`` version
#: oMLX uses, so local counts match the server endpoint exactly.
OMLX_PYTHON = Path("/Applications/oMLX.app/Contents/Resources/Python/cpython-3.11/bin/python3")
OMLX_PYTHON_HOME = Path("/Applications/oMLX.app/Contents/Resources/Python/cpython-3.11")
OMLX_PYTHON_PATH = (
    "/Applications/oMLX.app/Contents/Resources:"
    "/Applications/oMLX.app/Contents/Resources/Python/framework-mlx-base/lib/python3.11/site-packages"
)
TOKENIZER_DIR = Path(
    os.environ.get("MB_TOKENIZER_DIR", os.path.expanduser("~/.omlx/models/openbmb/MiniCPM5-1B"))
)
_WORKER = Path(__file__).with_name("tokenize_worker.py")


def local_tokenizer_available() -> bool:
    """True when the model's tokenizer files and the oMLX interpreter exist."""
    if os.environ.get("MB_TOKENIZER_LOCAL", "1").lower() in ("0", "false", "no"):
        return False
    return (
        (TOKENIZER_DIR / "tokenizer.json").is_file()
        and (TOKENIZER_DIR / "chat_template.jinja").is_file()
        and OMLX_PYTHON.is_file()
    )


class _LocalWorker:
    """Persistent stdin/stdout token-count worker (see ``tokenize_worker.py``)."""

    def __init__(self) -> None:
        env = dict(os.environ)
        env["PYTHONHOME"] = str(OMLX_PYTHON_HOME)
        env["PYTHONPATH"] = OMLX_PYTHON_PATH
        env["MB_TOKENIZER_DIR"] = str(TOKENIZER_DIR)
        # stderr goes to a file: a full stderr pipe would deadlock the worker.
        self._err = open("/tmp/mb_tokenizer_worker.err", "ab")
        self._proc = subprocess.Popen(
            [str(OMLX_PYTHON), str(_WORKER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._err,
            text=True,
            bufsize=1,
            env=env,
        )

    def ask(self, item: Dict[str, Any]) -> int:
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(json.dumps(item, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("tokenizer worker died; see /tmp/mb_tokenizer_worker.err")
        reply = json.loads(line)
        if "error" in reply:
            raise RuntimeError(f"tokenizer worker error: {reply['error']}")
        return int(reply["count"])

    def close(self) -> None:
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.wait(timeout=5)
        except Exception:  # noqa: BLE001 - best-effort cleanup
            self._proc.kill()
        try:
            self._err.close()
        except Exception:  # noqa: BLE001
            pass


class Tokenizer:
    """Token counter with a disk cache (sha256(payload) -> count).

    Offline mode (default when oMLX.app + the model tokenizer files exist)
    reproduces oMLX ``/v1/tokenize`` counts locally; otherwise the builder falls
    back to the HTTP endpoint.  The cache is shared by both modes and keyed by
    the exact payload, so a rebuild reuses every previously measured count.
    """

    def __init__(self, model: str = TOKEN_MODEL, mode: Optional[str] = None) -> None:
        self.model = model
        self.cache_path = CACHE / "tokenize.json"
        self.cache: Dict[str, int] = {}
        if self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.cache = {}
        self.calls = 0
        self.cache_hits = 0
        wanted = mode or os.environ.get("MB_TOKENIZER_MODE", "auto")
        self.mode = "local" if wanted in ("auto", "local") and local_tokenizer_available() else "omlx"
        self._worker: Optional[_LocalWorker] = None
        self._key = ""
        self._client: Optional[httpx.Client] = None
        if self.mode == "local":
            self._worker = _LocalWorker()
        else:
            self._key = self._load_key()
            self._client = httpx.Client(base_url=OMLX, timeout=60.0)

    @staticmethod
    def _load_key() -> str:
        env = os.environ.get("OMLX_API_KEY")
        if env:
            return env
        settings = Path.home() / ".omlx" / "settings.json"
        if settings.exists():
            data = json.loads(settings.read_text(encoding="utf-8"))
            key = (data.get("auth") or {}).get("api_key")
            if key:
                return key
        raise RuntimeError("no oMLX API key found (OMLX_API_KEY or ~/.omlx/settings.json:auth.api_key)")

    @staticmethod
    def _cache_key(payload: Dict[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()

    def _count_payload(self, payload: Dict[str, Any], item: Dict[str, Any]) -> int:
        key = self._cache_key(payload)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        if self.mode == "local":
            if self._worker is None:  # pragma: no cover - defensive
                raise RuntimeError("local tokenizer worker missing")
            count = self._worker.ask(item)
        else:
            assert self._client is not None
            count = 0
            for attempt in range(3):
                try:
                    r = self._client.post(
                        "/v1/tokenize",
                        json=payload,
                        headers={"Authorization": f"Bearer {self._key}"},
                    )
                    r.raise_for_status()
                    count = int(r.json()["count"])
                    break
                except Exception:  # noqa: BLE001 - transient errors retried
                    if attempt == 2:
                        raise
                    time.sleep(1.0 + attempt)
        self.cache[key] = count
        self.calls += 1
        return count

    def count_text(self, text: str) -> int:
        payload = {"model": self.model, "prompt": text}
        return self._count_payload(payload, {"kind": "text", "text": text})

    def count_messages(
        self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        if tools and self.mode == "local":
            # The builders count API tools as JSON text separately; a chat
            # template with tool definitions is not part of the sample spec.
            raise ValueError("count_messages(tools=...) is not supported offline; count tools as text")
        payload: Dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = tools
        return self._count_payload(payload, {"kind": "messages", "messages": messages})

    def info(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "mode": self.mode,
            "tokenizer_dir": str(TOKENIZER_DIR),
            "semantics": (
                "messages: chat_template(add_generation_prompt=True, enable_thinking=True) + special tokens; "
                "text: encode(add_special_tokens=True); validated against oMLX /v1/tokenize cache"
            ),
        }

    def flush(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")

    def close(self) -> None:
        self.flush()
        if self._worker is not None:
            self._worker.close()
        if self._client is not None:
            self._client.close()


# --------------------------------------------------------------------------- #
# text shaping helpers
# --------------------------------------------------------------------------- #
def _cut_at_sentence(text: str, limit: int) -> str:
    """Cut text at <= limit chars, preferring a boundary close to the limit.

    Only accept a boundary in the last 10% of the window; otherwise the cut is
    dominated by far-away paragraph breaks and the binary search cannot grow.
    """
    if len(text) <= limit:
        return text
    window = text[:limit]
    floor = int(limit * 0.9)
    best = 0
    for sep in ("\n\n", ". ", "。", "\n", " "):
        idx = window.rfind(sep)
        if idx >= floor:
            best = max(best, idx + len(sep))
    return window[:best] if best else window


def fit_text_to_tokens(tok: Tokenizer, text: str, target: int, *, suffix: str = "") -> Tuple[str, int]:
    """Binary-search a character cut so that ``tok(text)`` is closest to target."""
    if not text:
        raise ValueError("empty text")
    lo, hi = 1, min(len(text), max(512, target * 8))
    best, best_n = _cut_at_sentence(text, lo), tok.count_text(_cut_at_sentence(text, lo) + suffix)
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = _cut_at_sentence(text, mid)
        n = tok.count_text(candidate + suffix)
        if abs(n - target) < abs(best_n - target):
            best, best_n = candidate, n
        if n < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return best, best_n


def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", (text or "")).strip().lower()
    for ch in "。，、；：！？.,;:!?\"'`()[]{}":
        text = text.replace(ch, " ")
    return " ".join(text.split())


def last_number(text: str) -> Optional[str]:
    import re

    nums = re.findall(r"-?\d+(?:\.\d+)?", text or "")
    return nums[-1] if nums else None


def last_option_letter(text: str, letters: str = "ABCDEFGHIJ") -> Optional[str]:
    import re

    candidates = re.findall(rf"\b([{letters}])\b", (text or "").upper())
    if candidates:
        return candidates[-1]
    stripped = (text or "").strip().upper()
    if stripped[:1] in letters:
        return stripped[:1]
    return None


# --------------------------------------------------------------------------- #
# streaming json array reader (for large data.json files)
# --------------------------------------------------------------------------- #
def iter_json_array(path: Path, chunk_size: int = 1 << 20):
    """Yield objects from a top-level JSON array without loading it whole."""
    decoder = json.JSONDecoder()
    buf = ""
    started = False
    with path.open(encoding="utf-8") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if chunk:
                buf += chunk
            pos = 0
            while True:
                while pos < len(buf) and buf[pos] in " \t\r\n,":
                    pos += 1
                if not started:
                    if pos < len(buf) and buf[pos] == "[":
                        started = True
                        pos += 1
                        continue
                    break
                if pos < len(buf) and buf[pos] == "]":
                    return
                try:
                    obj, end = decoder.raw_decode(buf, pos)
                except json.JSONDecodeError:
                    break
                yield obj
                pos = end
            buf = buf[pos:]
            if not chunk:
                return


def hf_tree(repo: str, revision: str = "main") -> List[Dict[str, Any]]:
    """List files of a HF dataset repo through the mirror API."""
    url = f"{MIRROR}/api/datasets/{repo}/tree/{revision}?recursive=true"
    with httpx.Client(timeout=60.0, headers=UA, follow_redirects=True) as c:
        r = c.get(url)
        r.raise_for_status()
        return [x for x in r.json() if x.get("type") == "file"]


# --------------------------------------------------------------------------- #
# schema normalisation (BFCL uses Python type names; tools must be JSON Schema)
# --------------------------------------------------------------------------- #
_JSON_SCHEMA_TYPES = {
    "dict": "object",
    "object": "object",
    "float": "number",
    "number": "number",
    "int": "integer",
    "integer": "integer",
    "str": "string",
    "string": "string",
    "bool": "boolean",
    "boolean": "boolean",
    "list": "array",
    "array": "array",
    "tuple": "array",
    "none": "null",
    "null": "null",
}


def normalize_json_schema(node: Any) -> Any:
    """Recursively map non-standard type names onto JSON Schema.

    Keeps every other key untouched; unknown scalar type names are dropped so a
    strict server never receives ``"type": "any"`` style values.
    """
    if isinstance(node, dict):
        out: Dict[str, Any] = {}
        for key, value in node.items():
            if key == "type":
                if isinstance(value, str):
                    mapped = _JSON_SCHEMA_TYPES.get(value.strip().lower())
                    if mapped:
                        out[key] = mapped
                    continue
                if isinstance(value, list):
                    mapped_list = [
                        _JSON_SCHEMA_TYPES.get(str(item).strip().lower(), str(item))
                        for item in value
                    ]
                    out[key] = mapped_list
                    continue
            out[key] = normalize_json_schema(value)
        # "dict" parameters often carry additionalProperties semantics.
        if out.get("type") == "object" and "properties" not in out:
            out.setdefault("properties", {})
        return out
    if isinstance(node, list):
        return [normalize_json_schema(item) for item in node]
    return node


def normalize_tools(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """Return OpenAI tool envelopes with valid JSON Schema parameters."""
    if not tools:
        return None
    out = []
    for tool in tools:
        fn = tool.get("function", tool)
        parameters = normalize_json_schema(
            fn.get("parameters") or {"type": "object", "properties": {}}
        )
        if parameters.get("type") != "object":
            parameters["type"] = "object"
        out.append(
            {
                "type": "function",
                "function": {
                    "name": fn.get("name"),
                    "description": fn.get("description", ""),
                    "parameters": parameters,
                },
            }
        )
    return out


def invalid_schema_types(node: Any, path: str = "") -> List[Tuple[str, Any]]:
    """List every ``type`` value that is not a JSON Schema type (for build asserts)."""
    allowed = {"object", "array", "string", "number", "integer", "boolean", "null", "function"}
    found: List[Tuple[str, Any]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "type":
                if isinstance(value, str) and value not in allowed:
                    found.append((path + "/type", value))
                elif isinstance(value, list):
                    for item in value:
                        if str(item) not in allowed:
                            found.append((path + "/type", item))
            else:
                found.extend(invalid_schema_types(value, path + "/" + str(key)))
    elif isinstance(node, list):
        for idx, item in enumerate(node):
            found.extend(invalid_schema_types(item, path + f"[{idx}]"))
    return found


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #
def provenance_entry(
    source_url: str,
    license_name: str,
    files: List[Tuple[Path, str]],
) -> Dict[str, Any]:
    """Build a manifest provenance record with exact file sizes and SHA-256."""
    entries = []
    for path, logical in files:
        if not path.is_file():
            raise FileNotFoundError(f"provenance input missing: {path}")
        entries.append(
            {
                "logical_path": logical,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {"source_url": source_url, "license": license_name, "files": entries}


# --------------------------------------------------------------------------- #
# LiveCodeBench private tests (zlib + pickle of a JSON string; decoded safely)
# --------------------------------------------------------------------------- #
def decode_lcb_tests(private_test_cases: str) -> List[Dict[str, Any]]:
    """Decode LiveCodeBench ``private_test_cases`` without unpickling.

    Upstream stores ``base64(zlib(pickle(json.dumps(tests))))``.  We walk the
    pickle opcodes with :mod:`pickletools` and only JSON-decode the embedded
    string constant, so no pickle payload is ever executed.
    """
    import base64
    import io
    import pickletools
    import zlib

    payload = zlib.decompress(base64.b64decode(private_test_cases))
    for _op, arg, _pos in pickletools.genops(io.BytesIO(payload)):
        if isinstance(arg, str) and arg.lstrip().startswith("["):
            try:
                tests = json.loads(arg)
            except json.JSONDecodeError:
                continue
            if isinstance(tests, list) and tests and isinstance(tests[0], dict):
                return tests
    raise ValueError("no JSON test list found in private_test_cases")


def load_zip_jsonl(zip_path: Path, member: str) -> List[Dict[str, Any]]:
    """Read one ``.jsonl`` member out of a zip archive."""
    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        raw = zf.read(member).decode("utf-8", "ignore")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]
