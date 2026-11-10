#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Derive the HF ``tokenizers`` JSON file from the cl100k_base tiktoken blob.

The performance test set sizes its samples with **cl100k_base**, the generic
OpenAI tokenizer (the industry de-facto reference for token counting), instead
of any model-specific tokenizer.  The Rust ``tokenizers`` library cannot read
the tiktoken ``.tiktoken`` blob directly, so this script converts it once into
an equivalent ``tokenizer.json``; ``fetch.py`` regenerates it deterministically
(byte-identical, SHA-256 pinned in ``raw_spec.json``).

Method (canonical incremental BPE): tiktoken ranks 0-255 are the 256 single
bytes; every rank ``r >= 256`` is the byte string produced by one merge.  For
each such ``r`` in ascending order, the token is segmented using only the
merges of rank ``< r``; the result must be exactly two parts, which become the
merge pair.  A greedy max-rank-first split is NOT equivalent; this incremental
form is the canonical derivation and yields 0 anomalies over the 100,000
merges.  Both vocab keys and merge parts use the GPT-2 ``bytes_to_unicode``
mapping (the same 256-character table the ``ByteLevel`` pre-tokenizer/
post-processor/decoder round-trip through).

The output is verified byte-exact against ``tiktoken.Encoding.encode`` (full
240-row performance test set + edge texts all matched) and against the pinned
SHA-256 below.

Usage (standard library only; needs the tiktoken blob fetched first):

    python tool/dataset/performance/derive_cl100k.py           # write + verify
    python tool/dataset/performance/derive_cl100k.py --check   # verify only
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BLOB_PATH = ROOT / "dataset" / "raw" / "tokenizer" / "cl100k_base" / "cl100k_base.tiktoken"
OUT_PATH = ROOT / "dataset" / "raw" / "tokenizer" / "cl100k_base" / "tokenizer.json"

BLOB_SHA256 = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
OUT_SHA256 = "745dc2f8e2eb601a99627c0bef0c11fcb5bf8b311b74472c9b87184aa7c1a154"
OUT_BYTES = 2_729_984

#: The cl100k pre-tokenizer regex (tiktoken's cl100k_base pattern).  The HF
#: ``ByteLevel`` pre-tokenizer's built-in regex is GPT-2's, which splits text
#: differently, so the pattern is supplied explicitly via ``Split`` and
#: ``ByteLevel`` then maps bytes without re-splitting (``use_regex=False``).
CL100K_PATTERN = (
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|"""
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
)


def bytes_to_unicode() -> dict:
    """The GPT-2 / ByteLevel 256-character byte encoding table."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    chars = printable[:]
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            chars.append(256 + extra)
            extra += 1
    return {byte: chr(char) for byte, char in zip(printable, chars)}


def load_ranks(blob_path: Path) -> list:
    """[(token_bytes, rank)] in ascending rank order (rank 0-255 = bytes)."""
    if not blob_path.is_file():
        raise FileNotFoundError(
            f"tiktoken blob not found: {blob_path}\n"
            "run: python tool/dataset/performance/fetch.py --only tokenizer"
        )
    digest = hashlib.sha256(blob_path.read_bytes()).hexdigest()
    if digest != BLOB_SHA256:
        raise RuntimeError(f"unexpected blob sha256: {digest}")
    ranks = {}
    with blob_path.open("rb") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            token_b64, rank_text = line.split(b" ")
            ranks[base64.b64decode(token_b64)] = int(rank_text)
    if len(ranks) != 100_256:
        raise RuntimeError(f"expected 100,256 ranks, got {len(ranks)}")
    return sorted(ranks.items(), key=lambda item: item[1])


def _segment(token: bytes, merge_ranks: dict) -> list:
    """BPE-segment ``token`` using exactly the given merge rules (min rank)."""
    parts = [bytes([b]) for b in token]
    while len(parts) > 1:
        best_rank = None
        best_index = None
        for index in range(len(parts) - 1):
            rank = merge_ranks.get((parts[index], parts[index + 1]))
            if rank is not None and (best_rank is None or rank < best_rank):
                best_rank = rank
                best_index = index
        if best_index is None:
            break
        parts[best_index : best_index + 2] = [parts[best_index] + parts[best_index + 1]]
    return parts


def build_document(ordered: list) -> dict:
    """The HF ``tokenizers`` JSON document equivalent to the tiktoken blob."""
    b2u = bytes_to_unicode()
    vocab = {}
    for token, rank in ordered:
        vocab["".join(b2u[byte] for byte in token)] = rank

    merge_ranks = {}
    merges = []
    for token, rank in ordered:
        if rank < 256:  # the 256 single bytes are the base vocabulary
            continue
        parts = _segment(token, merge_ranks)
        if len(parts) != 2:
            raise RuntimeError(
                f"rank {rank} does not split into exactly 2 parts: {token[:16]!r}"
            )
        merge_ranks[(parts[0], parts[1])] = rank
        left = "".join(b2u[byte] for byte in parts[0])
        right = "".join(b2u[byte] for byte in parts[1])
        merges.append(f"{left} {right}")
    if len(merges) != 100_000:
        raise RuntimeError(f"expected 100,000 merges, got {len(merges)}")

    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": CL100K_PATTERN},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": False,
                },
            ],
        },
        "post_processor": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": False,
            "use_regex": True,
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": None,
            "vocab": vocab,
            "merges": merges,
        },
    }


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify the output only")
    args = parser.parse_args(argv)

    if args.check:
        if not OUT_PATH.is_file():
            print(f"missing derived tokenizer: {OUT_PATH}", file=sys.stderr)
            return 1
        data = OUT_PATH.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if len(data) != OUT_BYTES or digest != OUT_SHA256:
            print(f"tokenizer.json mismatch: {digest[:12]}", file=sys.stderr)
            return 1
        print("[ok] cl100k tokenizer.json (derived, deterministic)")
        return 0

    ordered = load_ranks(BLOB_PATH)
    document = build_document(ordered)
    payload = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    if len(payload) != OUT_BYTES or digest != OUT_SHA256:
        raise RuntimeError(
            "derived bytes differ from the pinned digest "
            f"({len(payload)} bytes, {digest[:12]}); the derivation is "
            "deterministic, so the blob or the derivation changed"
        )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_bytes(payload)
    print(f"[ok] wrote {OUT_PATH} ({OUT_BYTES} bytes, sha256 {digest[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
