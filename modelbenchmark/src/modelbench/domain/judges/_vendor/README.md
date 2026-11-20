# Vendored judge assets

Everything here ships inside the installed package (wheel ≈ 0.66 MB) so
`method=accuracy` can judge offline, on a fresh machine, with no extra download.

## `instruction_following_eval/`

Official Google IFEval instruction-following checker, vendored verbatim so the
`instruction_following` scenario still works when the offline
`dataset/raw/ifeval` assets were never fetched.

- Upstream: https://github.com/google-research/google-research/tree/master/instruction_following_eval
- License: Apache License 2.0 (per-file headers retained)
- Files: `evaluation_lib.py`, `evaluation_main.py`, `instructions.py`,
  `instructions_registry.py`, `instructions_util.py`, `__init__.py`
- Runtime deps: `absl-py`, `nltk`, `langdetect`, `immutabledict` (core
  dependencies in `pyproject.toml`) plus the NLTK `punkt`/`punkt_tab` data below.

Resolution order in `modelbench/domain/judges/instruction.py`:

1. `dataset/raw/ifeval/instruction_following_eval` (or `$MB_RAW_DIR/ifeval/...`) —
   lets a checkout override/pin the checker;
2. this vendored copy — always present once the package is installed.

A candidate only counts as usable when `instruction_following_eval/evaluation_lib.py`
exists, so a half-finished download is never mistaken for a checker.

## `nltk_data/`

English-only subset of the NLTK punkt tokenizer data, needed by the checker:

- `tokenizers/punkt/english.pickle` (≈ 423 KB)
- `tokenizers/punkt_tab/english/{abbrev_types.txt,collocations.tab,ortho_context.tab,sent_starters.txt}`
  (≈ 244 KB)

The checker's word counting uses `nltk.tokenize.RegexpTokenizer(r"\w+")`, so only
the **sentence** tokenizer data is required; that is why the full 18 MB
multi-language `punkt.zip`/`punkt_tab.zip` are not vendored.

- Upstream: https://github.com/nltk/nltk_data (`packages/tokenizers/punkt.zip`,
  `packages/tokenizers/punkt_tab.zip`), extracted English subset.
- License: Apache License 2.0 (NLTK data packages).

Resolution order in `_ensure_nltk_data()`:

1. `dataset/raw/nltk_data` when it already contains `tokenizers/punkt`
   (`$MB_RAW_DIR/nltk_data`, e.g. a pinned full copy);
2. this vendored English subset (sets `NLTK_DATA` to this directory);
3. NLTK's own default search path (`~/nltk_data`, `/usr/local/share/nltk_data`,
   `/usr/share/nltk_data`).

`judges.missing_assets(["ifeval"])` reports whichever of the checker / punkt data
cannot be found, and the accuracy run + `validate` preflight surface that list
instead of silently scoring 0.

## Refreshing

- Checker code: re-download the five upstream files, keep the Apache headers,
  and `sha256sum` them against `tool/dataset/accuracy/raw_spec.json`. The
  `fetch.py --only ifeval` path writes verified copies under `dataset/raw/ifeval`
  (that copy wins over this one when present).
- punkt data: `python tool/dataset/accuracy/fetch.py --only nltk` downloads and
  extracts the full multi-language archives; copy only `punkt/english.pickle`
  and `punkt_tab/english/` from there into this directory.
