"""Tests for the performance test-set builder (``tool/dataset/performance``).

Contract under test (see ``docs/spec/11-performance-testset.md``):

* 8 scenarios x 3 buckets x 10 targets = 240 rows, one fixed target set per scenario;
* a row's ``tokens`` is the **prompt token target** (1024 / 8192 / 32768) and the
  build must hit it exactly with the reference tokenizer (no tolerance band);
* the three rows of one target are strictly nested context prefixes;
* sampling is deterministic (``sha256(seed|id)`` ordering + largest remainder).

The offline-safe cases run everywhere.  Tokenizer-dependent cases (``make_row``,
``fit_prefix``, the built artifacts, the CLI smoke test) are skipped when
``dataset/raw`` has not been downloaded, exactly like the clone-after-fetch flow.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tool.dataset.performance import build


RAW_TOKENIZER = build.RAW_DIR / build.REFERENCE_TOKENIZER_REL
OUT_JSONL = build.OUT_DIR / "testset.jsonl"
OUT_MANIFEST = build.OUT_DIR / "manifest.json"
OUT_YAML = build.OUT_DIR / "testset.yaml"

needs_tokenizer = pytest.mark.skipif(
    not RAW_TOKENIZER.is_file(), reason="dataset/raw reference tokenizer is not downloaded"
)
needs_built_set = pytest.mark.skipif(
    not OUT_JSONL.is_file(), reason="dataset/test/performance/testset.jsonl was not built"
)


# --------------------------------------------------------------------------- #
# constants and shared helpers
# --------------------------------------------------------------------------- #
def test_constants_match_the_design():
    assert build.BUCKETS == {"1k": 1024, "8k": 8192, "32k": 32768}
    assert build.SCENARIOS == (
        "knowledge_qa",
        "instruction_following",
        "long_context",
        "math_logic",
        "coding",
        "general_agent",
        "search_agent",
        "multilingual",
    )
    assert build.TARGETS_PER_SCENARIO == 10
    assert build.OSL == 50 and build.MIN_COMPLETION_TOKENS == 50
    assert build.TASK_MAX_TOKENS == 400
    for bucket, target in build.BUCKETS.items():
        assert build.content_goal(bucket) == target
    assert build.SCENARIO_LANG["knowledge_qa"] == "zh"
    assert build.SCENARIO_LANG["instruction_following"] == "zh"
    assert build.SCENARIO_LANG["multilingual"] == "multi"


def test_content_goal_honours_the_template_reserve(monkeypatch):
    monkeypatch.setattr(build, "CONTENT_RESERVE", 64)
    assert build.content_goal("1k") == 1024 - 64
    assert build.content_goal("32k") == 32768 - 64


def test_sampling_module_is_the_shared_implementation():
    from tool.dataset.performance import sampling

    assert build.deterministic_pick is sampling.deterministic_pick
    assert build.allocate is sampling.allocate


# --------------------------------------------------------------------------- #
# deterministic stratified sampling
# --------------------------------------------------------------------------- #
def test_deterministic_pick_is_deterministic_and_order_independent():
    ids = [f"row-{index:03d}" for index in range(50)]
    first = build.deterministic_pick(ids, 10)
    assert first == build.deterministic_pick(list(reversed(ids)), 10)
    assert first == build.deterministic_pick(ids, 10)
    assert len(first) == 10 and len(set(first)) == 10
    assert build.deterministic_pick(ids, 10, seed=7) != first


def test_deterministic_pick_rejects_oversized_request():
    with pytest.raises(ValueError):
        build.deterministic_pick(["a", "b"], 3)


def test_allocate_uses_largest_remainder_and_respects_pools():
    pools = {"a": 10, "b": 3, "c": 1}
    picked = build.allocate(pools, 10, min_each=1)
    assert sum(picked.values()) == 10
    assert all(0 <= count <= pools[key] for key, count in picked.items())
    assert picked["a"] >= picked["b"] >= picked["c"]


def test_allocate_rejects_k_above_the_pool():
    with pytest.raises(ValueError):
        build.allocate({"a": 1}, 2)


# --------------------------------------------------------------------------- #
# exact alignment primitives
# --------------------------------------------------------------------------- #
@needs_tokenizer
def test_fit_prefix_hits_the_budget_exactly():
    counter = build.Counter(RAW_TOKENIZER)
    units = [
        "Sentence number %d explains why exact token budgets matter for prefill timing."
        % index
        for index in range(2000)
    ]
    for budget in (256, 1024, 8192):
        text, got = build.fit_prefix(counter, units, budget)
        assert got == budget == counter.n(text)


@needs_tokenizer
def test_make_row_is_exact_and_nested():
    counter = build.Counter(RAW_TOKENIZER)
    units = [
        ("Section %d. " % index)
        + ("The quick brown fox jumps over the lazy dog. " * 40)
        for index in range(200)
    ]
    target = {
        "target_id": "tgt-long_context-000",
        "upstream_id": "unit-test|nested",
        "source": "unit-test",
        "units": units,
        "task": "Summarize the material above in at least 150 words.",
        "stream_tokens": 40_000,
        "meta": {"lang": "en"},
    }
    bodies = {}
    for bucket in ("1k", "8k", "32k"):
        row, body = build.make_row(counter, "long_context", target, bucket, 0)
        bodies[bucket] = body
        assert row["tokens"] == build.BUCKETS[bucket]
        assert row["messages_tokens"] == build.BUCKETS[bucket]
        assert row["meta"]["content_tokens"] == build.content_goal(bucket)
        assert row["meta"]["task_sha256"] == build.sha256_text(target["task"])[:16]
        assert counter.n(row["messages"][0]["content"]) == build.BUCKETS[bucket]
        assert row["tools"] == [] and row["tools_tokens"] == 0
    assert bodies["8k"].startswith(bodies["1k"])
    assert bodies["32k"].startswith(bodies["8k"])


def _fake_targets(count: int = build.TARGETS_PER_SCENARIO):
    targets = {}
    for scenario in build.SCENARIOS:
        targets[scenario] = [
            {
                "target_id": f"tgt-{scenario}-{index:03d}",
                "upstream_id": f"synthetic|{scenario}|{index:03d}",
                "source": "synthetic",
                "units": [],
                "task": f"task for {scenario}",
                "stream_tokens": build.STREAM_MIN_TOKENS,
                "meta": {},
            }
            for index in range(count)
        ]
    return targets


def _fake_rows(force_lang: str | None = None):
    rows = []
    bodies = {}
    for scenario in build.SCENARIOS:
        lang = force_lang or build.SCENARIO_LANG[scenario]
        for index in range(build.TARGETS_PER_SCENARIO):
            target_id = f"tgt-{scenario}-{index:03d}"
            task = f"task for {scenario}"
            task_sha = build.sha256_text(task)[:16]
            for bucket in build.BUCKETS:
                body = f"{scenario} body {index} "
                body += {"1k": "1k", "8k": "1k 8k", "32k": "1k 8k 32k"}[bucket]
                bodies[(target_id, bucket)] = body
                rows.append(
                    {
                        "id": f"perf-{scenario}-{bucket}-{index:03d}",
                        "scenario": scenario,
                        "bucket": bucket,
                        "target_id": target_id,
                        "dataset": build.SCENARIO_DATASET[scenario],
                        "tokens": build.BUCKETS[bucket],
                        "messages_tokens": build.BUCKETS[bucket],
                        "tools_tokens": 0,
                        "messages": [{"role": "user", "content": body + task}],
                        "tools": [],
                        "constructed": True,
                        "meta": {
                            "lang": lang,
                            "content_tokens": build.content_goal(bucket),
                            "task_sha256": task_sha,
                        },
                    }
                )
    return rows, bodies


def test_validate_rows_accepts_a_good_set():
    rows, bodies = _fake_rows()
    report = build.validate_rows(rows, _fake_targets(), list(build.BUCKETS), bodies)
    assert report["problems"] == []
    assert report["langs"]["zh"] == 60


def test_validate_rows_rejects_short_cells():
    rows, bodies = _fake_rows()
    report = build.validate_rows(rows[:-3], _fake_targets(), list(build.BUCKETS), bodies)
    assert any("rows != 10" in problem for problem in report["problems"])


def test_validate_rows_rejects_token_drift():
    rows, bodies = _fake_rows()
    rows[0]["tokens"] = 1000
    report = build.validate_rows(rows, _fake_targets(), list(build.BUCKETS), bodies)
    assert any("tokens 1000" in problem for problem in report["problems"])


def test_validate_rows_rejects_content_drift():
    rows, bodies = _fake_rows()
    rows[0]["meta"]["content_tokens"] = 700
    report = build.validate_rows(rows, _fake_targets(), list(build.BUCKETS), bodies)
    assert any("content 700" in problem for problem in report["problems"])


def test_validate_rows_rejects_tools_accounting():
    rows, bodies = _fake_rows()
    rows[0]["tools"] = [{"type": "function", "function": {"name": "search"}}]
    report = build.validate_rows(rows, _fake_targets(), list(build.BUCKETS), bodies)
    assert any("accounting mismatch" in problem for problem in report["problems"])


def test_validate_rows_rejects_unnested_contexts():
    rows, bodies = _fake_rows()
    target_id = rows[0]["target_id"]
    bodies[(target_id, "8k")] = "different 8k body"
    report = build.validate_rows(rows, _fake_targets(), list(build.BUCKETS), bodies)
    assert any("not nested prefixes" in problem for problem in report["problems"])


def test_validate_rows_rejects_missing_chinese():
    rows, bodies = _fake_rows(force_lang="en")
    report = build.validate_rows(rows, _fake_targets(), list(build.BUCKETS), bodies)
    assert "no Chinese sample in the test set" in report["problems"]


# --------------------------------------------------------------------------- #
# built artifacts
# --------------------------------------------------------------------------- #
def _artifact_oracle() -> dict:
    """Resolve the tokenizer/reserve that produced the current ``testset.jsonl``.

    The file is gitignored and may be either the reference build (SHA matches the
    committed manifest) or a pre-run aligned rebuild (SHA matches an alignment
    report under ``dataset/.cache/performance``).  Both are valid; tests must not
    assume one of them -- same rule as ``build.check_outputs``.
    """

    sha = hashlib.sha256(OUT_JSONL.read_bytes()).hexdigest()
    manifest = json.loads(OUT_MANIFEST.read_text(encoding="utf-8"))
    if sha == manifest["rows_sha256"]:
        return {
            "sha256": sha,
            "reserve": int(manifest["construction"]["template_reserve_tokens"]),
            "tokenizer": manifest["tokenizer"]["path"],
            "source": "manifest",
        }
    cache = build.OUT_DIR.parents[1] / ".cache" / "performance"
    for report_path in sorted(cache.glob("alignment-*.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("rows_sha256") == sha:
            return {
                "sha256": sha,
                "reserve": int(report["reserve_tokens"]),
                "tokenizer": report["tokenizer"]["path"],
                "source": report_path.name,
            }
    raise AssertionError(
        "testset.jsonl does not match the manifest nor any alignment report; "
        "run tool/dataset/performance/build.py (reference) or align.py (target model)"
    )


@needs_built_set
def test_built_testset_matches_the_contract():
    oracle = _artifact_oracle()
    rows = [json.loads(line) for line in OUT_JSONL.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(build.SCENARIOS) * len(build.BUCKETS) * build.TARGETS_PER_SCENARIO
    assert len({row["target_id"] for row in rows}) == 80
    cells = {}
    for row in rows:
        cells.setdefault(row["scenario"], {}).setdefault(row["bucket"], 0)
        cells[row["scenario"]][row["bucket"]] += 1
        assert row["tokens"] == build.BUCKETS[row["bucket"]]
        assert row["messages_tokens"] == row["tokens"]
        assert row["tools"] == [] and row["tools_tokens"] == 0
        assert row["meta"]["content_tokens"] == build.BUCKETS[row["bucket"]] - oracle["reserve"]
        assert row["meta"]["template_reserve_tokens"] == oracle["reserve"]
        assert row["meta"]["task_tokens"] <= build.TASK_MAX_TOKENS
        assert len(row["messages"]) == 1 and row["messages"][0]["role"] == "user"
    assert set(cells) == set(build.SCENARIOS)
    assert all(count == 10 for per_bucket in cells.values() for count in per_bucket.values())
    langs = {}
    for row in rows:
        langs[row["meta"]["lang"]] = langs.get(row["meta"]["lang"], 0) + 1
    assert langs["zh"] >= 60
    assert langs["en"] >= 60
    assert len(langs) == 10  # en, zh + the 8 non-Chinese MMLU languages


@needs_built_set
def test_built_manifest_matches_the_rows():
    oracle = _artifact_oracle()
    rows = [json.loads(line) for line in OUT_JSONL.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(OUT_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["name"] == "performance"
    assert manifest["rows"] == len(rows) == 240
    assert oracle["sha256"] == hashlib.sha256(OUT_JSONL.read_bytes()).hexdigest()
    assert manifest["buckets"] == build.BUCKETS
    assert manifest["construction"]["tolerance_tokens"] == 0
    assert manifest["construction"]["prompt_tokens"] == build.BUCKETS
    assert "never a model under test" in manifest["tokenizer"]["role"]
    assert manifest["alignment"]["tolerance_tokens"] == 0
    assert manifest["alignment"]["verified_by"].startswith("tool/dataset/performance/align.py")
    assert manifest["max_tokens"] == build.OSL
    assert manifest["min_completion_tokens_required"] == 50


@needs_built_set
def test_check_outputs_accepts_the_built_set(capsys):
    assert build.check_outputs(build.OUT_DIR) == 0
    out = capsys.readouterr().out
    assert "[check] ok" in out and "tokenizer oracle" in out


@needs_built_set
def test_check_outputs_rejects_token_drift(tmp_path, capsys):
    import json
    import shutil

    for name in ("manifest.json", "testset.jsonl", "testset.yaml"):
        shutil.copy(build.OUT_DIR / name, tmp_path / name)
    rows = [
        json.loads(line)
        for line in (tmp_path / "testset.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows[0]["tokens"] = rows[0]["tokens"] - 1
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    (tmp_path / "testset.jsonl").write_text(payload, encoding="utf-8")

    assert build.check_outputs(tmp_path) == 1
    out = capsys.readouterr().out
    assert "does not match manifest rows_sha256" in out


@needs_built_set
def test_check_outputs_cli_entry(capsys):
    assert build.main(["--check"]) == 0
    assert "[check] ok" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# align.py: pre-run rebuild contract
# --------------------------------------------------------------------------- #
from tool.dataset.performance import align as align_mod  # noqa: E402


@needs_tokenizer
def test_align_requires_model_when_server_mode():
    with pytest.raises(SystemExit) as exc:
        align_mod.main(
            ["--tokenizer", str(RAW_TOKENIZER), "--base-url", "http://127.0.0.1:8000"]
        )
    assert exc.value.code == 2


@needs_tokenizer
def test_align_offline_requires_reserve():
    with pytest.raises(SystemExit) as exc:
        align_mod.main(["--tokenizer", str(RAW_TOKENIZER)])
    assert exc.value.code == 2


def test_align_resolve_api_key_prefers_env(monkeypatch):
    monkeypatch.setenv("OMLX_API_KEY", "env-key")
    assert align_mod.resolve_api_key("http://127.0.0.1:9") == "env-key"
