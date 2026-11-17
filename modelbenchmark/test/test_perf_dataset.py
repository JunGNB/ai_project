"""Tests for the performance test-set builder (``tool/dataset/performance``).

Contract under test (see ``docs/spec/11-performance-testset.md``):

* 8 scenarios x 3 buckets (1K/4K/32K) x 10 targets = 240 rows;
* a row carries ``content_goal = 2 x bucket`` reference tokens (primary + filler)
  and declares ``tokens = bucket`` as the run-time prompt-token target;
* the primary part must survive every local model tokenizer and the task block
  (build-time validity), so run-time trimming never cuts into it;
* sampling is deterministic (``sha256(seed|id)`` ordering + largest remainder).

Tokenizer-dependent cases are skipped when ``dataset/raw`` or the built set is
missing, exactly like the clone-after-fetch flow.
"""

from __future__ import annotations

import json
import shutil
from typing import Any, Dict, List

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


class FakeMaxModel:
    """Cheap stand-in for :class:`build.MaxModelCounter` (no model zoo needed)."""

    def tokens(self, text: str, lang: str) -> int:
        return int(len(text) / 3) + 1


def _counter() -> "build.Counter":
    return build.Counter(RAW_TOKENIZER)


# --------------------------------------------------------------------------- #
# constants and pure helpers
# --------------------------------------------------------------------------- #


def test_constants_match_the_documented_contract():
    assert build.BUCKETS == {"1k": 1024, "4k": 4096, "32k": 32768}
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
    assert build.SEED == 42
    assert build.OSL == 50
    assert build.OVERSIZE == 2
    assert build.TEMPLATE_MARGIN == 2


def test_content_goal_is_two_buckets():
    assert build.content_goal("1k") == 2048
    assert build.content_goal("4k") == 8192
    assert build.content_goal("32k") == 65536


def test_scenario_langs_follows_the_design():
    assert build.scenario_langs("knowledge_qa", "1k") == ("en",)
    assert build.scenario_langs("knowledge_qa", "4k") == ("en",)
    assert build.scenario_langs("knowledge_qa", "32k") == ("zh",)
    assert build.scenario_langs("instruction_following", "4k") == ("zh",)
    assert build.scenario_langs("long_context", "32k") == ("en",)
    assert build.scenario_langs("multilingual", "1k") == build.MULTILINGUAL_LANGS


def test_scenario_datasets_use_hotpotqa_for_short_knowledge_and_search():
    assert build.SCENARIO_DATASET[("knowledge_qa", "1k")] == "hotpotqa"
    assert build.SCENARIO_DATASET[("knowledge_qa", "4k")] == "hotpotqa"
    assert build.SCENARIO_DATASET[("knowledge_qa", "32k")].startswith("longbench_pro")
    assert build.SCENARIO_DATASET[("search_agent", "4k")] == "hotpotqa"
    assert build.SCENARIO_DATASET[("coding", "1k")] == "humaneval"


@needs_tokenizer
def test_trim_to_goal_hits_the_goal_within_two_tokens():
    counter = _counter()
    text = "\n\n".join(
        f"Paragraph {i}: the quick brown fox jumps over the lazy dog near the river."
        for i in range(400)
    )
    out = build.trim_to_goal(counter, text, 2048)
    assert abs(counter.n(out) - 2048) <= 2
    assert text.startswith(out)


# --------------------------------------------------------------------------- #
# make_row: body = primary + filler, task kept apart
# --------------------------------------------------------------------------- #


def _target() -> Dict[str, Any]:
    return {
        "target_id": "tgt-knowledge_qa-abc123",
        "scenario": "knowledge_qa",
        "bucket": "1k",
        "lang": "en",
        "source": "hotpotqa",
        "upstream_id": "hotpotqa#abc123",
        "stratum": "all",
        "primary": ["Aurora was born in the small town of Riverton in 1902."],
        "filler": [],
        "task": "Answer the question with evidence. Name the place Aurora was born.",
        "evidence": "Riverton",
        "cut_policy": "paragraph",
        "meta": {"subject": "hotpotqa"},
    }


@needs_tokenizer
def test_make_row_keeps_primary_intact_and_task_separate():
    counter = _counter()
    target = _target()
    target["filler"] = [
        f"Filler paragraph {i}: unrelated background material for the same corpus."
        for i in range(200)
    ]
    row = build.make_row(counter, FakeMaxModel(), target, "1k", 0)

    assert row["id"] == "perf-knowledge_qa-1k-000"
    assert row["bucket"] == "1k"
    assert row["tokens"] == 1024
    assert row["dataset"] == "hotpotqa"
    assert [m["role"] for m in row["messages"]] == ["user", "user"]
    body, task = row["messages"][0]["content"], row["messages"][1]["content"]
    assert body.startswith(target["primary"][0])  # trimming never eats the primary
    assert task == target["task"]
    meta = row["meta"]
    assert meta["primary_complete"] is True
    assert meta["evidence_in_primary"] is True
    assert meta["fits_bucket"] is True
    assert meta["content_tokens"] == meta["body_tokens"] + meta["task_tokens"]
    assert abs(meta["body_tokens"] - build.content_goal("1k")) <= 2
    assert meta["content_tokens"] > build.content_goal("1k")  # body + task block


@needs_tokenizer
def test_make_row_rejects_empty_or_oversized_primary():
    counter = _counter()
    target = _target()
    target["primary"] = [""]
    with pytest.raises(ValueError, match="empty primary"):
        build.make_row(counter, FakeMaxModel(), target, "1k", 0)

    target = _target()
    target["primary"] = ["word " * 5000]
    with pytest.raises(ValueError, match="primary part"):
        build.make_row(counter, FakeMaxModel(), target, "1k", 0)


@needs_tokenizer
def test_make_row_enforces_primary_plus_task_bucket_margin():
    counter = _counter()
    target = _target()
    target["filler"] = ["x " * 8000]
    target["task"] = "y " * 4000  # blows past the 1K bucket even for cheap tokenizers
    with pytest.raises(ValueError, match="exceeds bucket"):
        build.make_row(counter, FakeMaxModel(), target, "1k", 0)


def test_pick_targets_is_deterministic_and_stratified():
    candidates = [
        {"target_id": f"tgt-{name}-{i}", "stratum": name}
        for name in ("a", "b")
        for i in range(6)
    ]
    first = build.pick_targets(candidates, 4)
    second = build.pick_targets(list(reversed(candidates)), 4)
    assert [c["target_id"] for c in first] == [c["target_id"] for c in second]
    assert len({c["stratum"] for c in first}) == 2  # both strata served
    with pytest.raises(ValueError, match="allocate"):
        build.pick_targets(candidates[:2], 4)


# --------------------------------------------------------------------------- #
# validate_rows: rejects every contract break
# --------------------------------------------------------------------------- #


def _synthetic_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scenario in build.SCENARIOS:
        for bucket in build.BUCKETS:
            langs = build.scenario_langs(scenario, bucket)
            for index in range(build.TARGETS_PER_SCENARIO):
                if scenario == "multilingual":
                    lang = build.MULTILINGUAL_LANGS[index]
                else:
                    lang = langs[0]
                goal = build.content_goal(bucket)
                meta = {
                    "lang": lang,
                    "content_tokens": goal,
                    "body_tokens": goal - 100,
                    "task_tokens": 100,
                    "primary_complete": True,
                    "evidence_in_primary": True,
                    "fits_bucket": True,
                    "primary_model_tokens": 10,
                    "task_model_tokens": 10,
                }
                rows.append(
                    {
                        "id": f"perf-{scenario}-{bucket}-{index:03d}",
                        "scenario": scenario,
                        "bucket": bucket,
                        "tokens": build.BUCKETS[bucket],
                        "tools": [],
                        "messages": [
                            {"role": "user", "content": "body"},
                            {"role": "user", "content": "task"},
                        ],
                        "meta": meta,
                    }
                )
    return rows


def test_validate_rows_accepts_the_synthetic_contract():
    build.validate_rows(_synthetic_rows(), [])


def test_validate_rows_rejects_missing_cells_and_short_cells():
    rows = _synthetic_rows()
    with pytest.raises(ValueError, match="cells mismatch"):
        build.validate_rows(rows[:-10], [])
    with pytest.raises(ValueError, match="rows != 10"):
        build.validate_rows(rows[1:], [])


def test_validate_rows_rejects_token_and_accounting_drift():
    rows = _synthetic_rows()
    rows[0]["tokens"] = 2048
    with pytest.raises(ValueError, match="tokens != bucket"):
        build.validate_rows(rows, [])

    rows = _synthetic_rows()
    rows[0]["meta"]["content_tokens"] = 1
    with pytest.raises(ValueError, match="content_tokens"):
        build.validate_rows(rows, [])

    rows = _synthetic_rows()
    rows[0]["meta"]["body_tokens"] = 1
    with pytest.raises(ValueError, match="accounting"):
        build.validate_rows(rows, [])


def test_validate_rows_rejects_shape_flags_and_language():
    rows = _synthetic_rows()
    rows[0]["tools"] = [{"name": "t"}]
    with pytest.raises(ValueError, match="two user messages"):
        build.validate_rows(rows, [])

    rows = _synthetic_rows()
    rows[0]["messages"] = [{"role": "system", "content": "x"}]
    with pytest.raises(ValueError, match="two user messages"):
        build.validate_rows(rows, [])

    rows = _synthetic_rows()
    rows[0]["meta"]["fits_bucket"] = False
    with pytest.raises(ValueError, match="do not fit"):
        build.validate_rows(rows, [])

    rows = _synthetic_rows()
    rows[0]["meta"]["lang"] = "de"
    with pytest.raises(ValueError, match="unexpected language"):
        build.validate_rows(rows, [])


# --------------------------------------------------------------------------- #
# built artifacts
# --------------------------------------------------------------------------- #


@needs_built_set
def test_built_testset_matches_the_contract():
    rows = [
        json.loads(line)
        for line in OUT_JSONL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 240
    cells = {(row["scenario"], row["bucket"]) for row in rows}
    assert cells == {(s, b) for s in build.SCENARIOS for b in build.BUCKETS}
    for bucket in build.BUCKETS:
        cell_rows = [row for row in rows if row["bucket"] == bucket]
        assert len(cell_rows) == 80
        assert {row["tokens"] for row in cell_rows} == {build.BUCKETS[bucket]}
        for row in cell_rows:
            meta = row["meta"]
            assert meta["primary_complete"] is True
            assert meta["fits_bucket"] is True
            assert isinstance(meta["primary_share"], (int, float))
    assert sum(1 for row in rows if row["meta"]["lang"] == "zh") >= 40
    assert {row["meta"]["lang"] for row in rows} >= set(build.MULTILINGUAL_LANGS)


@needs_built_set
def test_built_manifest_matches_the_rows():
    manifest = json.loads(OUT_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["name"] == "performance"
    assert manifest["buckets"] == {"1k": 1024, "4k": 4096, "32k": 32768}
    assert manifest["max_tokens"] == 50
    construction = manifest["construction"]
    assert construction["oversize"] == 2
    assert construction["content_tokens"] == {"1k": 2048, "4k": 8192, "32k": 65536}
    runtime = construction["runtime_alignment"]
    assert runtime["mode"] == "trim+pad"
    assert runtime["tolerance_tokens"] == 0
    assert runtime["skipped_on_mismatch"] is False
    assert manifest["rows"] == 240
    assert sum(int(cell["rows"]) for cell in manifest["cells"].values()) == 240
    # The committed file is either the reference build or a per-model aligned build;
    # both are valid as long as the SHA matches the manifest or an alignment report.
    rows_sha = build.sha256_file(OUT_JSONL)
    report_shas = set()
    for path in sorted(
        (build.PROJECT_ROOT / "dataset" / ".cache" / "performance").glob("alignment-*.json")
    ):
        try:
            report_shas.add(json.loads(path.read_text(encoding="utf-8"))["rows_sha256"])
        except Exception:  # pragma: no cover - unreadable cache entry
            continue
    assert rows_sha == manifest["rows_sha256"] or rows_sha in report_shas


@needs_built_set
def test_built_yaml_declares_buckets_alignment_and_generation():
    import yaml

    data = yaml.safe_load(OUT_YAML.read_text(encoding="utf-8"))
    assert data["name"] == "performance"
    assert data["buckets"] == {"1k": 1024, "4k": 4096, "32k": 32768}
    assert data["max_tokens"] == 50
    assert data["strict_input_tokens"] is True
    assert "osl" not in data
    assert len(data["scenarios"]) == 8
    for scenario, spec in data["scenarios"].items():
        assert spec["dataset"] == "perf_jsonl"
        assert spec["dataset_params"]["scenario"] == scenario
        assert spec["dataset_params"]["path"].endswith("dataset/test/performance/testset.jsonl")


@needs_built_set
def test_check_outputs_accepts_the_built_set():
    build.check_outputs(build.OUT_DIR)


@needs_built_set
def test_check_outputs_rejects_token_drift(tmp_path):
    shutil.copy(OUT_JSONL, tmp_path / "testset.jsonl")
    shutil.copy(OUT_MANIFEST, tmp_path / "manifest.json")
    shutil.copy(OUT_YAML, tmp_path / "testset.yaml")
    rows = [
        json.loads(line)
        for line in (tmp_path / "testset.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows[0]["tokens"] = 2048
    (tmp_path / "testset.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        build.check_outputs(tmp_path)


def test_check_reports_missing_directory(tmp_path):
    with pytest.raises(Exception):
        build.main(["--check", "--out-dir", str(tmp_path / "absent")])
