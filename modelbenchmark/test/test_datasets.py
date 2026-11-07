"""Offline tests for the dataset download/normalisation layer.

No network access is performed here: tiny synthetic raw files are written into
``tmp_path`` (parquet via pyarrow, CSV/JSONL with plain Python) and the adapters
are exercised against them.  Download plumbing is covered by monkeypatching
``modelbench.adapters.datasets._download._fetch`` so retry/resume/reuse logic runs offline too.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from modelbench.adapters.datasets._download import (
    MANIFEST_NAME,
    DownloadError,
    download_file,
    list_raw_files,
    read_manifest,
    sha256_file,
)
from modelbench.adapters.datasets import _download
from modelbench.adapters.datasets.bfcl import BFCLAdapter, parse_ground_truth
from modelbench.adapters.datasets.gsm8k import GSM8KAdapter, extract_gold_number
from modelbench.adapters.datasets.hotpotqa import HotpotQAAdapter
from modelbench.adapters.datasets.humaneval import HumanEvalAdapter
from modelbench.adapters.datasets.ifeval import IFEvalAdapter
from modelbench.adapters.datasets.mmlu import DEFAULT_SUBJECTS, MMLUAdapter
from modelbench.adapters.datasets.mmmlu import LANGUAGES, MMMLUAdapter, normalize_language
from modelbench.domain.models import AnswerType, Sample

try:  # pragma: no cover - optional dependency
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pa = None
    pq = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _require_pyarrow() -> None:
    if pq is None:  # pragma: no cover - environment without pyarrow
        pytest.skip("pyarrow is not installed")


def write_parquet(path: Path, rows: list[dict], schema=None) -> None:
    _require_pyarrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def assert_deterministic(adapter) -> list[Sample]:
    first = adapter.load()
    second = adapter.load()
    assert [s.model_dump(mode="json") for s in first] == [
        s.model_dump(mode="json") for s in second
    ]
    return first


@pytest.fixture
def dirs(tmp_path: Path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    processed.mkdir()
    return raw, processed


# --------------------------------------------------------------------------- #
# download helper (offline, monkeypatched transport)
# --------------------------------------------------------------------------- #


def test_download_file_records_sha_and_reuses_without_network(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw"
    dest = raw_root / "all" / "data.parquet"
    payload = b"hello dataset" * 100

    calls: list[str] = []

    def fake_fetch(url, path, **_kwargs):
        calls.append(url)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(payload)
        return Path(path)

    monkeypatch.setattr(_download, "_fetch", fake_fetch)
    download_file("https://example.invalid/data.parquet", dest, raw_root=raw_root)

    assert calls == ["https://example.invalid/data.parquet"]
    manifest = read_manifest(raw_root)
    entry = manifest["files"]["all/data.parquet"]
    assert entry["size"] == len(payload)
    assert entry["sha256"] == sha256_file(dest)
    assert entry["url"] == "https://example.invalid/data.parquet"

    # Second call: manifest + file match -> no network at all.
    def boom(*_args, **_kwargs):  # pragma: no cover - must not be called
        raise AssertionError("network used although the file was verified")

    monkeypatch.setattr(_download, "_fetch", boom)
    download_file("https://example.invalid/data.parquet", dest, raw_root=raw_root)
    assert dest.read_bytes() == payload


def test_download_file_redownloads_on_checksum_mismatch(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw"
    dest = raw_root / "file.bin"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"corrupted")
    (raw_root / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "raw",
                "files": {"file.bin": {"url": "u", "size": 9, "sha256": "0" * 64}},
            }
        ),
        encoding="utf-8",
    )

    calls: list[str] = []

    def fake_fetch(url, path, **_kwargs):
        calls.append(url)
        Path(path).write_bytes(b"good bytes")
        return Path(path)

    monkeypatch.setattr(_download, "_fetch", fake_fetch)
    download_file("https://example.invalid/file.bin", dest, raw_root=raw_root)

    assert calls == ["https://example.invalid/file.bin"]
    assert dest.read_bytes() == b"good bytes"
    manifest = read_manifest(raw_root)
    assert manifest["files"]["file.bin"]["sha256"] == sha256_file(dest)


def test_list_raw_files_skips_manifest(tmp_path):
    raw_root = tmp_path / "raw"
    (raw_root / "sub").mkdir(parents=True)
    (raw_root / "sub" / "a.txt").write_text("a", encoding="utf-8")
    (raw_root / MANIFEST_NAME).write_text("{}", encoding="utf-8")
    (raw_root / "sub" / "b.part").write_text("partial", encoding="utf-8")

    rows = list_raw_files(raw_root)
    assert [row["relative_path"] for row in rows] == ["sub/a.txt"]
    assert rows[0]["sha256"] == sha256_file(raw_root / "sub" / "a.txt")


# --------------------------------------------------------------------------- #
# MMLU
# --------------------------------------------------------------------------- #


def _write_mmlu_fixture(raw: Path) -> Path:
    rows: list[dict] = []
    for subject in DEFAULT_SUBJECTS:
        for index in range(20):
            rows.append(
                {
                    "question": f"{subject} question {index}",
                    "subject": subject,
                    "choices": [f"{index}-a", f"{index}-b", f"{index}-c", f"{index}-d"],
                    "answer": index % 4,
                }
            )
    rows.append(
        {
            "question": "virology extra",
            "subject": "virology",
            "choices": ["x0", "x1", "x2", "x3"],
            "answer": 2,
        }
    )
    path = raw / "mmlu" / "all" / "test-00000-of-00001.parquet"
    write_parquet(
        path,
        rows,
        schema=pa.schema(
            [
                ("question", pa.string()),
                ("subject", pa.string()),
                ("choices", pa.list_(pa.string())),
                ("answer", pa.int64()),
            ]
        ),
    )
    return path


def test_mmlu_default_load_and_field_mapping(dirs):
    raw, processed = dirs
    _write_mmlu_fixture(raw)
    adapter = MMLUAdapter(raw / "mmlu", processed, {})

    samples = assert_deterministic(adapter)
    assert len(samples) == len(DEFAULT_SUBJECTS) * 20

    first = samples[0]
    assert first.sample_id == "mmlu-abstract_algebra-00000"
    assert first.dataset == "mmlu"
    assert first.scenario == "knowledge_qa"
    assert first.subtask == "abstract_algebra"
    assert first.answer_type is AnswerType.MC_LETTER
    assert first.answer == "A"
    assert first.choices == ["0-a", "0-b", "0-c", "0-d"]
    assert first.metadata["subject"] == "abstract_algebra"
    # every answer is a valid letter and ids are unique
    assert all(sample.answer in "ABCD" for sample in samples)
    assert len({sample.sample_id for sample in samples}) == len(samples)


def test_mmlu_subject_filter_per_subject_and_limit(dirs):
    raw, processed = dirs
    _write_mmlu_fixture(raw)

    adapter = MMLUAdapter(
        raw / "mmlu", processed, {"subjects": ["virology"], "per_subject": 1, "limit": 1}
    )
    samples = adapter.load()
    assert len(samples) == 1
    assert samples[0].sample_id == "mmlu-virology-00240"
    assert samples[0].answer == "C"
    assert samples[0].metadata["row_index"] == 240

    # unknown subject must fail loudly instead of silently returning nothing
    with pytest.raises(ValueError):
        MMLUAdapter(raw / "mmlu", processed, {"subjects": ["not_a_subject"]}).load()


def test_mmlu_explicit_null_subjects_means_all(dirs):
    raw, processed = dirs
    _write_mmlu_fixture(raw)
    # scenes.yaml passes `subjects: null` -> every subject in the file
    samples = MMLUAdapter(raw / "mmlu", processed, {"subjects": None, "limit": 50}).load()
    assert len(samples) == 50
    assert "virology" in {sample.subtask for sample in samples}


def test_mmlu_limit_is_round_robin_over_subjects(dirs):
    raw, processed = dirs
    _write_mmlu_fixture(raw)
    samples = MMLUAdapter(
        raw / "mmlu",
        processed,
        {"subjects": ["abstract_algebra", "college_physics"], "per_subject": 2, "limit": 3},
    ).load()
    assert [sample.subtask for sample in samples] == [
        "abstract_algebra",
        "college_physics",
        "abstract_algebra",
    ]
    assert [sample.sample_id for sample in samples] == [
        "mmlu-abstract_algebra-00000",
        "mmlu-college_physics-00080",
        "mmlu-abstract_algebra-00001",
    ]


def test_mmlu_processed_cache_roundtrip(dirs):
    raw, processed = dirs
    _write_mmlu_fixture(raw)
    adapter = MMLUAdapter(raw / "mmlu", processed, {"subjects": ["philosophy"], "per_subject": 2})
    samples = adapter.load()
    path = adapter.save_processed(samples)
    assert path == processed / "mmlu.jsonl"

    reloaded = adapter.load_processed()
    assert reloaded is not None
    assert [s.model_dump(mode="json") for s in reloaded] == [
        s.model_dump(mode="json") for s in samples
    ]
    assert adapter.name == "mmlu" and adapter.scenario == "knowledge_qa"


# --------------------------------------------------------------------------- #
# GSM8K
# --------------------------------------------------------------------------- #


def _write_gsm8k_fixture(raw: Path) -> Path:
    path = raw / "gsm8k" / "main" / "test-00000-of-00001.parquet"
    write_parquet(
        path,
        [
            {"question": "q1", "answer": "2+2=4\n#### 18"},
            {"question": "q2", "answer": "long\n#### 1,450,000"},
            {"question": "q3", "answer": "steps\n#### -3"},
        ],
        schema=pa.schema([("question", pa.string()), ("answer", pa.string())]),
    )
    return path


def test_gsm8k_load(dirs):
    raw, processed = dirs
    _write_gsm8k_fixture(raw)
    adapter = GSM8KAdapter(raw / "gsm8k", processed, {"limit": 2})
    samples = assert_deterministic(adapter)

    assert len(samples) == 2
    first, second = samples
    assert first.sample_id == "gsm8k-00000"
    assert first.dataset == "gsm8k"
    assert first.scenario == "math_logic"
    assert first.answer_type is AnswerType.NUMERIC
    assert "#### 18" in first.answer
    assert first.metadata["gold_number"] == "18"
    assert second.metadata["gold_number"] == "1450000"


def test_extract_gold_number_variants():
    assert extract_gold_number("reasoning\n#### 18") == "18"
    assert extract_gold_number("reasoning\n#### 1,450,000") == "1450000"
    assert extract_gold_number("reasoning\n#### $42") == "42"
    assert extract_gold_number("x\n#### -3") == "-3"
    with pytest.raises(ValueError):
        extract_gold_number("no marker at all")
    with pytest.raises(ValueError):
        extract_gold_number("#### not-a-number")


# --------------------------------------------------------------------------- #
# IFEval
# --------------------------------------------------------------------------- #


def _write_ifeval_fixture(raw: Path) -> Path:
    path = raw / "ifeval" / "ifeval_input_data.jsonl"
    write_jsonl(
        path,
        [
            {"key": 0, "prompt": "p0", "instruction_id_list": ["alpha"], "kwargs": [{"k": 1}]},
            {"key": 1, "prompt": "p1", "instruction_id_list": ["beta"], "kwargs": [{}]},
            {
                "key": 2,
                "prompt": "p2",
                "instruction_id_list": ["alpha", "beta"],
                "kwargs": [{"k": 2}, {}],
            },
            {"key": 3, "prompt": "p3", "instruction_id_list": ["gamma"], "kwargs": []},
        ],
    )
    return path


def test_ifeval_load_and_metadata(dirs):
    raw, processed = dirs
    _write_ifeval_fixture(raw)
    adapter = IFEvalAdapter(raw / "ifeval", processed, {})
    samples = assert_deterministic(adapter)

    assert len(samples) == 4  # default limit 30 > 4 available rows
    first = samples[0]
    assert first.sample_id == "ifeval-0000"
    assert first.question == "p0"
    assert first.answer is None
    assert first.answer_type is AnswerType.CONSTRAINTS
    assert first.metadata["instruction_id_list"] == ["alpha"]
    assert first.metadata["kwargs"] == [{"k": 1}]
    assert first.dataset == "ifeval" and first.scenario == "instruction_following"


def test_ifeval_uniform_sampling_across_instruction_types(dirs):
    raw, processed = dirs
    _write_ifeval_fixture(raw)

    two = IFEvalAdapter(raw / "ifeval", processed, {"limit": 2}).load()
    assert [sample.sample_id for sample in two] == ["ifeval-0000", "ifeval-0001"]

    three = IFEvalAdapter(raw / "ifeval", processed, {"limit": 3}).load()
    assert [sample.sample_id for sample in three] == ["ifeval-0000", "ifeval-0001", "ifeval-0003"]
    # row 2 belongs to alpha+beta but is never picked twice thanks to dedupe
    assert len({sample.sample_id for sample in three}) == 3

    filtered = IFEvalAdapter(raw / "ifeval", processed, {"instructions": ["gamma"]}).load()
    assert [sample.sample_id for sample in filtered] == ["ifeval-0003"]


# --------------------------------------------------------------------------- #
# HumanEval
# --------------------------------------------------------------------------- #


def _write_humaneval_fixture(raw: Path) -> Path:
    path = raw / "humaneval" / "openai_humaneval" / "test-00000-of-00001.parquet"
    write_parquet(
        path,
        [
            {
                "task_id": "HumanEval/0",
                "prompt": "def f():\n    pass\n",
                "canonical_solution": "    return 1\n",
                "test": "def check(candidate):\n    assert candidate() == 1\n",
                "entry_point": "f",
            },
            {
                "task_id": "HumanEval/10",
                "prompt": "def g():\n    pass\n",
                "canonical_solution": "    return 2\n",
                "test": "def check(candidate):\n    assert candidate() == 2\n",
                "entry_point": "g",
            },
        ],
        schema=pa.schema(
            [
                ("task_id", pa.string()),
                ("prompt", pa.string()),
                ("canonical_solution", pa.string()),
                ("test", pa.string()),
                ("entry_point", pa.string()),
            ]
        ),
    )
    return path


def test_humaneval_load(dirs):
    raw, processed = dirs
    _write_humaneval_fixture(raw)
    samples = assert_deterministic(HumanEvalAdapter(raw / "humaneval", processed, {}))

    assert len(samples) == 2
    first = samples[0]
    assert first.sample_id == "humaneval-000"
    assert samples[1].sample_id == "humaneval-010"
    assert first.dataset == "humaneval" and first.scenario == "coding"
    assert first.answer is None
    assert first.answer_type is AnswerType.CODE
    assert first.question == "def f():\n    pass\n"
    assert first.metadata["task_id"] == "HumanEval/0"
    assert first.metadata["entry_point"] == "f"
    assert "assert candidate() == 1" in first.metadata["test"]
    assert first.metadata["canonical_solution"] == "    return 1\n"
    assert first.metadata["prompt"] == first.question


# --------------------------------------------------------------------------- #
# MMMLU
# --------------------------------------------------------------------------- #


def _write_mmmlu_fixture(raw: Path, language: str, rows: list[list[str]]) -> Path:
    path = raw / "mmmlu" / "test" / f"mmlu_{language}.csv"
    write_csv(path, ["", "Question", "A", "B", "C", "D", "Answer", "Subject"], rows)
    return path


def _mmmlu_rows(prefix: str) -> list[list[str]]:
    return [
        ["0", f"{prefix} q0", "a0", "b0", "c0", "d0", "B", "abstract_algebra"],
        ["1", f"{prefix} q1", "a1", "b1", "c1", "d1", "D", "philosophy"],
        ["2", f"{prefix} q2", "a2", "b2", "c2", "d2", "1", "sociology"],
    ]


def test_mmmlu_load_and_language_metadata(dirs):
    raw, processed = dirs
    _write_mmmlu_fixture(raw, "ZH-CN", _mmmlu_rows("zh"))
    _write_mmmlu_fixture(raw, "JA-JP", _mmmlu_rows("ja"))

    adapter = MMMLUAdapter(
        raw / "mmmlu", processed, {"languages": ["ZH-CN"], "per_language": 2}
    )
    samples = assert_deterministic(adapter)
    assert len(samples) == 2
    first = samples[0]
    assert first.sample_id == "mmmlu-ZH-CN-0000"
    assert first.dataset == "mmmlu"
    assert first.scenario == "multilingual"
    assert first.subtask == "ZH-CN"
    assert first.metadata["language"] == "ZH-CN"
    assert first.metadata["subject"] == "abstract_algebra"
    assert first.choices == ["a0", "b0", "c0", "d0"]
    assert first.answer == "B"
    assert first.answer_type is AnswerType.MC_LETTER


def test_mmmlu_aliases_order_and_numeric_answer(dirs):
    raw, processed = dirs
    # write JA first on disk to prove load order follows params, not filesystem
    _write_mmmlu_fixture(raw, "JA-JP", _mmmlu_rows("ja"))
    _write_mmmlu_fixture(raw, "ZH-CN", _mmmlu_rows("zh"))

    samples = MMMLUAdapter(raw / "mmmlu", processed, {"languages": ["zh", "ja"]}).load()
    assert [sample.metadata["language"] for sample in samples] == [
        "ZH-CN",
        "ZH-CN",
        "ZH-CN",
        "JA-JP",
        "JA-JP",
        "JA-JP",
    ]
    assert samples[2].answer == "B"  # numeric "1" -> index 1 -> "B"

    assert normalize_language("fr") == "FR-FR"
    with pytest.raises(ValueError):
        normalize_language("xx-YY")


def test_mmmlu_limit_per_language_alias(dirs):
    raw, processed = dirs
    _write_mmmlu_fixture(raw, "ZH-CN", _mmmlu_rows("zh"))
    _write_mmmlu_fixture(raw, "JA-JP", _mmmlu_rows("ja"))
    # scenes.yaml uses `limit_per_language`
    samples = MMMLUAdapter(
        raw / "mmmlu",
        processed,
        {"languages": ["ZH-CN", "JA-JP"], "limit_per_language": 2},
    ).load()
    assert len(samples) == 4
    assert [sample.metadata["language"] for sample in samples] == ["ZH-CN", "ZH-CN", "JA-JP", "JA-JP"]


def test_mmmlu_missing_file_raises_clear_error(dirs):
    raw, processed = dirs
    _write_mmmlu_fixture(raw, "ZH-CN", _mmmlu_rows("zh"))
    adapter = MMMLUAdapter(raw / "mmmlu", processed, {"languages": ["ZH-CN", "JA-JP"]})
    with pytest.raises(FileNotFoundError) as excinfo:
        adapter.load()
    assert "mmlu_JA-JP.csv" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# HotpotQA
# --------------------------------------------------------------------------- #


def _write_hotpotqa_fixture(raw: Path) -> Path:
    path = raw / "hotpotqa" / "distractor" / "validation-00000-of-00001.parquet"
    rows = []
    for index in range(3):
        rows.append(
            {
                "id": f"id-{index}",
                "question": f"question {index}?",
                "answer": f"answer-{index}",
                "type": "comparison",
                "level": "hard",
                "supporting_facts": {
                    "title": [f"T{index}a", f"T{index}b"],
                    "sent_id": [0, 1],
                },
                "context": {
                    "title": [f"T{index}a", f"T{index}b"],
                    "sentences": [
                        ["first sentence.", "second sentence."],
                        ["other text."],
                    ],
                },
            }
        )
    _require_pyarrow()
    schema = pa.schema(
        [
            ("id", pa.string()),
            ("question", pa.string()),
            ("answer", pa.string()),
            ("type", pa.string()),
            ("level", pa.string()),
            (
                "supporting_facts",
                pa.struct([("title", pa.list_(pa.string())), ("sent_id", pa.list_(pa.int32()))]),
            ),
            (
                "context",
                pa.struct(
                    [
                        ("title", pa.list_(pa.string())),
                        ("sentences", pa.list_(pa.list_(pa.string()))),
                    ]
                ),
            ),
        ]
    )
    write_parquet(path, rows, schema=schema)
    return path


def test_hotpotqa_load_corpus_and_metadata(dirs):
    raw, processed = dirs
    _write_hotpotqa_fixture(raw)
    samples = assert_deterministic(HotpotQAAdapter(raw / "hotpotqa", processed, {"limit": 2}))

    assert len(samples) == 2
    first = samples[0]
    assert first.sample_id == "hotpotqa-id-0"
    assert first.dataset == "hotpotqa"
    assert first.scenario == "search_agent"
    assert first.question == "question 0?"
    assert first.answer == "answer-0"
    assert first.answer_type is AnswerType.SEARCH_ANSWER
    assert first.context == (
        "T0a: first sentence. second sentence.\n\nT0b: other text."
    )
    assert first.metadata["paragraphs"] == [
        ["T0a", "first sentence. second sentence."],
        ["T0b", "other text."],
    ]
    assert first.metadata["supporting_facts"] == [
        {"title": "T0a", "sent_id": 0},
        {"title": "T0b", "sent_id": 1},
    ]
    assert first.metadata["level"] == "hard"


# --------------------------------------------------------------------------- #
# BFCL
# --------------------------------------------------------------------------- #


def _bfcl_question(category_file: str, rows: list[dict]) -> None:
    write_jsonl(category_file, rows)


def _write_bfcl_fixture(raw: Path) -> Path:
    bfcl = raw / "bfcl"
    bfcl.mkdir(parents=True)

    _bfcl_question(
        bfcl / "BFCL_v4_simple_python.json",
        [
            {
                "id": "simple_python_0",
                "question": [[{"role": "user", "content": "Find the area"}]],
                "function": [
                    {
                        "name": "calculate_area",
                        "description": "area",
                        "parameters": {
                            "type": "object",
                            "properties": {"base": {"type": "integer"}},
                        },
                    }
                ],
            },
            {
                "id": "simple_python_1",
                "question": [[{"role": "user", "content": "Second request"}]],
                "function": [{"name": "noop", "description": "", "parameters": {}}],
            },
        ],
    )
    _bfcl_question(
        bfcl / "BFCL_v4_multiple.json",
        [
            {
                "id": "multiple_0",
                "question": [[{"role": "user", "content": "Multiple request"}]],
                "function": [
                    {"name": "f1", "description": "", "parameters": {}},
                    {"name": "f2", "description": "", "parameters": {}},
                ],
            }
        ],
    )
    _bfcl_question(
        bfcl / "BFCL_v4_parallel.json",
        [
            {
                "id": "parallel_0",
                "question": [[{"role": "user", "content": "Parallel request"}]],
                "function": [{"name": "play", "description": "", "parameters": {}}],
            }
        ],
    )
    _bfcl_question(
        bfcl / "BFCL_v4_multi_turn_base.json",
        [
            {
                "id": "multi_turn_base_0",
                "question": [
                    [{"role": "user", "content": "Move the file"}],
                    [{"role": "user", "content": "Now search it"}],
                ],
                "initial_config": {"path": []},
                "involved_classes": ["GorillaFileSystem"],
                "path": [],
            }
        ],
    )

    gt_dir = bfcl / "possible_answer"
    gt_dir.mkdir()
    write_jsonl(
        gt_dir / "BFCL_v4_simple_python.json",
        [
            {
                "id": "simple_python_0",
                "ground_truth": [{"calculate_area": {"base": [10], "unit": ["units", ""]}}],
            },
            {
                "id": "simple_python_1",
                "ground_truth": [{"noop": {"x": [1]}}],
            },
        ],
    )
    write_jsonl(
        gt_dir / "BFCL_v4_multiple.json",
        [
            {
                "id": "multiple_0",
                "ground_truth": [{"f1": {"x": [1]}}, {"f2": {"y": ["a"]}}],
            }
        ],
    )
    write_jsonl(
        gt_dir / "BFCL_v4_parallel.json",
        [
            {
                "id": "parallel_0",
                "ground_truth": [{"play": {"artist": ["A"]}}, {"play": {"artist": ["B"]}}],
            }
        ],
    )
    write_jsonl(
        gt_dir / "BFCL_v4_multi_turn_base.json",
        [
            {
                "id": "multi_turn_base_0",
                "ground_truth": [
                    ["cd(folder='document')", "mkdir(dir_name='temp')"],
                    ["sort('final_report.pdf')"],
                ],
            }
        ],
    )
    return bfcl


def test_bfcl_load_all_categories(dirs):
    raw, processed = dirs
    _write_bfcl_fixture(raw)
    adapter = BFCLAdapter(raw / "bfcl", processed, {})
    samples = assert_deterministic(adapter)

    assert [sample.subtask for sample in samples] == [
        "simple",
        "simple",
        "multiple",
        "parallel",
        "multi_turn",
    ]
    assert all(sample.dataset == "bfcl" for sample in samples)
    assert all(sample.scenario == "general_agent" for sample in samples)
    assert all(sample.answer is None for sample in samples)
    assert all(sample.answer_type is AnswerType.TOOL_CALL for sample in samples)

    simple = samples[0]
    assert simple.sample_id == "bfcl-simple_python_0"
    assert simple.question == "Find the area"
    assert simple.metadata["category"] == "simple"
    assert simple.metadata["raw_id"] == "simple_python_0"
    assert simple.metadata["num_functions"] == 1
    assert simple.tools is not None and len(simple.tools) == 1
    assert simple.tools[0].type == "function"
    assert simple.tools[0].function["name"] == "calculate_area"
    # single-element acceptable-value lists are unwrapped, multi-value kept
    assert simple.metadata["gold_calls"] == [
        {
            "name": "calculate_area",
            "arguments": {"base": 10, "unit": ["units", ""]},
            "has_alternatives": True,
        }
    ]


def test_bfcl_multiple_parallel_and_multi_turn_ground_truth(dirs):
    raw, processed = dirs
    _write_bfcl_fixture(raw)

    multiple = BFCLAdapter(
        raw / "bfcl", processed, {"categories": ["multiple"]}
    ).load()[0]
    assert multiple.metadata["gold_calls"] == [
        {"name": "f1", "arguments": {"x": 1}},
        {"name": "f2", "arguments": {"y": "a"}},
    ]

    parallel = BFCLAdapter(
        raw / "bfcl", processed, {"categories": ["parallel"]}
    ).load()[0]
    assert [call["name"] for call in parallel.metadata["gold_calls"]] == ["play", "play"]
    assert [call["arguments"]["artist"] for call in parallel.metadata["gold_calls"]] == ["A", "B"]

    multi = BFCLAdapter(
        raw / "bfcl", processed, {"categories": ["multi_turn"], "per_category": 5}
    ).load()[0]
    assert multi.question == "Move the file\nNow search it"
    assert multi.tools is None
    assert multi.metadata["involved_classes"] == ["GorillaFileSystem"]
    assert multi.metadata["num_turns"] == 2
    calls = multi.metadata["gold_calls"]
    assert [call["name"] for call in calls] == ["cd", "mkdir", "sort"]
    assert [call["turn"] for call in calls] == [0, 0, 1]
    assert calls[0]["arguments"] == {"folder": "document"}
    assert calls[2]["arguments"] == {}
    assert calls[2]["positional_arguments"] == ["final_report.pdf"]


def test_bfcl_limit_and_category_filter(dirs):
    raw, processed = dirs
    _write_bfcl_fixture(raw)

    limited = BFCLAdapter(raw / "bfcl", processed, {"per_category": 1}).load()
    assert [sample.subtask for sample in limited] == [
        "simple",
        "multiple",
        "parallel",
        "multi_turn",
    ]

    simple_only = BFCLAdapter(raw / "bfcl", processed, {"categories": ["simple_python"]}).load()
    assert [sample.subtask for sample in simple_only] == ["simple", "simple"]


def test_bfcl_overall_limit_spans_categories(dirs):
    raw, processed = dirs
    _write_bfcl_fixture(raw)
    # scenes.yaml passes `limit: 40` with three categories
    samples = BFCLAdapter(raw / "bfcl", processed, {"limit": 3}).load()
    assert [sample.subtask for sample in samples] == ["simple", "multiple", "parallel"]


def test_bfcl_missing_category_is_skipped(dirs):
    raw, processed = dirs
    _write_bfcl_fixture(raw)
    # remove parallel ground truth -> that category must be skipped, not invented
    (raw / "bfcl" / "possible_answer" / "BFCL_v4_parallel.json").unlink()
    adapter = BFCLAdapter(raw / "bfcl", processed, {})
    samples = adapter.load()
    assert "parallel" not in {sample.subtask for sample in samples}
    assert adapter.skipped_categories == ["parallel"]

    adapter_only_parallel = BFCLAdapter(raw / "bfcl", processed, {"categories": ["parallel"]})
    with pytest.raises(FileNotFoundError):
        adapter_only_parallel.load()


def test_bfcl_download_records_errors_and_tolerates_partial_failure(tmp_path, monkeypatch):
    raw = tmp_path / "bfcl"
    processed = tmp_path / "processed"
    adapter = BFCLAdapter(raw, processed, {"categories": ["simple", "parallel"]})

    def fake_download(url, dest, **_kwargs):
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if "parallel" in url:
            raise DownloadError("simulated network failure")
        dest.write_text('{"id": "x"}\n', encoding="utf-8")
        return dest

    monkeypatch.setattr("modelbench.adapters.datasets.bfcl.download_file", fake_download)
    adapter.ensure_downloaded()

    assert (raw / "BFCL_v4_simple_python.json").is_file()
    errors = json.loads((raw / "download_errors.json").read_text(encoding="utf-8"))
    # both the question and ground-truth transfer failed for "parallel"
    assert len(errors) == 2
    assert {error["kind"] for error in errors} == {"questions", "ground_truth"}
    assert {error["category"] for error in errors} == {"parallel"}

    # all downloads failing -> clear exception
    def failing_download(url, dest, **_kwargs):
        raise DownloadError("offline")

    monkeypatch.setattr("modelbench.adapters.datasets.bfcl.download_file", failing_download)
    with pytest.raises(DownloadError):
        BFCLAdapter(tmp_path / "bfcl_empty", processed, {"categories": ["simple"]}).ensure_downloaded()


def test_bfcl_parse_ground_truth_handles_unknown_shapes():
    assert parse_ground_truth(None) == []
    assert parse_ground_truth([]) == []
    assert parse_ground_truth("garbage") == []
    assert parse_ground_truth([{"f": {"a": [1, 2]}}]) == [
        {"name": "f", "arguments": {"a": [1, 2]}, "has_alternatives": True}
    ]


def test_bfcl_missing_raw_files_raise(dirs):
    raw, processed = dirs
    adapter = BFCLAdapter(raw / "bfcl", processed, {"categories": ["simple"]})
    with pytest.raises(FileNotFoundError) as excinfo:
        adapter.load()
    assert "modelbenchmark download --dataset bfcl" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Cross-adapter invariants
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "builder,adapter_factory",
    [
        ("mmlu", lambda raw, processed: MMLUAdapter(raw / "mmlu", processed, {"limit": 2})),
        ("gsm8k", lambda raw, processed: GSM8KAdapter(raw / "gsm8k", processed, {"limit": 2})),
        ("ifeval", lambda raw, processed: IFEvalAdapter(raw / "ifeval", processed, {"limit": 2})),
        (
            "humaneval",
            lambda raw, processed: HumanEvalAdapter(raw / "humaneval", processed, {"limit": 1}),
        ),
        (
            "hotpotqa",
            lambda raw, processed: HotpotQAAdapter(raw / "hotpotqa", processed, {"limit": 1}),
        ),
    ],
)
def test_adapters_raise_helpful_error_when_raw_missing(dirs, builder, adapter_factory):
    raw, processed = dirs
    adapter = adapter_factory(raw, processed)
    with pytest.raises(FileNotFoundError) as excinfo:
        adapter.load()
    assert "modelbenchmark download" in str(excinfo.value)
    assert adapter.name == builder


def test_longbench_streaming_json_array(tmp_path):
    import json as _json

    from modelbench.adapters.datasets.longbench_v2 import _iter_json_array

    rows = [
        {"_id": "b", "context": "x" * 50, "question": "q2"},
        {"_id": "a", "context": "y" * 70, "question": "q1"},
    ]
    path = tmp_path / "data.json"
    path.write_text("\n  " + _json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8")
    for chunk in (7, 64, 1 << 20):
        assert [row["_id"] for row in _iter_json_array(path, chunk_size=chunk)] == ["b", "a"]
