"""Documentation must stay aligned with code and filesystem."""

from pathlib import Path

from harness import docs_contract as dc


def test_docs_contract():
    report = dc.check_all()
    assert report["ok"], "\n".join(report["issues"])


def test_docs_cover_prompts_and_nested_docs():
    """Role prompts and docs/<subdir>/ pages are part of the contract."""
    rels = {p.relative_to(dc.ROOT).as_posix() for p in dc.DOCS}
    assert any(rel.startswith("harness/prompts/") for rel in rels), rels
    assert "docs/process/50-validation-plan.md" in rels, rels
    # New numbered layout is fully covered by the recursive glob.
    for expected in (
        "docs/spec/21-test-plan.md",
        "docs/design/30-architecture.md",
        "docs/design/31-output-contract.md",
        "docs/guides/40-install.md",
    ):
        assert expected in rels, (expected, rels)
    # archived documents are historical and intentionally outside the contract
    assert not any("/archive/" in f"/{rel}" for rel in rels), rels


def test_dead_artifact_detection(tmp_path: Path):
    doc = tmp_path / "sample.md"
    doc.write_text(
        "旧产物：`summary.json`、`hardware.json`、`output/<run_id>/`\n",
        encoding="utf-8",
    )
    issues = dc.check_dead_artifacts([doc])
    assert issues, "removed artifact names must be flagged"
    assert all("removed artifact" in issue for issue in issues)
    assert any("summary.json" in issue or "hardware.json" in issue for issue in issues)


def test_dead_artifact_whitelist_and_false_positive(tmp_path: Path):
    """History/policy documents may name removed artifacts; lookalike names are fine."""
    whitelisted = [p for p in dc.DEAD_ARTIFACT_WHITELIST if p.is_file()]
    assert whitelisted, "expected at least one whitelisted doc"
    assert dc.check_dead_artifacts(whitelisted) == []

    ok_doc = tmp_path / "ok.md"
    ok_doc.write_text("`pilot_summary.json` 不是已删除产物名。\n", encoding="utf-8")
    assert dc.check_dead_artifacts([ok_doc]) == []


def test_iter_docs_is_deduplicated():
    paths = dc.iter_docs()
    assert len(paths) == len(set(paths))
    assert all(p.is_file() for p in paths)
