"""Documentation must stay aligned with code and filesystem."""

from harness.docs_contract import check_all


def test_docs_contract():
    report = check_all()
    assert report["ok"], "\n".join(report["issues"])
