#!/usr/bin/env python3
"""文档契约测试：把 tools/docs_contract.py 接进 pytest/unittest。

运行：
    python3 -m unittest discover -s tests -p 'test_*.py'
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import docs_contract as dc  # noqa: E402


class DocsContractTest(unittest.TestCase):
    def test_contract_passes(self):
        report = dc.check_all()
        self.assertTrue(report["ok"], "\n".join(report["issues"]))

    def test_every_doc_is_indexed(self):
        self.assertEqual(dc.check_index(), [])

    def test_kernel_matrix_matches_source(self):
        self.assertEqual(dc.check_kernels(), [])

    def test_schema_matches_latest_result(self):
        self.assertEqual(dc.check_schema(), [])

    def test_docs_listed(self):
        docs = [dc.rel(p) for p in dc.iter_docs()]
        self.assertIn("docs/spec/10-methodology.md", docs)
        self.assertIn("docs/design/21-output-contract.md", docs)
        self.assertIn("docs/process/40-validation.md", docs)
        self.assertEqual(len(docs), len(set(docs)))


if __name__ == "__main__":
    unittest.main()
