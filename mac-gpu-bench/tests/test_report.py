#!/usr/bin/env python3
"""report.py 的回归测试：规格匹配与中位数回退。

这两处都出过错：子串匹配把 M1 Pro 当成 M1，中位数缺失时又必须能退回峰值。
运行：
    python3 -m unittest discover -s tests -p 'test_*.py'
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import report as rep  # noqa: E402


class SpecLookupTest(unittest.TestCase):
    def test_pro_not_matched_as_base(self):
        name, spec = rep.lookup_spec("Apple M1 Pro")
        self.assertEqual(name, "Apple M1 Pro")
        self.assertEqual(spec["bandwidth_gbs"], 200.0)

    def test_max_and_ultra(self):
        self.assertEqual(rep.lookup_spec("Apple M4 Max")[0], "Apple M4 Max")
        self.assertEqual(rep.lookup_spec("Apple M1 Ultra")[0], "Apple M1 Ultra")
        self.assertEqual(rep.lookup_spec("Apple M3 Pro")[0], "Apple M3 Pro")

    def test_base(self):
        self.assertEqual(rep.lookup_spec("Apple M1")[0], "Apple M1")

    def test_unknown_is_none(self):
        self.assertIsNone(rep.lookup_spec("Apple M9 Ultra")[0])


class StableTest(unittest.TestCase):
    def test_prefers_median(self):
        self.assertEqual(rep.stable({"value": 3.0, "value_median": 2.0}, "value"), 2.0)

    def test_falls_back_for_legacy(self):
        self.assertEqual(rep.stable({"value": 3.0}, "value"), 3.0)

    def test_gbps_field(self):
        self.assertEqual(rep.stable({"gbps": 55.0, "gbps_median": 54.0}, "gbps"), 54.0)


class RawViewTest(unittest.TestCase):
    def test_keeps_measured_sections_only(self):
        text = ("# t\n\n## 1. 结论\n\nx\n\n"
                "## 2. 算力\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n"
                "## 5. roofline\n\nzzz\n")
        out = rep.raw_view(text, {"notes": ["n1"]})
        self.assertIn("## 2.", out)
        self.assertNotIn("## 1.", out)
        self.assertNotIn("## 5.", out)
        self.assertIn("## 备注", out)

    def test_drops_empty_tables(self):
        text = ("# t\n\n## 2. 算力\n\n| a | b |\n|---|---|\n\n"
                "## 3. GEMM\n\n| a | b |\n|---|---|\n| 1 | 2 |\n")
        out = rep.raw_view(text, {})
        self.assertNotIn("## 2.", out)
        self.assertIn("## 3.", out)


if __name__ == "__main__":
    unittest.main()
