import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.analyze_slot3_reduction_search import analyze


class ReductionSearchAnalyzerTests(unittest.TestCase):
    def test_selects_lower_nrmse_candidate_and_checks_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.raw"; reference.write_bytes(b"ref")
            baseline = root / "baseline.raw"; baseline.write_bytes(b"base")
            for name, target, data in (("a", "%r1855", b"one"), ("b", "%r2235", b"two")):
                candidate = root / name; candidate.mkdir()
                (candidate / "output.raw").write_bytes(data)
                (candidate / "reduction.json").write_text(json.dumps({"replacements": [{"destination": target}]}))
                (candidate / "probe.json").write_text(json.dumps({
                    "pass": True, "device_name": "AMD Radeon RX 9070 XT [ZLUDA]",
                    "module_loaded": True, "function_resolved": True,
                    "kernel_launched": True, "execution_verified": True,
                }))
            def fake_compare(_reference, value):
                score = {b"base": .3, b"one": .2, b"two": .1}[value]
                return {"nrmse_vs_rtx_stddev": score, "pearson_correlation": 1-score,
                        "exact_fraction": 0.9, "exact_or_adjacent_fraction": 0.99,
                        "sign_mismatches": 0}
            with patch("scripts.analyze_slot3_reduction_search.compare", side_effect=fake_compare):
                report = analyze([root], reference, baseline)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["candidate_count"], 2)
            self.assertEqual(report["best"]["targets"], ["%r2235"])


if __name__ == "__main__":
    unittest.main()
