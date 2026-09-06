import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_fp16_e4m3_oracle.py"
SPEC = importlib.util.spec_from_file_location("fp16_e4m3_analyzer", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class Fp16E4m3AnalyzerTest(unittest.TestCase):
    def test_nvidia_nan_canonicalization_and_lane_order(self):
        oracle = bytes(MODULE.encode_half_bits(i, False) for i in range(65536))
        report = MODULE.analyze(oracle)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["initial_total_mismatches"], 1023)
        self.assertEqual(report["initial_nan_mismatches"], 1023)
        self.assertEqual(report["corrected_total_mismatches"], 0)
        self.assertGreater(report["pair_swapped_mismatches"], 0)


if __name__ == "__main__":
    unittest.main()
