import importlib.util
import math
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_n0_cross_vendor.py"
SPEC = importlib.util.spec_from_file_location("analyze_n0_cross_vendor", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class AnalyzeN0CrossVendorTest(unittest.TestCase):
    def test_exact_and_adjacent_e4m3_gate(self):
        reference = bytes(range(1, 65)) * 4
        candidate = bytearray(reference)
        for index in range(0, len(candidate), 32):
            candidate[index] += 1
        report = MODULE.compare(reference, bytes(candidate))
        self.assertGreater(report["exact_or_adjacent_fraction"], 0.95)
        self.assertTrue(report["parity_gate"]["pass"])

    def test_e4m3_special_values(self):
        self.assertEqual(MODULE.e4m3(0), 0.0)
        self.assertTrue(math.isnan(MODULE.e4m3(0x7F)))
        self.assertEqual(math.copysign(1.0, MODULE.e4m3(0x80)), -1.0)


if __name__ == "__main__":
    unittest.main()
