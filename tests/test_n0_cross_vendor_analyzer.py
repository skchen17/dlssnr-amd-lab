import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_n0_cross_vendor.py"
SPEC = importlib.util.spec_from_file_location("n0_cross_vendor", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class N0CrossVendorAnalyzerTest(unittest.TestCase):
    def test_identity_passes(self):
        raw = bytes(range(127)) * 4
        report = MODULE.compare(raw, raw)
        self.assertEqual(report["exact_fraction"], 1.0)
        self.assertTrue(report["parity_gate"]["pass"])

    def test_sign_flip_fails(self):
        reference = bytes(range(1, 120))
        candidate = bytes(value | 0x80 for value in reference)
        report = MODULE.compare(reference, candidate)
        self.assertEqual(report["sign_mismatch_fraction"], 1.0)
        self.assertFalse(report["parity_gate"]["pass"])


if __name__ == "__main__":
    unittest.main()
