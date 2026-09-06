import importlib.util
import struct
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_n1_cross_vendor.py"
SPEC = importlib.util.spec_from_file_location("analyze_n1_cross_vendor", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class AnalyzeN1CrossVendorTest(unittest.TestCase):
    def test_sync_summary_expected_release_pattern(self):
        words = [0] * MODULE.EXPECTED_RELEASES
        words += [0xFFFFFFFF] * (MODULE.SYNC_WORDS - MODULE.EXPECTED_RELEASES)
        report = MODULE.sync_summary(struct.pack(f"<{MODULE.SYNC_WORDS}I", *words))
        self.assertEqual(report["zero_words"], MODULE.EXPECTED_RELEASES)
        self.assertEqual(report["sentinel_words"], MODULE.SYNC_WORDS - MODULE.EXPECTED_RELEASES)
        self.assertEqual(report["unexpected_words"], 0)

    def test_e4m3_adjacent_values_pass_numerical_gate(self):
        reference = bytes(range(1, 65)) * 4
        candidate = bytearray(reference)
        for index in range(0, len(candidate), 64):
            candidate[index] += 1
        report = MODULE.compare(reference, bytes(candidate))
        self.assertTrue(report["parity_gate"]["pass"])


if __name__ == "__main__":
    unittest.main()
