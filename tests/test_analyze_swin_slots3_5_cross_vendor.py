import importlib.util
import struct
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_swin_slots3_5_cross_vendor.py"
SPEC = importlib.util.spec_from_file_location("analyze_swin_slots3_5_cross_vendor", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class AnalyzeSwinCrossVendorTest(unittest.TestCase):
    def test_adjacent_fp16_values_pass(self):
        bits = list(range(0x3C00, 0x3C00 + 256))
        reference = struct.pack("<256H", *bits)
        candidate = struct.pack("<256H", *(value + (index % 64 == 0) for index, value in enumerate(bits)))
        report = MODULE.compare_fp16(reference, candidate, 256)
        self.assertGreater(report["exact_or_adjacent_fraction"], 0.95)
        self.assertTrue(report["parity_gate"]["pass"])


if __name__ == "__main__":
    unittest.main()
