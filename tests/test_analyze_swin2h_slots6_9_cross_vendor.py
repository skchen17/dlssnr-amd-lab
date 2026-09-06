import importlib.util
import struct
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_swin2h_slots6_9_cross_vendor.py"
SPEC = importlib.util.spec_from_file_location("analyze_swin2h_slots6_9_cross_vendor", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class AnalyzeSwin2hCrossVendorTest(unittest.TestCase):
    def test_adjacent_fp16_values_and_zero_tails_pass(self):
        bits = list(range(0x3C00, 0x3C00 + 256))
        reference = struct.pack("<256H", *bits) + bytes(32)
        candidate = struct.pack(
            "<256H", *(value + (index % 64 == 0) for index, value in enumerate(bits))
        ) + bytes(32)
        report = MODULE.compare_fp16(reference, candidate, 256)
        self.assertGreater(report["exact_or_adjacent_fraction"], 0.95)
        self.assertTrue(report["parity_gate"]["pass"])

    def test_nonzero_tail_fails(self):
        reference = struct.pack("<4e", 1.0, 2.0, 3.0, 4.0) + b"\x00"
        candidate = struct.pack("<4e", 1.0, 2.0, 3.0, 4.0) + b"\x01"
        report = MODULE.compare_fp16(reference, candidate, 4)
        self.assertFalse(report["parity_gate"]["pass"])


if __name__ == "__main__":
    unittest.main()
