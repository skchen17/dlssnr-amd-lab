import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_full_graph_swin4h_replay.py"
SPEC = importlib.util.spec_from_file_location("analyze_full_graph_swin4h_replay", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FullGraphSwin4hNumericsTest(unittest.TestCase):
    def test_e4m3_identical_and_adjacent(self):
        identical = MODULE.compare_e4m3(bytes([0x38, 0x40]), bytes([0x38, 0x40]))
        self.assertTrue(identical["parity_gate"]["pass"])
        self.assertEqual(identical["exact_fraction"], 1.0)
        reference = bytes([0x30, 0x38, 0x40, 0x48] * 25)
        candidate_values = bytearray(reference)
        candidate_values[0] += 1
        candidate = bytes(candidate_values)
        adjacent = MODULE.compare_e4m3(reference, candidate)
        self.assertTrue(adjacent["parity_gate"]["pass"])
        self.assertEqual(adjacent["exact_or_adjacent_fraction"], 1.0)

    def test_fp16_identical(self):
        data = b"\x00<\x00@" * 32
        report = MODULE.compare_fp16(data, data)
        self.assertTrue(report["parity_gate"]["pass"])
        self.assertEqual(report["exact_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
