import tempfile
import unittest
from pathlib import Path

from scripts.analyze_slot3_mma_trace import (
    LANE_BYTES,
    LANES,
    MMA_COUNT,
    analyze,
    candidate_models,
    candidate_half,
    decode_register_trace,
    half_bits,
    round_mantissa,
    truncate_mantissa,
)


class AnalyzeSlot3MmaTraceTest(unittest.TestCase):
    def test_localizes_first_d_only_mismatch(self):
        data = bytearray(MMA_COUNT * LANES * LANE_BYTES)
        other = bytearray(data)
        offset = (7 * LANES + 3) * LANE_BYTES + 32
        other[offset] = 1
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rtx.raw").write_bytes(data)
            (root / "amd.raw").write_bytes(other)
            report = analyze(root / "rtx.raw", root / "amd.raw", include_models=False)
        self.assertTrue(report["input_fragment_gate"])
        self.assertFalse(report["mma_result_bitwise_gate"])
        self.assertEqual(report["first_d_mismatch_mma"], 7)
        self.assertEqual(report["d_half_mismatches"], 1)

    def test_zero_fragment_matches_scalar_model(self):
        data = bytes(MMA_COUNT * LANES * LANE_BYTES)
        lanes = decode_register_trace(data)[0]
        self.assertEqual(candidate_half(lanes, 0, 0, {"name": "current"}), 0)

    def test_candidate_inventory_and_rounding_modes(self):
        models = candidate_models()
        names = [model["name"] for model in models]
        self.assertEqual(len(models), 43)
        self.assertEqual(len(names), len(set(names)))
        value = 1.0006
        self.assertEqual(truncate_mantissa(value, 10), 1.0)
        self.assertEqual(round_mantissa(value, 10), 1.0009765625)
        self.assertEqual(half_bits(70000.0), 0x7C00)
        self.assertEqual(half_bits(-70000.0), 0xFC00)


if __name__ == "__main__":
    unittest.main()
