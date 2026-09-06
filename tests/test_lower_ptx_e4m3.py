import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "lower_ptx_e4m3.py"
SPEC = importlib.util.spec_from_file_location("lower_ptx_e4m3", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)

REFERENCE_SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_fp16_e4m3_oracle.py"
REFERENCE_SPEC = importlib.util.spec_from_file_location("fp16_e4m3_reference", REFERENCE_SCRIPT)
REFERENCE = importlib.util.module_from_spec(REFERENCE_SPEC)
assert REFERENCE_SPEC.loader
REFERENCE_SPEC.loader.exec_module(REFERENCE)


class LowerPtxE4M3Test(unittest.TestCase):
    def test_integer_algorithm_matches_all_half_patterns(self):
        mismatches = [
            bits for bits in range(65536)
            if MODULE.encode_half_bits_integer(bits) != REFERENCE.encode_half_bits(bits, False)
        ]
        self.assertEqual(mismatches, [])

    def test_rewrite_packs_both_lanes(self):
        source = "cvt.rn.satfinite.e4m3x2.f16x2 %rs5, %r99;"
        lowered, count = MODULE.lower(source)
        self.assertEqual(count, 1)
        self.assertNotIn("cvt.rn.satfinite.e4m3x2.f16x2", lowered)
        self.assertIn("cvt.u16.u32 %rs5", lowered)
        self.assertIn("%r99", lowered)


if __name__ == "__main__":
    unittest.main()
