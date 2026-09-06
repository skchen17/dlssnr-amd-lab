import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_n0_downsample_relation.py"
SPEC = importlib.util.spec_from_file_location("n0_downsample", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class N0DownsampleRelationTest(unittest.TestCase):
    def test_constant_2x2_tensor(self):
        code = MODULE.float_to_e4m3(1.5)
        logical_scratch = bytes([code] * (4 * 4 * 32))
        logical_output = bytes([code] * (2 * 2 * 32))
        comparison = MODULE.compare(logical_scratch, logical_output, 4, 4)
        self.assertEqual(comparison["mismatches"], 0)

    def test_detile_constant(self):
        code = MODULE.float_to_e4m3(1.5)
        scratch = bytes([code] * (4 * 4 * 32))
        output = bytes([code] * (2 * 2 * 32))
        # Both tile and channel-plane layouts are indistinguishable for a constant.
        report = MODULE.analyze(scratch, output, 4, 4)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["mismatches"], 0)

    def test_known_codes_roundtrip(self):
        for code in range(256):
            if (code & 0x7F) == 0x7F:
                continue
            self.assertEqual(MODULE.float_to_e4m3(MODULE.e4m3_to_float(code)), code)


if __name__ == "__main__":
    unittest.main()
