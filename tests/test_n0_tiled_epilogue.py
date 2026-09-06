import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_n0_tiled_epilogue.py"
SPEC = importlib.util.spec_from_file_location("n0_tiled_epilogue", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class N0TiledEpilogueTest(unittest.TestCase):
    def test_constant_tile_reconstructs_constant_output(self):
        code = MODULE.RELATION.float_to_e4m3(1.5)
        scratch = bytes([code] * (8 * 8 * 32))
        expected = bytes([code] * (4 * 4 * 32))
        self.assertEqual(MODULE.reconstruct(scratch, 8, 8), expected)
        report = MODULE.analyze(scratch, expected, 8, 8)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["mismatches"], 0)


if __name__ == "__main__":
    unittest.main()
