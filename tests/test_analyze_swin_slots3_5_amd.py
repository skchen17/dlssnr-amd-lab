import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_swin_slots3_5_amd.py"
SPEC = importlib.util.spec_from_file_location("analyze_swin_slots3_5_amd", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class AnalyzeSwinSlotsTest(unittest.TestCase):
    def test_release_layout_matches_captured_grids(self):
        self.assertEqual(MODULE.EXPECTED_RELEASES[3], 41 * 25)
        self.assertEqual(MODULE.EXPECTED_RELEASES[4], 41 * 24)
        self.assertEqual(MODULE.EXPECTED_RELEASES[5], 0)


if __name__ == "__main__":
    unittest.main()
