import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "extract_full_graph_swin4h_cases.py"
SPEC = importlib.util.spec_from_file_location("extract_full_graph_swin4h_cases", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FullGraphSwin4hExtractionHelpersTest(unittest.TestCase):
    def test_fit_state_truncates_and_zero_extends(self):
        self.assertEqual(MODULE.fit_state(b"abcdef", 4), b"abcd")
        self.assertEqual(MODULE.fit_state(b"ab", 4), b"ab\0\0")

    def test_declared_logical_sizes(self):
        self.assertEqual(MODULE.MAIN_LOGICAL_BYTES, 491_520)
        self.assertEqual(MODULE.EXTRA_LOGICAL_BYTES, 245_760)
        self.assertEqual(MODULE.SYNC_BYTES, 110_592)


if __name__ == "__main__":
    unittest.main()
