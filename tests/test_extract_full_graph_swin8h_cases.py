import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "extract_full_graph_swin8h_cases.py"
SPEC = importlib.util.spec_from_file_location("extract_full_graph_swin8h_cases", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FullGraphSwin8hExtractionHelpersTest(unittest.TestCase):
    def test_sizes_and_padding(self):
        self.assertEqual(MODULE.MAIN_LOGICAL_BYTES, 245_760)
        self.assertEqual(MODULE.EXTRA_LOGICAL_BYTES, 122_880)
        self.assertEqual(MODULE.fit_state(b"ab", 4), b"ab\0\0")


if __name__ == "__main__":
    unittest.main()
