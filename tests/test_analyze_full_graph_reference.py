import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_full_graph_reference.py"
SPEC = importlib.util.spec_from_file_location("analyze_full_graph_reference", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FullGraphReferenceHelpersTest(unittest.TestCase):
    def test_member_path_validation(self):
        self.assertTrue(MODULE.safe_member("full_graph_blobs/abc.raw"))
        self.assertFalse(MODULE.safe_member("../escape.raw"))
        self.assertFalse(MODULE.safe_member("/absolute.raw"))
        self.assertFalse(MODULE.safe_member("C:/absolute.raw"))

    def test_normalizes_windows_separators(self):
        self.assertEqual(
            MODULE.normalized(r"full_graph_blobs\abc.raw"),
            "full_graph_blobs/abc.raw",
        )


if __name__ == "__main__":
    unittest.main()
