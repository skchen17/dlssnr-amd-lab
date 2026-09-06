import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_full_graph_inventory.py"
SPEC = importlib.util.spec_from_file_location("build_full_graph_inventory", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class FullGraphInventoryTest(unittest.TestCase):
    def test_repository_capture_is_complete(self):
        root = Path(__file__).parents[1]
        report = MODULE.build(
            root / "results/20260831_002356_rtx5070_feature18_full_frame/frame_001_sequence.csv",
            root / "results/20260831_010100_all_runtime_modules/extraction_manifest.json",
        )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["slot_count"], 156)
        self.assertEqual(report["unique_function_count"], 43)


if __name__ == "__main__":
    unittest.main()
