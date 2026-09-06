import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_n0_pre_mma_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("n0_pre_mma_checkpoint", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class N0PreMmaCheckpointTest(unittest.TestCase):
    def test_split_lane_interleaving(self):
        lane = bytes(range(96))
        a, b = MODULE.split_fragments(lane * 32)
        self.assertEqual(a[:64], lane[:64])
        self.assertEqual(b[:32], lane[64:])
        self.assertEqual(len(a), 2048)
        self.assertEqual(len(b), 1024)


if __name__ == "__main__":
    unittest.main()
