import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_output_head_weight_layout.py"
SPEC = importlib.util.spec_from_file_location("output_head_weight_layout", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class OutputHeadWeightLayoutTest(unittest.TestCase):
    def test_mapping_is_bijective(self):
        indices = {
            MODULE.packed_weight_index(k, n)
            for k in range(512)
            for n in range(16)
        }
        self.assertEqual(indices, set(range(8192)))

    def test_first_fragment_coordinates(self):
        self.assertEqual(MODULE.packed_weight_index(0, 0), 0)
        self.assertEqual(MODULE.packed_weight_index(4, 0), 16)
        self.assertEqual(MODULE.packed_weight_index(16, 0), 4)
        self.assertEqual(MODULE.packed_weight_index(0, 1), 64)
        self.assertEqual(MODULE.packed_weight_index(0, 8), 8)
        self.assertEqual(MODULE.packed_weight_index(32, 0), 512)

    def test_mapping_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            MODULE.packed_weight_index(512, 0)
        with self.assertRaises(ValueError):
            MODULE.packed_weight_index(0, 16)


if __name__ == "__main__":
    unittest.main()
