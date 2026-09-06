import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_mma_fragments.py"
SPEC = importlib.util.spec_from_file_location("mma_fragments", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class MmaFragmentMappingTest(unittest.TestCase):
    def test_fragment_formulas_are_bijections(self):
        a = {MODULE.a_coord(lane, i) for lane in range(32) for i in range(16)}
        b = {MODULE.b_coord(lane, i) for lane in range(32) for i in range(8)}
        cd = {MODULE.cd_coord(lane, i) for lane in range(32) for i in range(4)}
        self.assertEqual(a, {(r, c) for r in range(16) for c in range(32)})
        self.assertEqual(b, {(r, c) for r in range(32) for c in range(8)})
        self.assertEqual(cd, {(r, c) for r in range(16) for c in range(8)})


if __name__ == "__main__":
    unittest.main()
