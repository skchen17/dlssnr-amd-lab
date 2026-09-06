import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "lower_ptx_f16_mma.py"
SPEC = importlib.util.spec_from_file_location("lower_ptx_f16_mma", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class LowerPtxF16MmaTest(unittest.TestCase):
    def test_fragment_sources_cover_requested_coordinates(self):
        for lane in range(32):
            for element in range(4):
                row, column = MODULE.cd_coord(lane, element)
                for k in range(16):
                    a_lane, a_element = MODULE.a_source(lane, element, k)
                    a_group, a_thread = a_lane >> 2, a_lane & 3
                    reconstructed_row = a_group + (8 if a_element & 2 else 0)
                    reconstructed_k = a_thread * 2 + (a_element & 1) + (8 if a_element & 4 else 0)
                    self.assertEqual((reconstructed_row, reconstructed_k), (row, k))
                    b_lane, b_element = MODULE.b_source(lane, element, k)
                    b_group, b_thread = b_lane >> 2, b_lane & 3
                    reconstructed_k = b_thread * 2 + (b_element & 1) + (8 if b_element >= 2 else 0)
                    reconstructed_column = b_group
                    self.assertEqual((reconstructed_k, reconstructed_column), (k, column))

    def test_a_fragment_matches_ptx_isa_layout_examples(self):
        # Lane 0 owns rows 0/8 and K columns 0/1/8/9 in a0/a1/a2/a3.
        self.assertEqual(MODULE.a_source(0, 0, 0), (0, 0))
        self.assertEqual(MODULE.a_source(0, 0, 1), (0, 1))
        self.assertEqual(MODULE.a_source(0, 2, 0), (0, 2))
        self.assertEqual(MODULE.a_source(0, 0, 8), (0, 4))
        self.assertEqual(MODULE.a_source(0, 2, 9), (0, 7))
        # Row 7, K=6 is sourced from lane 31; row 15 uses the high pair.
        self.assertEqual(MODULE.a_source(28, 0, 6), (31, 0))
        self.assertEqual(MODULE.a_source(28, 2, 6), (31, 2))

    def test_rewrite(self):
        statement = (
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
            "{%r1, %r2}, {%r3, %r4, %r5, %r6}, {%r7, %r8}, {%r9, %r10};"
        )
        lowered, count = MODULE.lower(statement)
        self.assertEqual(count, 1)
        self.assertNotIn("mma.sync", lowered)
        self.assertIn("fma.rn.f32", lowered)
        self.assertIn("mov.b32 %r1", lowered)


if __name__ == "__main__":
    unittest.main()
