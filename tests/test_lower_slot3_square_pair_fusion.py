import unittest

from scripts.lower_slot3_square_pair_fusion import lower


class Slot3SquarePairFusionTest(unittest.TestCase):
    TEXT = """
mul.f16x2 %r1807, %r1751, %r1751;
mul.f16x2 %r1808, %r1759, %r1759;
add.f16x2 %r1868, %r1807, %r1808;
"""

    def test_fuses_right_square_like_postblock_contract(self):
        output, report = lower(self.TEXT, "fuse_right", ("%r1868",))
        self.assertIn("fma.rn.f16x2 %r1868, %r1759, %r1759, %r1807;", output)
        self.assertEqual(report[0]["separately_rounded_square"], "%r1807")

    def test_fuses_left_square_alternative(self):
        output, _ = lower(self.TEXT, "fuse_left", ("%r1868",))
        self.assertIn("fma.rn.f16x2 %r1868, %r1751, %r1751, %r1808;", output)


if __name__ == "__main__":
    unittest.main()
