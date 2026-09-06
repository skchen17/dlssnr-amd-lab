import unittest

from scripts.lower_n0_f16x2_square_pair_fusion import lower


class LowerN0F16x2SquarePairFusionTests(unittest.TestCase):
    def test_lowers_left_square_into_explicit_half2_fma(self):
        source = """
{mul.f16x2 %r10,%r1,%r1;
}
{mul.f16x2 %r11,%r2,%r2;
}
{add.f16x2 %r12,%r10,%r11;
}
"""
        output, replacements = lower(source)
        self.assertIn("fma.rn.f16x2 %r12, %r1, %r1, %r11;", output)
        self.assertEqual(len(replacements), 1)
        self.assertEqual(replacements[0]["separately_rounded_square_register"], "%r11")

    def test_does_not_rewrite_non_square_add(self):
        source = "{add.f16x2 %r12,%r10,%r11;\n}"
        output, replacements = lower(source)
        self.assertEqual(output, source)
        self.assertEqual(replacements, [])
