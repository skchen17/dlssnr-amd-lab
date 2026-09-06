import unittest

from scripts.lower_postblock_mul_add_fusion import lower


class PostblockMulAddFusionTests(unittest.TestCase):
    def test_fuses_second_product_only(self):
        text = ("mul.f16x2 %r10, %r1, %r2;\n"
                "mul.f16x2 %r11, %r3, %r4;\n"
                "add.f16x2 %r12, %r10, %r11;\n")
        output, report = lower(text, 0, 1)
        self.assertEqual(report["candidate_count"], 1)
        self.assertIn("fma.rn.f32", output)
        self.assertIn("%r3", output); self.assertIn("%r4", output)
        self.assertNotIn("add.f16x2 %r12", output)
        self.assertIn("mul.f16x2 %r11", output)


if __name__ == "__main__":
    unittest.main()
