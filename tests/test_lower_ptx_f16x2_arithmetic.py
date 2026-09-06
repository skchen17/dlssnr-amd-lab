import unittest

from scripts.lower_ptx_f16x2_arithmetic import lower


class LowerPtxF16x2ArithmeticTest(unittest.TestCase):
    def test_all_supported_forms(self):
        source = (
            "mul.f16x2 %r0, %r1, %r2;\n"
            "add.f16x2 %r3, %r4, %r5;\n"
            "min.f16x2 %r6, %r7, %r8;\n"
            "max.f16x2 %r9, %r10, %r11;\n"
            "fma.rn.f16x2 %r12, %r13, %r14, %r15;\n"
            "abs.f16x2 %r16, %r17;\n"
        )
        output, counts = lower(source)
        self.assertEqual(counts, {"binary": 4, "fma": 1, "abs": 1})
        self.assertIn("mul.rn.f32", output)
        self.assertIn("fma.rn.f32", output)
        self.assertIn("and.b32 %r16, %r17, 2147450879;", output)
        self.assertNotIn(".f16x2", output)


if __name__ == "__main__":
    unittest.main()
