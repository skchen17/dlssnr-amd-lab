import unittest

from scripts.lower_ptx_nvidia_cos_correction import lower


class NvidiaCosCorrectionLoweringTests(unittest.TestCase):
    def test_sparse_model_skips_zero_correction(self):
        model = {
            "segments": 2,
            "degree": 0,
            "scale_u32": 0x3E22F983,
            "skip_zero_correction": True,
            "coefficients_u32": [[0], [0x33800000]],
        }
        source = ".visible .entry x() {\ncos.approx.ftz.f32 %r229, %r226;\n}\n"
        output, count = lower(source, model)
        self.assertEqual(count, 1)
        self.assertIn("setp.ne.f32", output)
        self.assertIn("@%__n0_cos_fix_0_nonzero add.rn.f32", output)

    def test_can_target_second_cosine_with_distinct_symbol(self):
        model = {
            "segments": 1,
            "degree": 0,
            "scale_u32": 0x3F800000,
            "coefficients_u32": [[0x00000000]],
            "skip_zero_correction": True,
        }
        source = (
            ".version 8.7\n.visible .entry test() {\n"
            "cos.approx.ftz.f32 %r229, %r226;\n"
            "cos.approx.ftz.f32 %r230, %r227;\n}\n"
        )
        output, count = lower(source, model, site="second")
        self.assertEqual(count, 1)
        self.assertIn("__n0_nvidia_cos1_correction", output)
        self.assertIn("mul.rn.f32 %__n0_cos_fix_0_position, %r227", output)
        self.assertEqual(output.count("%__n0_cos_fix_0_position"), 4)


if __name__ == "__main__":
    unittest.main()
