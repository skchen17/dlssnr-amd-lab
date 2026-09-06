import unittest

from scripts.lower_ptx_nvidia_sqrt0_ulp_correction import lower


class NvidiaSqrt0UlpCorrectionLoweringTests(unittest.TestCase):
    def test_model_targets_first_sqrt_and_changes_bits(self):
        model = {"segments": 2, "scale_u32": 0x3E800000, "ulp_shifts_i32": [0, -1]}
        source = ".visible .entry x() {\nsqrt.approx.ftz.f32 %r221, %r220;\n}\n"
        output, count = lower(source, model)
        self.assertEqual(count, 1)
        self.assertIn("__n0_nvidia_sqrt0_ulp_shift", output)
        self.assertIn("add.s32 %__n0_sqrt_ulp_0_bits", output)

    def test_model_can_target_second_sqrt(self):
        model = {"segments": 1, "scale_u32": 0x3F800000, "ulp_shifts_i32": [1]}
        source = ".visible .entry x() {\nsqrt.approx.ftz.f32 %r225, %r224;\n}\n"
        output, count = lower(source, model, site="second")
        self.assertEqual(count, 1)
        self.assertIn("__n0_nvidia_sqrt1_ulp_shift", output)
        self.assertIn("mov.b32 %r225", output)


if __name__ == "__main__":
    unittest.main()
