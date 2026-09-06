import unittest

from scripts.lower_ptx_nvidia_sqrt0_correction import lower


class NvidiaSqrt0CorrectionLoweringTests(unittest.TestCase):
    def test_sparse_model_targets_first_sqrt_and_indexes_output(self):
        model = {"segments": 2, "degree": 0, "scale_u32": 0x3E800000,
                 "coefficients_u32": [[0], [0x33800000]]}
        source = ".visible .entry x() {\nsqrt.approx.ftz.f32 %r221, %r220;\n}\n"
        output, count = lower(source, model)
        self.assertEqual(count, 1)
        self.assertIn("__n0_nvidia_sqrt0_correction", output)
        self.assertIn("mul.rn.f32 %__n0_sqrt_fix_0_position, %r221", output)
        self.assertIn("@%__n0_sqrt_fix_0_nonzero add.rn.f32 %r221", output)


if __name__ == "__main__":
    unittest.main()
