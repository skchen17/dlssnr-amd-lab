import unittest

from scripts.lower_ptx_nvidia_sin_correction import lower


class NvidiaSinCorrectionLoweringTests(unittest.TestCase):
    def test_sparse_model_targets_first_sine(self):
        model = {
            "segments": 2,
            "degree": 0,
            "scale_u32": 0x3E22F983,
            "skip_zero_correction": True,
            "coefficients_u32": [[0], [0x33800000]],
        }
        source = ".visible .entry x() {\nsin.approx.ftz.f32 %r228, %r226;\n}\n"
        output, count = lower(source, model)
        self.assertEqual(count, 1)
        self.assertIn("__n0_nvidia_sin0_correction", output)
        self.assertIn("@%__n0_sin_fix_0_nonzero add.rn.f32", output)


if __name__ == "__main__":
    unittest.main()
