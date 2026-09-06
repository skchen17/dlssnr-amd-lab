import unittest

from scripts.lower_ptx_nvidia_sin0_refinement import MARKER, lower


class NvidiaSin0RefinementTests(unittest.TestCase):
    def test_refinement_is_after_accepted_correction(self):
        model = {"segments": 2, "scale_u32": 0x3E22F983,
                 "coefficients_u32": [[0], [0x33800000]]}
        source = ".visible .entry x() {\n" + MARKER + "\n}\n"
        output, count = lower(source, model)
        self.assertEqual(count, 1)
        self.assertLess(output.rfind(MARKER), output.index("__n0_sin_refine_0_position"))


if __name__ == "__main__":
    unittest.main()
