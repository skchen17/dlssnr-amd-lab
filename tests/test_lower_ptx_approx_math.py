import unittest

from scripts.lower_ptx_approx_math import lower


class LowerPtxApproxMathTest(unittest.TestCase):
    def test_replaces_aliasing_approximate_math(self):
        source = (
            "rsqrt.approx.ftz.f32 %f1, %f1;\n"
            "rcp.approx.ftz.f32 fl, fu;\n"
        )
        output, rsqrt, rcp = lower(source)
        self.assertEqual((rsqrt, rcp), (1, 1))
        self.assertIn("sqrt.rn.ftz.f32 %__accurate_rsqrt_0, %f1;", output)
        self.assertIn("div.rn.ftz.f32 %f1, 0f3F800000, %__accurate_rsqrt_0;", output)
        self.assertIn("div.rn.ftz.f32 fl, 0f3F800000, fu;", output)
        self.assertNotIn(".approx", output)


if __name__ == "__main__":
    unittest.main()
