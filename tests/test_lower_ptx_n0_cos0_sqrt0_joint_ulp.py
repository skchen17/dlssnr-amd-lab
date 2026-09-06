import unittest

from scripts.lower_ptx_n0_cos0_sqrt0_joint_ulp import lower


class N0Cos0Sqrt0JointUlpLoweringTests(unittest.TestCase):
    def test_joint_table_shifts_cos0_and_sqrt0_at_first_cosine(self):
        model = {
            "segments": 2,
            "scale_u32": 0x3E22F983,
            "cos0_raw_ulp_shifts_i32": [0, -1],
            "sqrt0_post_ulp_shifts_i32": [0, -2],
        }
        source = ".visible .entry x() {\ncos.approx.ftz.f32 %r229, %r226;\n}\n"
        output, count = lower(source, model)
        self.assertEqual(count, 1)
        self.assertIn("__n0_cos0_sqrt0_joint_ulp", output)
        self.assertIn("mov.b32 %r229", output)
        self.assertIn("mov.b32 %r221", output)
        self.assertIn("mul.wide.u32 %__n0_joint_ulp_0_offset", output)


if __name__ == "__main__":
    unittest.main()
