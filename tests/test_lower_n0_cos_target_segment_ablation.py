import unittest

from scripts.lower_n0_cos_target_segment_ablation import lower


class TargetSegmentAblationTests(unittest.TestCase):
    def test_lowers_only_target_cosine(self):
        source = (
            "cos.approx.ftz.f32 %r229, %r226;\n"
            "cos.approx.ftz.f32 %r230, %r227;\n"
        )
        output, count, metadata = lower(source, 18)
        self.assertEqual(count, 1)
        self.assertIn("setp.eq.u32 %__n0_cos_target_0_positive", output)
        self.assertIn("208167", output)
        self.assertIn("0f33C00000", output)
        self.assertIn("0fB3800000", output)
        self.assertIn("cos.approx.ftz.f32 %r230, %r227;", output)
        self.assertEqual(metadata["negative_segment"], 161967)

    def test_rejects_unknown_segment_width(self):
        with self.assertRaises(ValueError):
            lower("cos.approx.ftz.f32 %r229, %r226;", 10)


if __name__ == "__main__":
    unittest.main()
