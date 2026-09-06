import unittest

from scripts.lower_n0_cos_observed_input_ablation import lower


class ObservedInputCosAblationTests(unittest.TestCase):
    def test_three_observed_inputs_are_guarded(self):
        source = "cos.approx.ftz.f32 %r229, %r226;\ncos.approx.ftz.f32 %r230, %r227;\n"
        output, replacements, metadata = lower(source, 3)
        self.assertEqual(replacements, 1)
        self.assertEqual(len(metadata), 3)
        self.assertIn("1083854280", output)  # 0x409A4DC8
        self.assertIn("0fB3C00000", output)
        self.assertIn("cos.approx.ftz.f32 %r230, %r227;", output)

    def test_invalid_count_is_rejected(self):
        with self.assertRaises(ValueError):
            lower("cos.approx.ftz.f32 %r229, %r226;", 0)


if __name__ == "__main__":
    unittest.main()
