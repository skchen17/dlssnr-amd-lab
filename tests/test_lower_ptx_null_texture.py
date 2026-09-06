import unittest

from scripts.lower_ptx_null_texture import lower


class NullTextureLoweringTests(unittest.TestCase):
    def test_texture_sample_becomes_zero_vector(self):
        source = "tex.2d.v4.f32.f32 {%r1, %r2, %r3, %r4}, [%rd5, {%r6, %r7}];\n"
        result, counts = lower(source)
        self.assertEqual(counts["null_texture_samples_lowered"], 1)
        self.assertEqual(counts["remaining_texture_samples"], 0)
        self.assertIn("mov.b32 %r1, 0f00000000;", result)
        self.assertIn("mov.b32 %r4, 0f00000000;", result)
        self.assertNotIn("tex.2d", result)


if __name__ == "__main__":
    unittest.main()
