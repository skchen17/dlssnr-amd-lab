import unittest

from scripts.lower_ptx_surface_compat import lower


class SurfaceCompatibilityLoweringTests(unittest.TestCase):
    def test_rgba16f_surface_store_is_lowered_to_linear_output(self):
        source = (
            "sust.p.2d.v4.b32.zero [%rd7, {%r1,%r2}], "
            "{%r3,%r4,%r5,%r6};\n"
        )
        result, counts = lower(source)
        self.assertEqual(counts["rgba16f_surface_stores_lowered"], 1)
        self.assertEqual(counts["remaining_rgba16f_surface_stores"], 0)
        self.assertIn("ld.param.b32 %__dlssnr_surface_0_width, [%rd11+172]", result)
        self.assertIn("mul.wide.u32 %__dlssnr_surface_0_row, %r2", result)
        self.assertIn("cvt.rn.f16.f32 %__dlssnr_surface_0_h3, %r6", result)
        self.assertIn("st.global.v2.b32", result)
        self.assertNotIn("sust.p.2d", result)


if __name__ == "__main__":
    unittest.main()
