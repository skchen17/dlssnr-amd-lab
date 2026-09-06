import unittest

from scripts.instrument_n0_normal0_boundary_full_grid_trace import instrument


class Normal0BoundaryGridTraceTests(unittest.TestCase):
    def test_instruments_both_fields_once(self):
        source = (
            "sqrt.approx.ftz.f32 %r221, %r220;\n"
            "cvt.rn.f16.f32 %rs15, %r149;\n"
        )
        output, counts = instrument(source)
        self.assertEqual(counts, {"sqrt0": 1, "normal0_f16": 1})
        self.assertIn("st.global.b32", output)
        self.assertIn("cvt.u32.u16", output)
        self.assertIn("7864324", output)


if __name__ == "__main__":
    unittest.main()
