import tempfile
import unittest
from pathlib import Path

from scripts.instrument_n0_normal2_full_grid_traces import generate


class Normal2GridTraceTests(unittest.TestCase):
    def test_generates_two_independent_variants(self):
        source = (
            "sqrt.approx.ftz.f32 %r225, %r224;\n"
            "cos.approx.ftz.f32 %r230, %r227;\n"
            "{ cvt.rn.f16.f32 %rs17, %r151;}\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            variants = generate(source, Path(directory))
            self.assertEqual([row["name"] for row in variants], ["cos1", "normal2_boundary"])
            self.assertIn("%r230", (Path(directory) / variants[0]["filename"]).read_text())
            self.assertIn("cvt.u32.u16", (Path(directory) / variants[1]["filename"]).read_text())


if __name__ == "__main__":
    unittest.main()
