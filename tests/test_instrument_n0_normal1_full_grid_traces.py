import tempfile
import unittest
from pathlib import Path

from scripts.instrument_n0_normal1_full_grid_traces import generate


class Normal1GridTraceTests(unittest.TestCase):
    def test_generates_two_independent_variants(self):
        source = (
            "sqrt.approx.ftz.f32 %r221, %r220;\n"
            "sin.approx.ftz.f32 %r228, %r226;\n"
            "{ cvt.rn.f16.f32 %rs16, %r150;}\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            variants = generate(source, Path(directory))
            self.assertEqual([row["name"] for row in variants], ["sin0", "normal1_boundary"])
            self.assertIn("%r228", (Path(directory) / variants[0]["filename"]).read_text())
            self.assertIn("cvt.u32.u16", (Path(directory) / variants[1]["filename"]).read_text())


if __name__ == "__main__":
    unittest.main()
