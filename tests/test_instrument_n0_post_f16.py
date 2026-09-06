import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "instrument_n0_post_f16.py"
SPEC = importlib.util.spec_from_file_location("instrument_n0_post_f16", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class InstrumentN0PostF16Test(unittest.TestCase):
    def test_keeps_sixteen_mmas_and_exports_outputs(self):
        mma = "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%r1}, {%r2};"
        source = ".version 9.4\n" + (mma + "\n") * 16 + "later;\n"
        output = MODULE.instrument(source)
        self.assertEqual(len(MODULE.F16_MMA.findall(output)), 16)
        self.assertNotIn("later", output)
        self.assertIn("%r637", output)
        self.assertIn(".version 8.7", output)


if __name__ == "__main__":
    unittest.main()
