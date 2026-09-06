import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "instrument_n0_pre_mma.py"
SPEC = importlib.util.spec_from_file_location("instrument_n0_pre_mma", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class InstrumentN0PreMmaTest(unittest.TestCase):
    def test_truncates_before_first_mma_and_normalizes_version(self):
        mma = "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
        source = ".version 9.4\n" + (mma + ";\n") * 16
        output = MODULE.instrument(source)
        self.assertIn(".version 8.7", output)
        self.assertNotIn(mma, output)
        self.assertIn("st.global.b32", output)
        self.assertTrue(output.rstrip().endswith("}"))


if __name__ == "__main__":
    unittest.main()
