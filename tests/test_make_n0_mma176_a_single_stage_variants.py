import tempfile
import unittest
from pathlib import Path

from scripts.make_n0_mma176_a_single_stage_variants import generate
from scripts.instrument_n0_mma176_a_path_trace import STAGES


class N0Mma176ASingleStageVariantsTests(unittest.TestCase):
    def test_generates_one_store_per_variant(self):
        source = Path("results/20260831_011219_zluda_ptx_probe/neural_isolated.ptx").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            variants = generate(source, Path(directory), 35, 0)
            first = (Path(directory) / variants[0]["filename"]).read_text(encoding="utf-8")
        self.assertEqual(len(variants), len(STAGES))
        self.assertEqual(first.count("@%__n0_a176_0_selected st.global.b32"), 1)


if __name__ == "__main__": unittest.main()
