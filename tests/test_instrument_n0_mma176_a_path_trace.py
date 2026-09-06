import unittest
from pathlib import Path

from scripts.instrument_n0_mma176_a_path_trace import STAGES, TRACE_BYTES, instrument


class InstrumentN0Mma176APathTraceTests(unittest.TestCase):
    def test_instruments_both_complete_reductions_for_selected_cta(self):
        source = Path("results/20260831_011219_zluda_ptx_probe/neural_isolated.ptx").read_text(encoding="utf-8")
        output, stages = instrument(source, 35, 0)
        self.assertEqual(len(stages), len(STAGES))
        self.assertEqual(TRACE_BYTES, 5760)
        self.assertIn("setp.eq.u32 %__n0_a176_0_px, %__n0_a176_0_ctax, 35;", output)
        self.assertIn("@%__n0_a176_44_selected st.global.b32", output)


if __name__ == "__main__": unittest.main()
