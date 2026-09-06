import unittest

from scripts.instrument_n0_box_muller_outputs_cta_trace import STAGES, instrument


class InstrumentN0BoxMullerOutputsCtaTraceTests(unittest.TestCase):
    def test_instruments_only_three_consumed_half_boundaries(self):
        source = "\n".join(statement for _, statement, _, _ in STAGES)
        output, count = instrument(source, 7, 1)
        self.assertEqual(count, 3)
        self.assertEqual(output.count("setp.eq.u32 %__n0_box_trace_0_px"), 1)
        self.assertIn("%__n0_box_trace_0_ctax, 7", output)
        self.assertIn("%__n0_box_trace_0_ctay, 1", output)
        self.assertEqual(output.count("st.global.b32"), 3)
