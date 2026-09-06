import unittest
from scripts.instrument_n0_lg2_full_grid_trace import SAMPLES,TRACE_BYTES,instrument

class InstrumentN0Lg2FullGridTraceTests(unittest.TestCase):
    def test_instruments_first_box_muller_lg2(self):
        output,count=instrument('lg2.approx.ftz.f32 %r218, %r181;')
        self.assertEqual(count,1);self.assertIn('mov.u32 %__n0_lg2_grid_trace_ctax, %ctaid.x;',output);self.assertIn('mad.lo.u32 %__n0_lg2_grid_trace_linear, %__n0_lg2_grid_trace_ctay, 80, %__n0_lg2_grid_trace_ctax;',output);self.assertIn('st.global.b32 [%__n0_lg2_grid_trace_address+4], %r218;',output);self.assertEqual(SAMPLES,245760);self.assertEqual(TRACE_BYTES,1966080)

if __name__=='__main__':unittest.main()
