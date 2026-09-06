import unittest

from scripts.instrument_n0_fp8_mma_trace import TRACE_OFFSET, instrument


class InstrumentN0Fp8MmaTraceTests(unittest.TestCase):
    def test_uses_scratch_pointer_and_wraps_fragments(self):
        source = """
mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16
{%d0, %d1}, {%a0, %a1, %a2, %a3}, {%b0, %b1}, {%c0, %c1};
"""
        output, count = instrument(source)
        self.assertEqual(count, 1)
        self.assertIn(f"%__n0_mma_0_lane, 40, {TRACE_OFFSET}", output)
        self.assertIn("add.s64 %__n0_mma_0_address, %rd6", output)
        self.assertIn("[%__n0_mma_0_address+0], %a0", output)
        self.assertIn("[%__n0_mma_0_address+28], %c1", output)
        self.assertIn("[%__n0_mma_0_address+36], %d1", output)


if __name__ == "__main__":
    unittest.main()
