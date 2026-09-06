import unittest

from scripts.instrument_n0_f16_mma_cta_trace import EXPECTED_MMA, TRACE_BYTES, instrument


class InstrumentN0F16MmaCtaTraceTests(unittest.TestCase):
    def test_instruments_selected_cta_without_removing_mma(self):
        statement = (
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
            "{%d0, %d1}, {%a0, %a1, %a2, %a3}, {%b0, %b1}, {%c0, %c1};"
        )
        output, count = instrument("\n".join(statement for _ in range(EXPECTED_MMA)), 8, 0)
        self.assertEqual(count, EXPECTED_MMA)
        self.assertEqual(output.count("%ctaid.x"), EXPECTED_MMA)
        self.assertIn("setp.eq.u32 %__n0_cta_f16_mma_0_px, %__n0_cta_f16_mma_0_ctax, 8;", output)
        self.assertIn("st.global.b32 [%__n0_cta_f16_mma_0_address+32], %d0;", output)
        self.assertEqual(TRACE_BYTES, 20480)


if __name__ == "__main__":
    unittest.main()
