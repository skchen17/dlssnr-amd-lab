import unittest

from scripts.instrument_fp8_mma_trace import instrument


class InstrumentFp8MmaTraceTest(unittest.TestCase):
    def test_instruments_before_and_after_fragments(self):
        source = """
.visible .entry kernel(.param .align 8 .b8 kernel_param_0[96]) {
mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16
{%d0, %d1}, {%a0, %a1, %a2, %a3}, {%b0, %b1}, {%c0, %c1};
ret;
}
"""
        output, count = instrument(source, "kernel")
        self.assertEqual(count, 1)
        self.assertIn("ld.param.u64 %__mma_ck_base, [kernel_param_0+64];", output)
        self.assertIn("[%__mma_ck_address+0], %a0", output)
        self.assertIn("[%__mma_ck_address+28], %c1", output)
        self.assertIn("[%__mma_ck_address+32], %d0", output)
        self.assertIn("[%__mma_ck_address+36], %d1", output)

    def test_selected_cta_is_encoded_in_guard(self):
        source = """
.visible .entry kernel(.param .align 8 .b8 kernel_param_0[96]) {
mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16
{%d0, %d1}, {%a0, %a1, %a2, %a3}, {%b0, %b1}, {%c0, %c1};
ret;
}
"""
        output, count = instrument(source, "kernel", 2, 3)
        self.assertEqual(count, 1)
        self.assertIn("%__mma_ck_ctax, 2;", output)
        self.assertIn("%__mma_ck_ctay, 3;", output)


if __name__ == "__main__":
    unittest.main()
