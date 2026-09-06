import unittest

from scripts.instrument_slot3_rsqrt_trace import (
    BYTES_PER_LANE,
    EXPECTED_BLOCKS,
    instrument,
)


class InstrumentSlot3RsqrtTraceTest(unittest.TestCase):
    def test_exports_each_rsqrt_block(self):
        block = """{.reg.b16 hl, hu;
.reg.b32 fl, fu;
mov.b32 {hl, hu}, %r1;
cvt.f32.f16 fl, hl;
cvt.f32.f16 fu, hu;
rsqrt.approx.ftz.f32 fl, fl;
rsqrt.approx.ftz.f32 fu, fu;
cvt.rn.f16.f32 hl, fl;
cvt.rn.f16.f32 hu, fu;
mov.b32 %r2, {hl, hu};
} """
        source = ".visible .entry kernel(.param .align 8 .b8 kernel_param_0[96]) {\n" + block * EXPECTED_BLOCKS + "ret;\n}"
        output, blocks = instrument(source, "kernel", 2, 3)
        self.assertEqual(len(blocks), EXPECTED_BLOCKS)
        self.assertIn("%__rsq_ctax, 2;", output)
        self.assertIn("%__rsq_ctay, 3;", output)
        self.assertIn("[%__rsq_address+0], %r1", output)
        self.assertIn(f"[%__rsq_address+{BYTES_PER_LANE - 4}], %r2", output)


if __name__ == "__main__":
    unittest.main()
