import unittest

from scripts.instrument_swin4h_fp8_mma_trace import instrument


FUNCTION = "cc_tinlayout_fused_swin_4h_128_4_inpview_tilesync_fp8"
MMA = """mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16
{%r9, %r10}, {%r1, %r2, %r3, %r4}, {%r5, %r6}, {%r7, %r8};"""


class Swin4hFp8MmaTraceTests(unittest.TestCase):
    def test_selected_warp_and_non_overlapping_mma_offsets(self):
        source = f""".version 9.4
.visible .entry {FUNCTION}(
.param .align 8 .b8 {FUNCTION}_param_0[88]
)
.maxnreg 168
{{
{MMA}
{MMA}
ret;
}}
"""
        output, count = instrument(source, FUNCTION, 4, 0, 1)
        self.assertEqual(count, 2)
        self.assertIn(f"ld.param.u64 %__s4mma_base, [{FUNCTION}_param_0+72];", output)
        self.assertIn("setp.eq.u32 %__s4mma_px, %__s4mma_ctax, 4;", output)
        self.assertIn("setp.eq.u32 %__s4mma_pw, %__s4mma_warpy, 1;", output)
        self.assertIn("mad.lo.u32 %__s4mma_offset, %__s4mma_lane, 40, 0;", output)
        self.assertIn("mad.lo.u32 %__s4mma_offset, %__s4mma_lane, 40, 1280;", output)
        self.assertEqual(output.count("[%__s4mma_address+36], %r10;"), 2)

    def test_missing_entry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not found exactly once"):
            instrument(MMA, FUNCTION, 4, 0, 1)


if __name__ == "__main__":
    unittest.main()
