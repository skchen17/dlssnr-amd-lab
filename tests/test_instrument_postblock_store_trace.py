import unittest

from scripts.instrument_postblock_store_trace import SITE_BYTES, TRACE_BYTES, instrument


class PostblockStoreTraceTests(unittest.TestCase):
    def test_expands_parameter_and_traces_both_stores(self):
        text = """.visible .entry cc_tinlayout_fused_post_block_swin_1h_32_fp8(
.param .align 8 .b8 cc_tinlayout_fused_post_block_swin_1h_32_fp8_param_0[184]
)
sust.p.2d.v4.b32.zero [%rd398, {%r1,%r2}], {%r3,%r4,%r5,%r6};
sust.p.2d.v4.b32.zero [%rd439, {%r7,%r8}], {%r9,%r10,%r11,%r12};
"""
        output, counts = instrument(text)
        self.assertEqual(counts["parameter_declarations_expanded"], 1)
        self.assertEqual(counts["surface_store_sites_instrumented"], 2)
        self.assertIn("param_0[192]", output)
        self.assertIn(f"add.u32 %__dlssnr_post_trace_1_byte, %__dlssnr_post_trace_1_byte, {SITE_BYTES};", output)
        self.assertEqual(output.count("sust.p.2d.v4.b32.zero"), 2)
        self.assertEqual(TRACE_BYTES, 2 * SITE_BYTES)


if __name__ == "__main__":
    unittest.main()
