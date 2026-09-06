import struct
import unittest

from scripts.analyze_postblock_e4m3_mov_trace import analyze_bytes
from scripts.instrument_postblock_e4m3_mov_trace import (
    E4M3_COUNT, MOV_COUNT, TRACE_BYTES, instrument,
)


class PostblockE4m3MovTraceTests(unittest.TestCase):
    def test_instrument_counts(self):
        conversion = "cvt.rn.satfinite.e4m3x2.f16x2 %rs1, %r2;\n"
        mov = "movmatrix.sync.trans.aligned.m8n8.b16 %r3, %r4;\n"
        text = (".visible .entry cc_tinlayout_fused_post_block_swin_1h_32_fp8(\n"
                ".param .align 8 .b8 cc_tinlayout_fused_post_block_swin_1h_32_fp8_param_0[184]\n)\n"
                "{\n.reg .b64 %rd11;\n" + conversion * E4M3_COUNT +
                mov * MOV_COUNT + "}\n")
        output, counts = instrument(text, 70, 26)
        self.assertIn("param_0[192]", output)
        self.assertEqual(counts["e4m3_conversions_instrumented"], E4M3_COUNT)
        self.assertEqual(counts["movmatrix_instrumented"], MOV_COUNT)

    def test_analysis_localizes_exact_input_conversion_output(self):
        rtx = bytearray(TRACE_BYTES)
        amd = bytearray(TRACE_BYTES)
        struct.pack_into("<II", rtx, 0, 0x12345678, 0xABCD)
        struct.pack_into("<II", amd, 0, 0x12345678, 0xABCC)
        report = analyze_bytes(bytes(rtx), bytes(amd))
        self.assertEqual(report["first_causal_mismatch"]["family"], "e4m3")
        self.assertEqual(report["families"][0]["input_exact_output_diff_records"], 1)


if __name__ == "__main__":
    unittest.main()
