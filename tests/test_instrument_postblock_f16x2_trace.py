import struct
import unittest

from scripts.analyze_postblock_f16x2_trace import analyze_bytes
from scripts.instrument_postblock_f16x2_trace import (
    EXPECTED_ABS, EXPECTED_BINARY, EXPECTED_FMA, TRACE_BYTES, instrument,
)


class PostblockF16x2TraceTests(unittest.TestCase):
    def test_instrument_counts_and_preserves_source_order(self):
        binary = "mul.f16x2 %r1, %r2, %r3;\n" * EXPECTED_BINARY
        fma = "fma.rn.f16x2 %r4, %r5, %r6, %r7;\n" * EXPECTED_FMA
        absolute = "abs.f16x2 %r8, %r9;\n" * EXPECTED_ABS
        text = (".visible .entry cc_tinlayout_fused_post_block_swin_1h_32_fp8(\n"
                ".param .align 8 .b8 cc_tinlayout_fused_post_block_swin_1h_32_fp8_param_0[184]\n)\n"
                "{\n.reg .b64 %rd11;\n" + binary + fma + absolute + "}\n")
        output, report = instrument(text, 70, 26)
        self.assertIn("param_0[192]", output)
        self.assertEqual(report["operation_count"], 1522)
        self.assertEqual(report["counts"],
                         {"binary": EXPECTED_BINARY, "fma": EXPECTED_FMA,
                          "abs": EXPECTED_ABS})
        self.assertLess(output.index("__dlssnr_post_f16x2_1073"),
                        output.index("__dlssnr_post_f16x2_1074"))

    def test_analysis_finds_exact_input_output_difference(self):
        rtx = bytearray(TRACE_BYTES)
        amd = bytearray(TRACE_BYTES)
        for trace in (rtx, amd):
            for offset in range(0, TRACE_BYTES, 20):
                struct.pack_into("<IIIII", trace, offset, 1, 2, 0, 3, 1)
        struct.pack_into("<I", amd, 12, 4)
        report = analyze_bytes(bytes(rtx), bytes(amd))
        self.assertEqual(report["first_causal_mismatch"]["operation"], 0)
        self.assertEqual(report["first_causal_mismatch"]["kind"], "mul")


if __name__ == "__main__":
    unittest.main()
