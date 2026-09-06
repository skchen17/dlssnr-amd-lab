import struct
import unittest

from scripts.analyze_postblock_f16x2_path_trace import analyze_bytes
from scripts.instrument_postblock_f16x2_path_trace import TRACE_BYTES, instrument
from scripts.instrument_postblock_f16x2_trace import (
    EXPECTED_ABS, EXPECTED_BINARY, EXPECTED_FMA,
)


class PostblockF16x2PathTraceTests(unittest.TestCase):
    def test_selects_three_operations_in_source_order(self):
        operations = []
        for index in range(EXPECTED_BINARY):
            dst = "%r816" if index == 0 else "%r817" if index == 32 else "%r944" if index == 64 else "%r1"
            sources = ("%r204, %r205" if index == 0 else "%r720, %r721" if index == 32
                       else "%r816, %r817" if index == 64 else "%r2, %r3")
            op = "add" if index == 64 else "mul"
            operations.append(f"{op}.f16x2 {dst}, {sources};\n")
        operations.extend("fma.rn.f16x2 %r4, %r5, %r6, %r7;\n" for _ in range(EXPECTED_FMA))
        operations.extend("abs.f16x2 %r8, %r9;\n" for _ in range(EXPECTED_ABS))
        text = (".visible .entry cc_tinlayout_fused_post_block_swin_1h_32_fp8(\n"
                ".param .align 8 .b8 cc_tinlayout_fused_post_block_swin_1h_32_fp8_param_0[184]\n)\n"
                "{\n.reg .b64 %rd11;\n" + "".join(operations) + "}\n")
        output, report = instrument(text, 70, 26)
        self.assertTrue(report["selection_matches_expected"])
        self.assertEqual(output.count("__dlssnr_post_f16x2_0_px"), 3)

    def test_analysis_finds_r944_exact_input_output_difference(self):
        rtx = bytearray(TRACE_BYTES)
        amd = bytearray(TRACE_BYTES)
        markers = (1, 1, 2)
        for operation, marker in enumerate(markers):
            for lane in range(32):
                offset = (operation * 32 + lane) * 20
                for trace in (rtx, amd):
                    struct.pack_into("<IIIII", trace, offset, 1, 2, 0, 3, marker)
        struct.pack_into("<I", amd, (2 * 32) * 20 + 12, 4)
        report = analyze_bytes(bytes(rtx), bytes(amd))
        self.assertEqual(report["first_causal_mismatch"]["dst"], "%r944")


if __name__ == "__main__":
    unittest.main()
