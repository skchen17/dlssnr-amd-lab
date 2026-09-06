import struct
import unittest

from scripts.analyze_postblock_r944_snapshot import analyze_bytes
from scripts.instrument_postblock_r944_snapshot import TRACE_BYTES, instrument


class PostblockR944SnapshotTests(unittest.TestCase):
    def test_instruments_only_conversion_four(self):
        conversions = []
        for index in range(388):
            src = "%r944" if index == 4 else "%r1"
            dst = "%rs53" if index == 4 else "%rs1"
            conversions.append(f"cvt.rn.satfinite.e4m3x2.f16x2 {dst}, {src};\n")
        text = (".visible .entry cc_tinlayout_fused_post_block_swin_1h_32_fp8(\n"
                ".param .align 8 .b8 cc_tinlayout_fused_post_block_swin_1h_32_fp8_param_0[184]\n)\n"
                "{\n.reg .b64 %rd11;\n" + "".join(conversions) + "}\n")
        output, report = instrument(text, 70, 26)
        self.assertTrue(report["selected_registers_match_expected"])
        self.assertEqual(output.count("__dlssnr_post_r944_snapshot_px"), 3)

    def test_analysis_localizes_r816_multiply(self):
        rtx = bytearray(TRACE_BYTES)
        amd = bytearray(TRACE_BYTES)
        for trace in (rtx, amd):
            for lane in range(32):
                struct.pack_into("<7IH", trace, lane * 32, 1, 2, 3, 4, 5, 6, 11, 12)
        struct.pack_into("<I", amd, 16, 7)
        report = analyze_bytes(bytes(rtx), bytes(amd))
        self.assertEqual(report["first_causal_mismatch"]["stage"], "mul_r816")


if __name__ == "__main__":
    unittest.main()
