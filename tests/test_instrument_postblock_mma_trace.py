import struct
import unittest

from scripts.analyze_postblock_mma_trace import analyze_bytes
from scripts.instrument_postblock_mma_trace import (
    F16_COUNT, FP8_COUNT, TRACE_BYTES, instrument,
)


def _mma(kind: str) -> str:
    if kind == "fp8":
        op = "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16"
    else:
        op = "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
    return f"{op} {{%r9,%r10}}, {{%r1,%r2,%r3,%r4}}, {{%r5,%r6}}, {{%r7,%r8}};"


class PostblockMmaTraceTests(unittest.TestCase):
    def test_instrument_counts_and_expands_parameter_block(self):
        text = (".visible .entry cc_tinlayout_fused_post_block_swin_1h_32_fp8(\n"
                ".param .align 8 .b8 cc_tinlayout_fused_post_block_swin_1h_32_fp8_param_0[184]\n)\n"
                "{\n.reg .b64 %rd11;\n" + (_mma("fp8") + "\n") * FP8_COUNT +
                (_mma("f16") + "\n") * F16_COUNT + "}\n")
        output, counts = instrument(text, 70, 26)
        self.assertIn("param_0[192]", output)
        self.assertEqual(counts["fp8_mma_instrumented"], FP8_COUNT)
        self.assertEqual(counts["f16_mma_instrumented"], F16_COUNT)
        self.assertIn(
            "setp.eq.u32 %__dlssnr_post_fp8_0_px, %__dlssnr_post_fp8_0_x, 70;",
            output)

    def test_analysis_finds_output_divergence_with_exact_inputs(self):
        rtx = bytearray(TRACE_BYTES)
        amd = bytearray(TRACE_BYTES)
        struct.pack_into("<10I", rtx, 0, *range(1, 11))
        struct.pack_into("<10I", amd, 0, *range(1, 10), 99)
        report = analyze_bytes(bytes(rtx), bytes(amd))
        self.assertFalse(report["bitwise_exact"])
        self.assertEqual(
            report["first_output_mismatch_with_local_record_inputs_exact"]["family"],
            "fp8")
        self.assertIsNotNone(report["first_output_mismatch_with_warp_inputs_exact"])
        self.assertEqual(report["families"][0]["input_word_exact_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
