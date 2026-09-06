import struct
import unittest

from scripts.analyze_n0_mma176_a_fusion_semantics import STAGE_BYTES, analyze


class AnalyzeN0Mma176AFusionSemanticsTests(unittest.TestCase):
    def test_right_square_separate_hfma_matches_known_clean_rtx_sample(self):
        raw = bytearray(45 * STAGE_BYTES)
        inputs = (0xB817, 0x333E, 0xB37F, 0xAB2E)
        for stage, value in zip((0, 2, 1, 3), inputs):
            struct.pack_into("<H", raw, stage * STAGE_BYTES + 1 * 4 + 2, value)
        struct.pack_into("<H", raw, 10 * STAGE_BYTES + 1 * 4 + 2, 0x35EE)
        report = analyze(bytes(raw))
        self.assertTrue(report["models"]["hfma_pair_right_square_separate"]["bitwise_gate"])
        self.assertFalse(report["models"]["strict_f16_each_op"]["bitwise_gate"])
