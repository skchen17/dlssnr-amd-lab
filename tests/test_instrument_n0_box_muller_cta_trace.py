import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.analyze_n0_box_muller_cta_trace import sha256
from scripts.instrument_n0_box_muller_cta_trace import BYTES_PER_SAMPLE, SAMPLES, STAGES, TRACE_BYTES, instrument
from scripts.instrument_n0_box_muller_full_cta_trace import (
    STAGES as FULL_STAGES,
    TRACE_BYTES as FULL_TRACE_BYTES,
    instrument as instrument_full,
)
from scripts.process_n0_box_muller_cta_trace_result import process


class InstrumentN0BoxMullerCtaTraceTests(unittest.TestCase):
    def test_instruments_all_stages_after_terminal_conversion(self):
        source = "\n".join(stage[1] for stage in __import__('scripts.instrument_n0_box_muller_cta_trace', fromlist=['STAGES']).STAGES)
        output, count = instrument(source, 8, 0)
        self.assertEqual(count, 8)
        self.assertIn("mad.lo.u32 %__n0_box_trace_0_offset, %r5556, 4, 7864320;", output)
        self.assertIn("st.global.b32 [%__n0_box_trace_7_address], %__n0_box_trace_7_half;", output)
        self.assertEqual(BYTES_PER_SAMPLE, 32)
        self.assertEqual(TRACE_BYTES, 2048)

    def test_receiver_accepts_a_nondefault_selected_cta(self):
        trace = bytes([1]) * TRACE_BYTES
        manifest = {
            "experiment": "rtx_n0_selected_cta_box_muller_trace",
            "status": "PASS",
            "payload_integrity": True,
            "device_name": "NVIDIA GeForce RTX 5070",
            "target_cta": [35, 0, 0],
            "grid": [80, 48, 1],
            "sample_count": SAMPLES,
            "stage_count": len(STAGES),
            "stages": [stage[0] for stage in STAGES],
            "checkpoint_bytes": TRACE_BYTES,
            "checkpoint_sha256": sha256(trace),
            "classification": "CONTROLLED_IMMEDIATE_POST_DEFINITION_APPROX_MATH_TRACE",
            "reference_output_preserved": True,
            "admissible_scope": "captured defining-instruction results only",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "result.zip"
            amd = root / "amd.raw"
            amd.write_bytes(trace)
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("result/manifest.json", json.dumps(manifest))
                zf.writestr("result/box_muller_trace.raw", trace)
            receipt = process(archive, amd, root / "out", [35, 0, 0])
        self.assertEqual(receipt["status"], "PASS")
        self.assertEqual(receipt["target_cta"], [35, 0, 0])

    def test_full_trace_covers_all_three_half_outputs(self):
        source = Path("results/20260831_011219_zluda_ptx_probe/neural_isolated.ptx").read_text(encoding="utf-8")
        output, count = instrument_full(source, 35, 0)
        self.assertEqual(count, len(FULL_STAGES))
        self.assertEqual(FULL_TRACE_BYTES, 5632)
        self.assertIn("st.global.b32 [%__n0_box_trace_21_address], %__n0_box_trace_21_half;", output)
        self.assertIn("setp.eq.u32 %__n0_box_trace_0_px, %__n0_box_trace_0_ctax, 35;", output)


if __name__ == "__main__":
    unittest.main()
