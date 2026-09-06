import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.instrument_n0_fp8_mma_cta_trace import TRACE_BYTES, TRACE_OFFSET, instrument
from scripts.process_n0_fp8_mma_cta_trace_result import process_archive
from scripts.process_slot3_mma_trace_result import sha256


class N0Fp8MmaCtaTraceTest(unittest.TestCase):
    def test_instruments_selected_cta_and_extended_scratch(self):
        source = Path("results/20260831_011219_zluda_ptx_probe/neural_isolated.ptx").read_text(encoding="utf-8")
        output, count = instrument(source, 1, 0)
        self.assertEqual(count, 256)
        self.assertEqual(TRACE_OFFSET, 7864320)
        self.assertIn("setp.eq.u32 %__n0_cta_mma_0_px, %__n0_cta_mma_0_ctax, 1;", output)
        self.assertIn("@%__n0_cta_mma_255_selected st.global.b32", output)

    def test_receiver_accepts_valid_selected_cta_result(self):
        trace = bytes([1]) * TRACE_BYTES
        manifest = {
            "schema": 1, "experiment": "rtx_n0_selected_cta_fp8_mma_trace", "status": "PASS",
            "payload_integrity": True, "probe_exit": 0, "probe_pass": True,
            "baseline_probe_exit": 0, "baseline_probe_pass": True,
            "device_name": "NVIDIA GeForce RTX 5070", "grid": [80, 48, 1], "block": [32, 1, 1],
            "target_cta": [1, 0, 0], "kernel_launched": True, "mma_count": 256,
            "checkpoint_bytes": TRACE_BYTES, "checkpoint_nonzero_bytes": TRACE_BYTES,
            "checkpoint_sha256": sha256(trace), "scratch_extra_bytes": TRACE_BYTES,
            "input_variant": "zero_rgba16f_full_graph", "reference_output_preserved": True,
            "instrumentation_perturbed": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); archive = root / "result.zip"; amd = root / "amd.raw"
            amd.write_bytes(trace)
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("result/manifest.json", json.dumps(manifest))
                zf.writestr("result/mma_trace.raw", trace)
            receipt = process_archive(archive, amd, root / "out")
        self.assertEqual(receipt["status"], "PASS")
        self.assertEqual(receipt["target_cta"], [1, 0, 0])


if __name__ == "__main__":
    unittest.main()
