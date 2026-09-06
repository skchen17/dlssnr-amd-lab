import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.instrument_postblock_mma_trace import TRACE_BYTES
from scripts.process_postblock_mma_trace_result import (
    EXPECTED_OUTPUT, EXPECTED_REVISION, process, sha256,
)


class ProcessPostblockMmaTraceResultTests(unittest.TestCase):
    def test_accepts_controlled_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = bytes([1]) + bytes(TRACE_BYTES - 1)
            run = {
                "probe_exit": 0, "probe_pass": True, "execution_verified": True,
                "device_name": "NVIDIA Test", "surface_initial_loaded": True,
                "trace_bytes": TRACE_BYTES, "trace_nonzero_bytes": 1,
                "trace_sha256": sha256(trace), "output_sha256": EXPECTED_OUTPUT,
                "output_reference_exact": True,
            }
            manifest = {
                "package_revision": EXPECTED_REVISION,
                "experiment": "rtx_postblock_exact_frame1_selected_cta_mma_trace",
                "status": "PASS", "classification": "RTX_POSTBLOCK_FP8_F16_MMA_ORACLE",
                "payload_integrity": True, "target_cta": [70, 26, 0],
                "trace_bytes": TRACE_BYTES, "trace_repeat_bitwise_exact": True,
                "output_reference_sha256": EXPECTED_OUTPUT,
                "runs": [dict(run, run=1), dict(run, run=2)],
            }
            archive = root / "result.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("result/manifest.json", json.dumps(manifest))
                zf.writestr("result/run1.trace.raw", trace)
                zf.writestr("result/run2.trace.raw", trace)
            amd = root / "amd.raw"
            amd.write_bytes(trace)
            receipt = process(archive, root / "accepted", amd)
            self.assertEqual(receipt["status"], "PASS")
            self.assertTrue(receipt["comparison_bitwise_exact"])


if __name__ == "__main__":
    unittest.main()
