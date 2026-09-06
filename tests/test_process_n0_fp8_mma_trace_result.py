import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.process_n0_fp8_mma_trace_result import TRACE_BYTES, process_archive


class N0MmaTraceReceiverTests(unittest.TestCase):
    def test_accepts_valid_archive_and_marks_preserved_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = bytearray(TRACE_BYTES); trace[-1] = 1; trace = bytes(trace)
            manifest = {"schema": 1, "experiment": "rtx5070_n0_all_fp8_mma_register_trace",
                        "status": "PASS", "payload_integrity": True, "probe_exit": 0,
                        "probe_pass": True, "baseline_probe_exit": 0,
                        "baseline_probe_pass": True, "device_name": "NVIDIA GeForce RTX 5070",
                        "grid": [1, 1, 1], "block": [32, 1, 1], "kernel_launched": True,
                        "mma_count": 256, "checkpoint_bytes": TRACE_BYTES,
                        "checkpoint_nonzero_bytes": 1,
                        "checkpoint_sha256": hashlib.sha256(trace).hexdigest().upper(),
                        "reference_output_preserved": True, "instrumentation_perturbed": False}
            archive = root / "result.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("_result/manifest.json", json.dumps(manifest))
                zf.writestr("_result/mma_trace.raw", trace)
            amd = root / "amd.raw"; amd.write_bytes(trace)
            receipt = process_archive(archive, amd, root / "output", include_models=False)
            self.assertEqual(receipt["status"], "PASS")
            self.assertTrue(receipt["reference_output_preserved"])
            self.assertTrue(receipt["input_fragment_gate"])
            self.assertTrue(receipt["mma_result_bitwise_gate"])


if __name__ == "__main__":
    unittest.main()
