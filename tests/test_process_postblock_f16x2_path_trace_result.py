import hashlib
import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.instrument_postblock_f16x2_path_trace import TRACE_BYTES
from scripts.process_postblock_f16x2_path_trace_result import EXPECTED_OUTPUT, process


class ProcessPostblockF16x2PathTraceResultTests(unittest.TestCase):
    def test_accepts_valid_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace = bytearray(TRACE_BYTES)
            for operation, marker in enumerate((1, 1, 2)):
                for lane in range(32):
                    struct.pack_into("<IIIII", trace, (operation * 32 + lane) * 20,
                                     1, 2, 0, 3, marker)
            trace = bytes(trace)
            trace_hash = hashlib.sha256(trace).hexdigest().upper()
            runs = [{
                "probe_exit": 0, "probe_pass": True, "execution_verified": True,
                "device_name": "NVIDIA GeForce RTX test", "surface_initial_loaded": True,
                "trace_bytes": TRACE_BYTES, "trace_nonzero_bytes": 1,
                "trace_sha256": trace_hash, "output_sha256": EXPECTED_OUTPUT,
                "output_reference_exact": True,
            } for _ in range(2)]
            manifest = {
                "package_revision":
                    "v1_postblock_exact_frame1_selected_cta_f16x2_r944_path_trace",
                "experiment": "rtx_postblock_exact_frame1_selected_cta_f16x2_r944_path_trace",
                "status": "PASS", "classification":
                    "RTX_POSTBLOCK_F16X2_R944_PATH_ORACLE",
                "payload_integrity": True, "target_cta": [70, 26, 0],
                "trace_bytes": TRACE_BYTES, "trace_repeat_bitwise_exact": True,
                "output_reference_sha256": EXPECTED_OUTPUT, "runs": runs,
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
            self.assertIsNone(receipt["first_causal_mismatch"])


if __name__ == "__main__":
    unittest.main()
