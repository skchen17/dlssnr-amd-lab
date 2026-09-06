import hashlib
import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.instrument_postblock_r944_snapshot import TRACE_BYTES
from scripts.process_postblock_r944_snapshot_result import EXPECTED_OUTPUT, process


class ProcessPostblockR944SnapshotResultTests(unittest.TestCase):
    def test_accepts_valid_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); trace = bytearray(TRACE_BYTES)
            for lane in range(32):
                struct.pack_into("<7IH", trace, lane * 32, 1, 2, 3, 4, 5, 6, 11, 12)
            trace = bytes(trace); trace_hash = hashlib.sha256(trace).hexdigest().upper()
            runs = [{"probe_exit": 0, "probe_pass": True, "execution_verified": True,
                     "device_name": "NVIDIA GeForce RTX test", "surface_initial_loaded": True,
                     "trace_bytes": TRACE_BYTES, "trace_nonzero_bytes": 1,
                     "trace_sha256": trace_hash, "output_sha256": EXPECTED_OUTPUT,
                     "output_reference_exact": True} for _ in range(2)]
            manifest = {"package_revision":
                        "v1_postblock_exact_frame1_selected_cta_r944_pre_e4_snapshot",
                        "experiment": "rtx_postblock_exact_frame1_selected_cta_r944_pre_e4_snapshot",
                        "status": "PASS", "classification":
                        "RTX_POSTBLOCK_R944_PRE_E4_SNAPSHOT_ORACLE",
                        "payload_integrity": True, "target_cta": [70, 26, 0],
                        "trace_bytes": TRACE_BYTES, "trace_repeat_bitwise_exact": True,
                        "output_reference_sha256": EXPECTED_OUTPUT, "runs": runs}
            archive = root / "result.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("result/manifest.json", json.dumps(manifest))
                zf.writestr("result/run1.trace.raw", trace); zf.writestr("result/run2.trace.raw", trace)
            amd = root / "amd.raw"; amd.write_bytes(trace)
            self.assertEqual(process(archive, root / "accepted", amd)["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
