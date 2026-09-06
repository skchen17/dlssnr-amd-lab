import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.instrument_slot3_f16_path_trace import REGISTERS
from scripts.process_slot3_f16_path_trace_result import TRACE_BYTES, process_archive


class ProcessSlot3F16PathTraceResultTest(unittest.TestCase):
    def make_archive(self, root: Path, *, traversal: bool = False) -> tuple[Path, bytes]:
        trace = bytearray(TRACE_BYTES)
        trace[-1] = 1
        trace = bytes(trace)
        manifest = {
            "schema": 1,
            "experiment": "rtx5070_slot3_f16_path_register_trace",
            "status": "PASS",
            "payload_integrity": True,
            "probe_exit": 0,
            "probe_pass": True,
            "device_name": "NVIDIA GeForce RTX 5070",
            "cta": [2, 0, 0],
            "registers": REGISTERS,
            "kernel_launched": True,
            "checkpoint_bytes": TRACE_BYTES,
            "checkpoint_nonzero_bytes": 1,
            "checkpoint_sha256": hashlib.sha256(trace).hexdigest().upper(),
        }
        archive = root / "return.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("_result/manifest.json", json.dumps(manifest))
            zf.writestr("_result/f16_path_trace.raw", trace)
            if traversal:
                zf.writestr("../outside.txt", "unsafe")
        return archive, trace

    def test_valid_archive_is_compared(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, trace = self.make_archive(root)
            amd = root / "amd.raw"
            amd.write_bytes(trace)
            receipt = process_archive(archive, amd, root / "output")
            self.assertTrue(receipt["bitwise_equal"])
            self.assertEqual(receipt["word_mismatches"], 0)

    def test_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, trace = self.make_archive(root, traversal=True)
            amd = root / "amd.raw"
            amd.write_bytes(trace)
            with self.assertRaisesRegex(ValueError, "unsafe ZIP member"):
                process_archive(archive, amd, root / "output")


if __name__ == "__main__":
    unittest.main()
