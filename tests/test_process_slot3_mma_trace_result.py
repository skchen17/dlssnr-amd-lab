import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.process_slot3_mma_trace_result import TRACE_BYTES, process_archive


class ProcessSlot3MmaTraceResultTest(unittest.TestCase):
    def make_archive(self, root: Path, *, bad_hash: bool = False,
                     traversal: bool = False) -> tuple[Path, bytes]:
        trace = bytearray(TRACE_BYTES)
        trace[-1] = 1
        trace = bytes(trace)
        digest = hashlib.sha256(trace).hexdigest().upper()
        manifest = {
            "schema": 1,
            "experiment": "rtx5070_slot3_all_fp8_mma_register_trace",
            "status": "PASS",
            "payload_integrity": True,
            "probe_exit": 0,
            "probe_pass": True,
            "device_name": "NVIDIA GeForce RTX 5070",
            "kernel_launched": True,
            "checkpoint_bytes": TRACE_BYTES,
            "checkpoint_nonzero_bytes": 1,
            "checkpoint_sha256": "0" * 64 if bad_hash else digest,
        }
        archive = root / "return.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("_slot3_result/manifest.json", json.dumps(manifest))
            zf.writestr("_slot3_result/mma_trace.raw", trace)
            if traversal:
                zf.writestr("../outside.txt", "unsafe")
        return archive, trace

    def test_valid_archive_is_compared_and_receipted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, trace = self.make_archive(root)
            amd = root / "amd.raw"
            amd.write_bytes(trace)
            output = root / "output"
            receipt = process_archive(archive, amd, output, include_models=False)
            self.assertEqual(receipt["status"], "PASS")
            self.assertTrue(receipt["input_fragment_gate"])
            self.assertTrue(receipt["mma_result_bitwise_gate"])
            self.assertTrue((output / "comparison.json").is_file())
            self.assertTrue((output / "receipt.json").is_file())

    def test_manifest_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, trace = self.make_archive(root, bad_hash=True)
            amd = root / "amd.raw"
            amd.write_bytes(trace)
            with self.assertRaisesRegex(ValueError, "checkpoint_sha256"):
                process_archive(archive, amd, root / "output", include_models=False)

    def test_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, trace = self.make_archive(root, traversal=True)
            amd = root / "amd.raw"
            amd.write_bytes(trace)
            with self.assertRaisesRegex(ValueError, "unsafe ZIP member"):
                process_archive(archive, amd, root / "output", include_models=False)


if __name__ == "__main__":
    unittest.main()
