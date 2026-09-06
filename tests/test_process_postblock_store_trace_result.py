import hashlib
import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts.process_postblock_store_trace_result import process


def trace_bytes():
    record = struct.pack("<4fIII", 1.0, 2.0, 3.0, 4.0, 5, 6, 1) + b"\0" * 4
    zero = b"\0" * 32
    return record + zero + record + zero


class PostblockStoreTraceReceiverTests(unittest.TestCase):
    def run_case(self, output_preserved=True):
        with tempfile.TemporaryDirectory() as directory, patch(
            "scripts.process_postblock_store_trace_result.EXPECTED_TRACE_BYTES", 128
        ), patch(
            "scripts.process_postblock_store_trace_result.EXPECTED_THREADS_PER_SITE", 2
        ), patch(
            "scripts.process_postblock_store_trace_result.EXPECTED_LOGICAL_OUTPUT_BYTES", 8
        ), patch(
            "scripts.process_postblock_store_trace_result.analyze_bytes",
            side_effect=lambda r, a: __import__(
                "scripts.analyze_postblock_store_trace", fromlist=["analyze_bytes"]
            ).analyze_bytes(r, a, threads_per_site=2),
        ), patch(
            "scripts.process_postblock_store_trace_result.reconstruct_rgba16f",
            return_value=(b"OUTPUT00", {
                "active_records": 2, "unique_coordinates": 2,
                "expected_pixels": 2, "complete_surface": True,
            }),
        ):
            root = Path(directory)
            trace = trace_bytes()
            digest = hashlib.sha256(trace).hexdigest().upper()
            output_digest = hashlib.sha256(b"OUTPUT00").hexdigest().upper()
            runs = [{
                "run": index, "probe_exit": 0, "probe_pass": True,
                "execution_verified": True, "device_name": "NVIDIA GeForce RTX 5070",
                "trace_sha256": digest, "output_sha256": output_digest,
            } for index in (1, 2)]
            manifest = {
                "experiment": "rtx_postblock_pre_surface_store_full_grid_trace",
                "status": "PASS" if output_preserved else "FAIL", "payload_integrity": True,
                "grid": [81, 49, 1], "block": [32, 1, 1],
                "record_bytes": 32, "threads_per_site": 2, "trace_bytes": len(trace),
                "trace_repeat_bitwise_exact": True, "trace_nonzero": True,
                "output_reference_preserved": output_preserved, "runs": runs,
            }
            archive = root / "returned.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("result/manifest.json", json.dumps(manifest))
                zf.writestr("result/trace_run1.raw", trace)
                zf.writestr("result/trace_run2.raw", trace)
                zf.writestr("result/output_run1.raw", b"OUTPUT00")
                zf.writestr("result/output_run2.raw", b"OUTPUT00")
            amd = root / "amd.raw"
            amd.write_bytes(trace)
            receipt = process(archive, amd, root / "output")
            return receipt

    def test_accepts_valid_repeat_archive(self):
        receipt = self.run_case()
        self.assertEqual(receipt["status"], "PASS")
        self.assertTrue(receipt["pre_surface_bitwise_gate"])

    def test_accepts_diagnostic_when_only_original_output_control_fails(self):
        receipt = self.run_case(output_preserved=False)
        self.assertEqual(receipt["status"], "DIAGNOSTIC_VALID")
        self.assertTrue(receipt["diagnostic_valid"])
        self.assertFalse(receipt["reference_output_preserved"])


if __name__ == "__main__":
    unittest.main()
