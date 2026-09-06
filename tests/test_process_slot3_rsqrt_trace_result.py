import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.process_slot3_rsqrt_trace_result import TRACE_BYTES, process_archive


class ProcessSlot3RsqrtTraceResultTest(unittest.TestCase):
    def make_archive(self, root: Path, traversal: bool = False):
        trace = bytearray(TRACE_BYTES); trace[-1] = 1; trace = bytes(trace)
        blocks = [{"index": i, "source": f"%r{i}", "destination": f"%r{i+1}"} for i in range(16)]
        manifest = {"schema": 1, "experiment": "rtx5070_slot3_rsqrt_register_trace",
                    "status": "PASS", "payload_integrity": True, "probe_exit": 0,
                    "probe_pass": True, "device_name": "NVIDIA GeForce RTX 5070",
                    "cta": [2, 0, 0], "blocks": blocks, "kernel_launched": True,
                    "checkpoint_bytes": TRACE_BYTES, "checkpoint_nonzero_bytes": 1,
                    "checkpoint_sha256": hashlib.sha256(trace).hexdigest().upper()}
        archive = root / "return.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("_result/manifest.json", json.dumps(manifest))
            zf.writestr("_result/rsqrt_trace.raw", trace)
            if traversal: zf.writestr("../outside.txt", "unsafe")
        return archive, trace

    def test_valid(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); archive, trace = self.make_archive(root)
            amd = root / "amd.raw"; amd.write_bytes(trace)
            self.assertTrue(process_archive(archive, amd, root / "out")["bitwise_equal"])

    def test_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); archive, trace = self.make_archive(root, True)
            amd = root / "amd.raw"; amd.write_bytes(trace)
            with self.assertRaisesRegex(ValueError, "unsafe ZIP member"):
                process_archive(archive, amd, root / "out")


if __name__ == "__main__": unittest.main()
