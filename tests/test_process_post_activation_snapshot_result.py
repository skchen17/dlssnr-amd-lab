import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts.process_post_activation_snapshot_result import process


class PostActivationSnapshotReceiverTests(unittest.TestCase):
    def test_accepts_valid_capture(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "scripts.process_post_activation_snapshot_result.MIN_ARENA_BYTES", 4
        ), patch(
            "scripts.process_post_activation_snapshot_result.MAX_ARENA_BYTES", 64
        ), patch(
            "scripts.process_post_activation_snapshot_result.EXPECTED_COPY_HASH",
            hashlib.sha256(b"copy").hexdigest().upper()
        ):
            root = Path(directory)
            arena = b"live-state"
            summary = {
                "package_revision": "v22_feature18_slot154_full_activation_snapshot",
                "status": "PASS", "host_exit": 0, "evaluates_ok": 300,
                "feature18_created": True, "feature18_evaluation_succeeded": True,
                "post_activation_snapshot": {
                    "bytes": len(arena), "sha256": hashlib.sha256(arena).hexdigest()},
            }
            metadata = {
                "status": "PASS",
                "classification": "SLOT154_PRELAUNCH_FULL_ACTIVATION_BUFFER",
                "frame": 1, "resource": "0x123", "bytes": len(arena),
                "file": "post_activation_arena_prelaunch.raw",
            }
            arm = {"ev": "post_activation_prelaunch_capture_arm", "frame": 1,
                   "status": "PASS", "resource": "0x123", "bytes": len(arena),
                   "offset0": 13873152, "offset8": 110592}
            archive = root / "return.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("r/summary.json", json.dumps(summary))
                zf.writestr("r/post_activation_arena_prelaunch.json", json.dumps(metadata))
                zf.writestr("r/module_trace.jsonl", json.dumps(arm))
                zf.writestr("r/post_activation_arena_prelaunch.raw", arena)
                zf.writestr("r/post_texture_input.raw", bytes(640 * 360 * 8))
                zf.writestr("r/copy_input.raw", b"copy")
                zf.writestr("r/copy_output.raw", b"copy")
            old = root / "old.raw"
            old.write_bytes(b"old-state!")
            receipt = process(archive, root / "out", old)
            self.assertEqual(receipt["status"], "PASS")
            self.assertFalse(receipt["reconstruction_exact"])


if __name__ == "__main__":
    unittest.main()
