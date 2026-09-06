import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts.process_frame_aligned_post_output_result import process


class FrameAlignedPostOutputReceiverTests(unittest.TestCase):
    def make_archive(self, path: Path, *, include_output: bool = True) -> None:
        frame = b"FRAME123"
        activation = b"activation"
        final = b"FINAL123"
        summary = {
            "package_revision": "v23_feature18_frame_aligned_post_output",
            "status": "PASS", "host_exit": 0, "evaluates_ok": 300,
            "feature18_created": True, "feature18_evaluation_succeeded": True,
            "frame1_post_output_snapshot": {
                "bytes": len(frame), "sha256": hashlib.sha256(frame).hexdigest()},
        }
        metadata = {
            "status": "PASS",
            "classification": "FRAME1_SLOT154_OUTPUT_BEFORE_SLOT155_COPY",
            "capture_frame": 1, "width": 640, "height": 360, "format": 10,
            "bytes": len(frame), "file": "frame1_post_output_pre_copy.raw",
        }
        activation_metadata = {"status": "PASS", "frame": 1,
                               "bytes": len(activation)}
        arm = {"ev": "frame1_post_output_capture_arm", "frame": 1,
               "status": "PASS"}
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("r/summary.json", json.dumps(summary))
            zf.writestr("r/module_trace.jsonl", json.dumps(arm))
            zf.writestr("r/frame1_post_output_pre_copy.json", json.dumps(metadata))
            if include_output:
                zf.writestr("r/frame1_post_output_pre_copy.raw", frame)
            zf.writestr("r/post_activation_arena_prelaunch.json",
                        json.dumps(activation_metadata))
            zf.writestr("r/post_activation_arena_prelaunch.raw", activation)
            zf.writestr("r/post_texture_input.raw", bytes(len(frame)))
            zf.writestr("r/copy_input.raw", final)
            zf.writestr("r/copy_output.raw", final)

    def test_accepts_valid_frame_aligned_capture(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "scripts.process_frame_aligned_post_output_result.EXPECTED_BYTES", 8
        ), patch(
            "scripts.process_frame_aligned_post_output_result.MIN_ARENA_BYTES", 4
        ), patch(
            "scripts.process_frame_aligned_post_output_result.MAX_ARENA_BYTES", 64
        ), patch(
            "scripts.process_frame_aligned_post_output_result.EXPECTED_COPY_HASH",
            hashlib.sha256(b"FINAL123").hexdigest().upper()
        ):
            root = Path(directory)
            archive = root / "return.zip"
            self.make_archive(archive)
            receipt = process(archive, root / "out")
            self.assertEqual(receipt["status"], "PASS")
            self.assertFalse(receipt["frame1_equals_frame300"])
            self.assertEqual((root / "out/frame1_post_output_pre_copy.raw").read_bytes(),
                             b"FRAME123")

    def test_rejects_missing_frame_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "return.zip"
            self.make_archive(archive, include_output=False)
            with self.assertRaisesRegex(ValueError, "expected one"):
                process(archive, root / "out")


if __name__ == "__main__":
    unittest.main()
