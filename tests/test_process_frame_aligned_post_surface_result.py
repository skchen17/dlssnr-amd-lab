import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts.process_frame_aligned_post_surface_result import process


class FrameAlignedPostSurfaceReceiverTests(unittest.TestCase):
    def test_accepts_valid_before_after_pair(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "scripts.process_frame_aligned_post_surface_result.EXPECTED_BYTES", 8
        ), patch(
            "scripts.process_frame_aligned_post_surface_result.MIN_ARENA_BYTES", 4
        ), patch(
            "scripts.process_frame_aligned_post_surface_result.MAX_ARENA_BYTES", 64
        ), patch(
            "scripts.process_frame_aligned_post_surface_result.EXPECTED_COPY_HASH",
            hashlib.sha256(b"FINAL123").hexdigest().upper()
        ):
            root = Path(directory)
            before, after, activation = b"BEFORE12", b"AFTER123", b"activation"
            base_meta = {"status": "PASS", "capture_frame": 1, "width": 640,
                         "height": 360, "format": 10, "bytes": 8,
                         "surface_resource": "0x123"}
            before_meta = dict(base_meta,
                               classification="FRAME1_SLOT154_OUTPUT_SURFACE_INITIAL")
            after_meta = dict(base_meta,
                              classification="FRAME1_SLOT154_OUTPUT_BEFORE_SLOT155_COPY")
            summary = {
                "package_revision": "v24_feature18_frame_aligned_post_surface_before_after",
                "status": "PASS", "host_exit": 0, "evaluates_ok": 300,
                "feature18_created": True, "feature18_evaluation_succeeded": True,
                "frame1_post_surface_initial_snapshot": {
                    "sha256": hashlib.sha256(before).hexdigest()},
                "frame1_post_output_snapshot": {
                    "sha256": hashlib.sha256(after).hexdigest()},
            }
            events = [
                {"ev": "frame1_post_surface_initial_capture_arm", "frame": 1,
                 "status": "PASS"},
                {"ev": "frame1_post_output_capture_arm", "frame": 1,
                 "status": "PASS"},
            ]
            archive = root / "return.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("r/summary.json", json.dumps(summary))
                zf.writestr("r/module_trace.jsonl",
                            "\n".join(json.dumps(e) for e in events))
                zf.writestr("r/frame1_post_surface_pre_slot154.json",
                            json.dumps(before_meta))
                zf.writestr("r/frame1_post_surface_pre_slot154.raw", before)
                zf.writestr("r/frame1_post_output_pre_copy.json", json.dumps(after_meta))
                zf.writestr("r/frame1_post_output_pre_copy.raw", after)
                zf.writestr("r/post_activation_arena_prelaunch.json",
                            json.dumps({"status": "PASS", "frame": 1,
                                        "bytes": len(activation)}))
                zf.writestr("r/post_activation_arena_prelaunch.raw", activation)
                zf.writestr("r/post_texture_input.raw", bytes(8))
                zf.writestr("r/copy_input.raw", b"FINAL123")
                zf.writestr("r/copy_output.raw", b"FINAL123")
            receipt = process(archive, root / "out")
            self.assertEqual(receipt["status"], "PASS")
            self.assertEqual(receipt["surface_changed_bytes"], 7)


if __name__ == "__main__":
    unittest.main()
