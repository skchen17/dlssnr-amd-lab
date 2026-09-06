import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts.process_post_texture_snapshot_result import EXPECTED_REVISION, process


class PostTextureSnapshotReceiverTests(unittest.TestCase):
    def make_archive(self, root: Path, *, corrupt_hash: bool = False) -> Path:
        texture = b"TEXTURE!"
        digest = hashlib.sha256(texture).hexdigest()
        summary = {
            "package_revision": EXPECTED_REVISION,
            "status": "PASS",
            "host_exit": 0,
            "evaluates_ok": 300,
            "feature18_created": True,
            "feature18_evaluation_succeeded": True,
            "d3d12_resource_trace": {"post_texture_resource_binds": 1},
            "post_texture_snapshot": {
                "status": "PASS", "width": 1, "height": 1,
                "format": 10, "bytes": 8,
                "nonzero_bytes": sum(value != 0 for value in texture),
                "sha256": "0" * 64 if corrupt_hash else digest,
            },
            "copy_content_snapshot": {
                "input_sha256": digest, "output_sha256": digest,
            },
        }
        snapshot = {
            "status": "PASS",
            "classification": "SLOT154_PRELAUNCH_TEXTURE_INPUT",
            "capture_frame": 1,
            "texture_object": "0x1234", "texture_resource": "0x5678",
            "width": 1, "height": 1, "format": 10,
            "row_size": 8, "num_rows": 1, "bytes": 8,
            "file": "post_texture_input.raw",
        }
        archive = root / "returned.zip"
        resource = "0x5678"
        events = [
            {"ev": "d3d12_resource_create", "resource": resource,
             "width": 1, "height": 1, "format": 10},
            {"ev": "d3d12_create_srv", "resource": resource},
            {"ev": "nvapi_get_cuda_merged_texture_sampler_call",
             "descriptor_resource": resource},
            {"ev": "post_texture_prelaunch_capture_arm", "frame": 1,
             "status": "PASS", "texture_resource": resource},
            {"ev": "post_texture_resource_bind", "frame": 1, "slot": 154,
             "prelaunch_capture": True, "texture_resource": resource},
        ]
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("result/summary.json", json.dumps(summary))
            zf.writestr("result/post_texture_snapshot.json", json.dumps(snapshot))
            zf.writestr("result/post_texture_input.raw", texture)
            zf.writestr("result/copy_input.raw", texture)
            zf.writestr("result/copy_output.raw", texture)
            zf.writestr("result/module_trace.jsonl",
                        "\n".join(json.dumps(event) for event in events))
        return archive

    def test_accepts_valid_snapshot(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "scripts.process_post_texture_snapshot_result.EXPECTED_WIDTH", 1
        ), patch(
            "scripts.process_post_texture_snapshot_result.EXPECTED_HEIGHT", 1
        ), patch(
            "scripts.process_post_texture_snapshot_result.EXPECTED_BYTES", 8
        ), patch(
            "scripts.process_post_texture_snapshot_result.EXPECTED_COPY_HASH",
            hashlib.sha256(b"TEXTURE!").hexdigest().upper()
        ):
            root = Path(directory)
            receipt = process(self.make_archive(root), root / "output")
            self.assertEqual(receipt["status"], "PASS")
            self.assertEqual((root / "output/post_texture_input.raw").read_bytes(), b"TEXTURE!")

    def test_rejects_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "scripts.process_post_texture_snapshot_result.EXPECTED_WIDTH", 1
        ), patch(
            "scripts.process_post_texture_snapshot_result.EXPECTED_HEIGHT", 1
        ), patch(
            "scripts.process_post_texture_snapshot_result.EXPECTED_BYTES", 8
        ), patch(
            "scripts.process_post_texture_snapshot_result.EXPECTED_COPY_HASH",
            hashlib.sha256(b"TEXTURE!").hexdigest().upper()
        ):
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "raw_hash"):
                process(self.make_archive(root, corrupt_hash=True), root / "output")


if __name__ == "__main__":
    unittest.main()
