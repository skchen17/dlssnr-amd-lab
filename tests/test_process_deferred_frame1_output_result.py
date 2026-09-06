import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.process_deferred_frame1_output_result import EXPECTED_REVISION, process, sha256


class DeferredFrame1OutputResultTests(unittest.TestCase):
    def test_accepts_queue_complete_single_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = bytes(640 * 360 * 8)
            digest = sha256(raw)
            summary = {
                "package_revision": EXPECTED_REVISION,
                "status": "PASS",
                "host_exit": 0,
                "evaluates_ok": 1,
                "feature18_created": True,
                "feature18_evaluation_succeeded": True,
                "deferred_frame1_output": {
                    "inline_snapshots_disabled": True,
                    "bitwise_equal": True,
                    "slot154_input_sha256": digest,
                    "slot155_output_sha256": digest,
                },
            }
            snapshot = {"status": "PASS", "width": 640, "height": 360,
                        "format": 10, "input_bytes": len(raw),
                        "output_bytes": len(raw)}
            events = [{"ev": "nvapi_launch_cu_kernel", "frame": 1, "slot": slot}
                      for slot in range(156)]
            archive = root / "result.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("summary.json", json.dumps(summary))
                zf.writestr("copy_snapshot.json", json.dumps(snapshot))
                zf.writestr("module_trace.jsonl", "\n".join(map(json.dumps, events)))
                zf.writestr("copy_input.raw", raw)
                zf.writestr("copy_output.raw", raw)
            receipt = process(archive, root / "accepted")
            self.assertEqual(receipt["status"], "PASS")
            self.assertTrue(receipt["slot155_copy_bitwise_exact"])


if __name__ == "__main__":
    unittest.main()
