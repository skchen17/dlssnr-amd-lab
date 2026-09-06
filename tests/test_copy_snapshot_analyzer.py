import json
import struct
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_copy_snapshot import analyze


class CopySnapshotAnalyzerTest(unittest.TestCase):
    def test_identity_rgba16f(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = b"".join(struct.pack("<e", float(i) / 16.0) for i in range(32))
            (root / "copy_input.raw").write_bytes(raw)
            (root / "copy_output.raw").write_bytes(raw)
            metadata = {
                "status": "PASS",
                "width": 4,
                "height": 2,
                "format": 10,
                "input_file": "copy_input.raw",
                "output_file": "copy_output.raw",
            }
            path = root / "copy_snapshot.json"
            path.write_text(json.dumps(metadata), encoding="utf-8")
            result = analyze(path)
            self.assertEqual(result["status"], "PASS")
            self.assertTrue(result["bitwise_equal"])
            self.assertEqual(result["scalar_mismatches"], 0)
            self.assertEqual(result["max_abs_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
