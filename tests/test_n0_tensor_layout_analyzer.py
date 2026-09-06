import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_n0_tensor_layout import analyze


class N0TensorLayoutAnalyzerTest(unittest.TestCase):
    def test_shape_factorization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scratch = bytes([0, 1]) * (4 * 4 * 32 // 2)
            output = bytes([2, 3]) * (2 * 2 * 32 // 2)
            weights = b"weight"
            for name, data in (("scratch_after.raw", scratch),
                               ("output_after.raw", output),
                               ("weights_before.raw", weights),
                               ("weights_after.raw", weights)):
                (root / name).write_bytes(data)
            metadata = {
                "status": "PASS",
                "function": "preblock",
                "padded_height": 4,
                "padded_width": 4,
                "windows": [
                    {"name": "scratch", "after_file": "scratch_after.raw"},
                    {"name": "output", "after_file": "output_after.raw"},
                    {"name": "weights", "before_file": "weights_before.raw",
                     "after_file": "weights_after.raw"},
                ],
            }
            path = root / "capture.json"
            path.write_text(json.dumps(metadata), encoding="utf-8")
            result = analyze(path)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["scratch"]["shape"], [4, 4, 32])
            self.assertEqual(result["output"]["shape"], [2, 2, 32])


if __name__ == "__main__":
    unittest.main()
