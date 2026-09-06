import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_n0_capture import analyze, fnv1a64


class N0CaptureAnalyzerTest(unittest.TestCase):
    def test_strict_activity_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            windows = []
            for name, offset, before, after in (
                ("scratch", 216, b"\0" * 8, b"\0\1" + b"\0" * 6),
                ("weights", 224, b"weights!", b"weights!"),
                ("output", 248, b"\0" * 8, b"\2\3" + b"\0" * 6),
            ):
                before_file = f"n0_{name}_before.raw"
                after_file = f"n0_{name}_after.raw"
                (root / before_file).write_bytes(before)
                (root / after_file).write_bytes(after)
                positions = [i for i, pair in enumerate(zip(before, after)) if pair[0] != pair[1]]
                windows.append(
                    {
                        "name": name,
                        "param_offset": offset,
                        "capture_bytes": len(before),
                        "before_file": before_file,
                        "after_file": after_file,
                        "before_fnv1a64": fnv1a64(before),
                        "after_fnv1a64": fnv1a64(after),
                        "changed_bytes": len(positions),
                        "first_changed_offset": positions[0] if positions else -1,
                        "last_changed_offset": positions[-1] if positions else -1,
                    }
                )
            metadata = {
                "status": "PASS",
                "function": "preblock",
                "frame": 1,
                "slot": 1,
                "padded_height": 384,
                "padded_width": 640,
                "windows": windows,
            }
            path = root / "n0_preblock_capture.json"
            path.write_text(json.dumps(metadata), encoding="utf-8")
            result = analyze(path, require_activity=True)
            self.assertEqual(result["status"], "PASS")
            self.assertTrue(result["neural_write_activity"])
            self.assertTrue(result["read_only_weight_observed"])


if __name__ == "__main__":
    unittest.main()
