import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_n0_post_f16_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("analyze_n0_post_f16_checkpoint", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class AnalyzeN0PostF16CheckpointTest(unittest.TestCase):
    def test_exact_fragments_pass(self):
        fragment = bytes((i * 17) & 0xFF for i in range(MODULE.FRAGMENT_BYTES))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rtx, amd = root / "rtx", root / "amd"
            rtx.mkdir()
            amd.mkdir()
            (rtx / "post_f16_fragments.raw").write_bytes(fragment)
            (amd / "scratch.raw").write_bytes(fragment + b"tail")
            (rtx / "manifest.json").write_text(json.dumps({
                "status": "PASS",
                "device_name": "NVIDIA GeForce RTX 5070",
                "fragments_sha256": MODULE.sha256(fragment),
            }), encoding="utf-8")
            report = MODULE.analyze(rtx, amd)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["half_mismatches"], 0)
        self.assertEqual(report["per_mma_half_mismatches"], [0] * 16)

    def test_mismatch_is_assigned_to_one_mma(self):
        left = bytes(MODULE.FRAGMENT_BYTES)
        right = bytearray(left)
        lane, mma, half = 3, 7, 2
        offset = lane * 128 + mma * 8 + half * 2
        right[offset] = 1
        counts = MODULE.per_mma_half_mismatches(left, bytes(right))
        self.assertEqual(counts[mma], 1)
        self.assertEqual(sum(counts), 1)


if __name__ == "__main__":
    unittest.main()
