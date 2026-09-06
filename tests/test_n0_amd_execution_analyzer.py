import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_n0_amd_execution.py"
SPEC = importlib.util.spec_from_file_location("n0_amd_execution", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class N0AMDExecutionAnalyzerTest(unittest.TestCase):
    def test_raw_summary_is_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.bin"
            path.write_bytes(b"\x00\x01\x00\xff")
            summary = MODULE.raw_summary(path)
            self.assertEqual(summary["bytes"], 4)
            self.assertEqual(summary["nonzero_bytes"], 2)
            self.assertEqual(
                summary["sha256"], hashlib.sha256(path.read_bytes()).hexdigest().upper()
            )

    def test_successful_step_requires_zero_code(self):
        probe = {"steps": [{"name": "launch", "code": 0}]}
        self.assertTrue(MODULE.successful_step(probe, "launch"))
        self.assertFalse(MODULE.successful_step(probe, "sync"))


if __name__ == "__main__":
    unittest.main()
