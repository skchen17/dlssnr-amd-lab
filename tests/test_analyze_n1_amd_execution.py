import importlib.util
import struct
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_n1_amd_execution.py"
SPEC = importlib.util.spec_from_file_location("analyze_n1_amd_execution", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class AnalyzeN1AmdExecutionTest(unittest.TestCase):
    def test_sync_sentinel_summary(self):
        words = [0] * MODULE.EXPECTED_RELEASES
        words += [0xFFFFFFFF] * (MODULE.SYNC_WORDS - MODULE.EXPECTED_RELEASES)
        report = MODULE.sync_summary(struct.pack(f"<{MODULE.SYNC_WORDS}I", *words))
        self.assertEqual(report["zero_words"], MODULE.EXPECTED_RELEASES)
        self.assertEqual(report["sentinel_words"], MODULE.SYNC_WORDS - MODULE.EXPECTED_RELEASES)
        self.assertEqual(report["unexpected_words"], 0)


if __name__ == "__main__":
    unittest.main()
