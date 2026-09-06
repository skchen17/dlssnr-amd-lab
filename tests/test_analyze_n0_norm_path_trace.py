import tempfile
import unittest
from pathlib import Path

from scripts.analyze_n0_norm_path_trace import analyze
from scripts.instrument_n0_norm_path_trace import LANES, TRACE_BYTES


class AnalyzeN0NormPathTraceTest(unittest.TestCase):
    def test_localizes_first_changed_stage_and_lane(self):
        rtx = bytearray(TRACE_BYTES)
        amd = bytearray(rtx)
        amd[(4 * LANES + 7) * 4 + 2] = 1
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rtx.raw").write_bytes(rtx)
            (root / "amd.raw").write_bytes(amd)
            report = analyze(root / "rtx.raw", root / "amd.raw")
        self.assertEqual(report["status"], "LOCALIZED")
        self.assertEqual(report["first_divergent_stage"], 4)
        self.assertEqual(report["stages"][4]["mismatches"][0]["lane"], 7)


if __name__ == "__main__":
    unittest.main()
