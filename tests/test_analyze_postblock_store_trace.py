import struct
import unittest

from scripts.analyze_postblock_store_trace import analyze_bytes, reconstruct_rgba16f


def record(values=(1.0, 2.0, 3.0, 4.0), xy=(5, 6), marker=1):
    return struct.pack("<4fIII", *values, *xy, marker) + b"\0" * 4


class PostblockStoreTraceAnalyzerTests(unittest.TestCase):
    def test_exact_trace_passes(self):
        trace = record() + record(marker=0) + record(marker=2) + record(marker=0)
        report = analyze_bytes(trace, trace, threads_per_site=2)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["pre_surface_bitwise_gate"])
        self.assertIsNone(report["first_mismatch"])

    def test_reports_first_value_mismatch(self):
        rtx = record() + record(marker=0) + record(marker=2) + record(marker=0)
        amd = record(values=(1.0, 2.0, 3.5, 4.0)) + record(marker=0) + record(marker=2) + record(marker=0)
        report = analyze_bytes(rtx, amd, threads_per_site=2)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["first_mismatch"]["site"], 0)
        self.assertEqual(report["first_mismatch"]["component"], 2)

    def test_reconstructs_half_surface_by_coordinates(self):
        first = record(values=(1.0, 2.0, 3.0, 4.0), xy=(0, 0))
        second = record(values=(5.0, 6.0, 7.0, 8.0), xy=(1, 0), marker=2)
        output, report = reconstruct_rgba16f(
            first + second, threads_per_site=1, width=2, height=1)
        self.assertEqual(output, struct.pack("<8e", 1, 2, 3, 4, 5, 6, 7, 8))
        self.assertTrue(report["complete_surface"])


if __name__ == "__main__":
    unittest.main()
