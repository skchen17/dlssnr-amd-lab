import unittest

from scripts.analyze_full_graph_variant_delta import analyze


class FullGraphVariantDeltaTests(unittest.TestCase):
    def test_reports_boundary_and_final_deltas(self):
        def manifest(nrmse, digest):
            return {"execution_gate": True, "counts_as_s7": False,
                    "final_amd_sha256": digest,
                    "final_comparison": {"nrmse_vs_rtx_stddev": nrmse},
                    "boundaries": {"5": {"amd_sha256": digest,
                        "comparison": {"nrmse_vs_rtx_stddev": nrmse / 10}}}}
        report = analyze(manifest(1.0, "A"), manifest(0.9, "B"))
        self.assertAlmostEqual(report["final_relative_nrmse_improvement"], 0.1)
        self.assertAlmostEqual(report["boundaries"][0]["nrmse_delta"], -0.01)
        self.assertFalse(report["boundaries"][0]["bitwise_equal"])
        self.assertEqual(report["improved_boundary_slots"], [5])
        self.assertEqual(report["worsened_boundary_slots"], [])
        self.assertIn("no worsened measured boundary", report["verdict"])


if __name__ == "__main__":
    unittest.main()
