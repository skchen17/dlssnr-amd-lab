import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_full_graph_partial_injection_sweep import MEASURED_SLOTS, analyze


def report(slot6, injected):
    boundaries = {}
    for slot in (3, *MEASURED_SLOTS):
        value = slot6 if slot == 6 else slot / 1000
        boundaries[str(slot)] = {
            "rtx_sha256": f"RTX{slot}",
            "comparison": {
                "nrmse_vs_rtx_stddev": value,
                "parity_gate": {"pass": value <= 0.1},
            },
        }
    return {
        "execution_gate": not injected,
        "boundaries": boundaries,
        "final_comparison": {"nrmse_vs_rtx_stddev": 1.0},
        "final_amd_sha256": "FINAL",
    }


class PartialInjectionSweepTests(unittest.TestCase):
    def test_full_coverage_and_ranking(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            baseline.mkdir()
            (baseline / "run1").mkdir()
            (baseline / "run1" / "slot3.raw").write_bytes(b"abcdefgh")
            baseline_analysis = baseline / "analysis.json"
            baseline_analysis.write_text(json.dumps(report(0.11, False)), encoding="utf-8")
            (baseline / "execution.json").write_text(json.dumps({
                "pass": True, "rtx_intermediate_state_injection": False,
            }), encoding="utf-8")

            reference = root / "reference.raw"
            reference.write_bytes(b"abXdefgY")
            candidates = []
            for index, (offset, value) in enumerate(((0, 0.099), (4, 0.097))):
                case = root / f"case{index}"
                case.mkdir()
                data = reference.read_bytes()[offset:offset + 4]
                plan = case / "plan.json"
                plan.write_text(json.dumps({
                    "checkpoints": {"3": {"logical_bytes": 8}},
                    "diagnostic_injections": [{
                        "reference": {"path": "../reference.raw"},
                    }],
                }), encoding="utf-8")
                (case / "execution.json").write_text(json.dumps({
                    "pass": True,
                    "rtx_intermediate_state_injection": True,
                    "plan": str(plan),
                    "diagnostic_injections": [{
                        "after_slot": 3,
                        "reference_offset": offset,
                        "bytes": 4,
                        "sha256": hashlib.sha256(data).hexdigest().upper(),
                    }],
                    "runs": [{"final_sha256": "A"}, {"final_sha256": "A"}],
                }), encoding="utf-8")
                path = case / "analysis.json"
                path.write_text(json.dumps(report(value, True)), encoding="utf-8")
                candidates.append(path)

            result = analyze(baseline_analysis, candidates)
            self.assertTrue(result["range_coverage"]["full_tensor"])
            self.assertTrue(result["all_ranges_make_slot6_pass"])
            self.assertEqual(result["best_slot6_range"]["reference_offset"], 4)
            self.assertEqual(result["slot6_passing_range_count"], 2)
            self.assertFalse(result["counts_as_s7"])


if __name__ == "__main__":
    unittest.main()
