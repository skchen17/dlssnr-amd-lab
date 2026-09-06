import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_full_graph_slot34_search import MEASURED_SLOTS, analyze


def analysis(injected, slot6):
    boundaries = {}
    for slot in MEASURED_SLOTS:
        nrmse = slot6 if slot == 6 else slot / 1000
        boundaries[str(slot)] = {
            "rtx_sha256": f"RTX{slot}",
            "comparison": {
                "nrmse_vs_rtx_stddev": nrmse,
                "parity_gate": {"pass": nrmse <= 0.1},
            },
        }
    return {
        "execution_gate": True,
        "amd_repeat_determinism_gate": True,
        "evidence": {"rtx_intermediate_state_injection": injected},
        "boundaries": boundaries,
        "final_comparison": {"nrmse_vs_rtx_stddev": 1.0},
        "final_amd_sha256": "FINAL",
    }


class FullGraphSlot34SearchTests(unittest.TestCase):
    def test_ranks_by_first_failing_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.json"
            baseline.write_text(json.dumps(analysis(False, 0.11)), encoding="utf-8")
            candidates = []
            for target, value in (("%a", 0.105), ("%b", 0.099)):
                case = root / target[1:]
                case.mkdir()
                plan = case / "plan.json"
                plan.write_text(json.dumps({"variant": {"targets": [target]}}), encoding="utf-8")
                (case / "execution.json").write_text(json.dumps({"plan": str(plan)}), encoding="utf-8")
                path = case / "analysis.json"
                path.write_text(json.dumps(analysis(False, value)), encoding="utf-8")
                candidates.append(path)
            result = analyze(baseline, candidates)
            self.assertEqual(result["best"]["target"], "%b")
            self.assertEqual(result["slot6_passing_targets"], ["%b"])
            self.assertFalse(result["counts_as_s7"])


if __name__ == "__main__":
    unittest.main()
