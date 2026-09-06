import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_full_graph_injection_sensitivity import analyze


def report(injected, values):
    boundaries = {}
    for slot, (nrmse, passed) in values.items():
        boundaries[str(slot)] = {
            "tensor_type": "fp16",
            "rtx_sha256": f"RTX{slot}",
            "comparison": {
                "nrmse_vs_rtx_stddev": nrmse,
                "parity_gate": {"pass": passed},
            },
        }
    return {
        "counts_as_s7": False,
        "execution_gate": not injected,
        "amd_repeat_determinism_gate": True,
        "evidence": {"rtx_intermediate_state_injection": injected},
        "boundaries": boundaries,
        "final_comparison": {"nrmse_vs_rtx_stddev": 1.0},
        "final_rgba16f_image_gate": False,
    }


class FullGraphInjectionSensitivityTests(unittest.TestCase):
    def test_finds_first_failure_and_upstream_sensitive_2h_family(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native = root / "native.json"
            injected = root / "injected.json"
            native.write_text(json.dumps(report(False, {
                5: (0.02, True), 6: (0.11, False), 7: (0.20, False), 8: (0.30, False)
            })), encoding="utf-8")
            injected.write_text(json.dumps(report(True, {
                5: (0.02, True), 6: (0.01, True), 7: (0.02, True), 8: (0.03, True)
            })), encoding="utf-8")
            result = analyze(native, {5: injected})
            self.assertEqual(result["native_first_strict_failure"], 6)
            self.assertTrue(result["findings"]["inject_after_slot5_makes_slots6_8_pass"])
            self.assertFalse(result["counts_as_s7"])


if __name__ == "__main__":
    unittest.main()
