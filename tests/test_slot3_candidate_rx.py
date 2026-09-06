import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_slot3_candidate_rx import analyze
from scripts.run_slot3_candidate_rx import probe_arguments


class Slot3CandidateRxTest(unittest.TestCase):
    def test_exact_candidate_is_ranked_before_changed_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_root = root / "candidates"
            candidate_root.mkdir()
            reference = bytes(range(64))
            changed = bytearray(reference)
            changed[3] = 20
            changed = bytes(changed)
            reference_path = root / "reference.raw"
            baseline_path = root / "baseline.raw"
            reference_path.write_bytes(reference)
            baseline_path.write_bytes(changed)
            executions = []
            for name, output in (("exact", reference), ("changed", changed)):
                directory_path = candidate_root / name
                directory_path.mkdir()
                (directory_path / "output.raw").write_bytes(output)
                executions.append({
                    "candidate_model": name,
                    "status": "PASS",
                    "device_name": "AMD Radeon RX 9070 XT [ZLUDA]",
                })
            (candidate_root / "summary.json").write_text(
                json.dumps({"candidates": executions}), encoding="utf-8"
            )
            report = analyze(candidate_root, reference_path, baseline_path)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["best_candidate"], "exact")
        self.assertTrue(report["best_candidate_improves_nrmse"])

    def test_probe_arguments_encode_captured_slot3_abi(self):
        args = probe_arguments(Path("C:/repo"), Path("C:/candidate.ptx"), Path("C:/out"))
        joined = " ".join(args)
        self.assertIn("--n1-grid-x 41 --n1-grid-y 25", joined)
        self.assertIn("--n1-arena-param-view 0:9940992", joined)
        self.assertIn("--n1-arena-param-view 8:11907072", joined)
        self.assertIn("--n1-weight-param-view 16:4512256", joined)


if __name__ == "__main__":
    unittest.main()
