import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "consolidate_full_graph_capture.ps1"
POWERSHELL = shutil.which("powershell")


@unittest.skipUnless(POWERSHELL, "Windows PowerShell is required")
class ConsolidateFullGraphCaptureTest(unittest.TestCase):
    def make_result(self, root: Path, copy_status: str = "PASS") -> Path:
        result = root / "result"
        capture_dir = result / "full_graph"
        capture_dir.mkdir(parents=True)
        (result / "summary.json").write_text(
            json.dumps({"status": "PASS"}), encoding="utf-8"
        )
        (result / "copy_snapshot.json").write_text(
            json.dumps(
                {
                    "status": copy_status,
                    "input_bytes": 4,
                    "output_bytes": 4,
                    "input_fnv1a64": "0x1234",
                    "output_fnv1a64": "0x1234",
                }
            ),
            encoding="utf-8",
        )
        windows = []
        for slot in range(155):
            before = f"slot_{slot:03d}_before.raw"
            after = f"slot_{slot:03d}_after.raw"
            (capture_dir / before).write_bytes(bytes([slot & 0xFF]))
            (capture_dir / after).write_bytes(bytes([(slot + 1) & 0xFF]))
            windows.append(
                {
                    "slot": slot,
                    "capture_bytes": 1,
                    "before_file": f"full_graph/{before}",
                    "after_file": f"full_graph/{after}",
                }
            )
        (result / "full_graph_capture.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "capture_limit_bytes": 8388608,
                    "window_count": len(windows),
                    "windows": windows,
                }
            ),
            encoding="utf-8",
        )
        return result

    def run_consolidator(self, result: Path):
        environment = os.environ.copy()
        # The desktop test runtime prepends PowerShell 7-only modules to
        # PSModulePath. Windows PowerShell 5.1 must select its inbox Utility
        # module so Get-FileHash is available, matching the cloud package host.
        environment["PSModulePath"] = os.pathsep.join(
            item
            for item in environment.get("PSModulePath", "").split(os.pathsep)
            if "codex-primary-runtime" not in item.lower()
        )
        completed = subprocess.run(
            [
                POWERSHELL,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(SCRIPT),
                "-ResultDir",
                str(result),
            ],
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        summary = json.loads(
            (result / "full_workflow_summary.json").read_text(encoding="utf-8-sig")
        )
        return completed, summary

    def test_strict_hybrid_coverage_passes_all_156_slots(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.make_result(Path(temporary))
            completed, summary = self.run_consolidator(result)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(summary["captured_buffer_slot_count"], 155)
            self.assertEqual(summary["covered_slot_count"], 156)
            self.assertEqual(summary["missing_slots_0_155"], [])
            self.assertEqual(summary["final_copy_status"], "PASS")

    def test_final_copy_is_required_for_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.make_result(Path(temporary), copy_status="FAIL")
            completed, summary = self.run_consolidator(result)
            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(summary["status"], "PARTIAL")
            self.assertEqual(summary["covered_slot_count"], 155)
            self.assertEqual(summary["missing_slots_0_155"], [155])


if __name__ == "__main__":
    unittest.main()
