import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.make_full_graph_swin4h_checkpoint_plan import make_plan


class MakeFullGraphSwin4hCheckpointPlanTests(unittest.TestCase):
    def test_adds_fp16_checkpoints_without_replacing_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_root = root / "plan"
            cases_root = root / "cases"
            plan_root.mkdir()
            cases_root.mkdir()
            slots = []
            cases = []
            for slot in range(156):
                views = [{"param_offset": 8, "arena_offset": 1000 + slot}]
                slots.append({"slot": slot, "ptx": f"slot{slot}.ptx",
                              "activation_param_views": views, "checkpoint": None})
                if 10 <= slot <= 14:
                    case_dir = cases_root / f"slot{slot}"
                    case_dir.mkdir()
                    data = bytes([slot]) * 16
                    (case_dir / "output_reference.raw").write_bytes(data)
                    cases.append({
                        "slot": slot,
                        "main_logical_bytes": 8,
                        "main_tensor_type": "fp16",
                        "assets": {"output_reference": {
                            "path": "output_reference.raw", "bytes": len(data),
                            "sha256": hashlib.sha256(data).hexdigest().upper(),
                        }},
                    })
            base = plan_root / "base.json"
            output = plan_root / "diagnostic.json"
            base.write_text(json.dumps({"slots": slots, "checkpoints": {}}), encoding="utf-8")
            (cases_root / "manifest.json").write_text(json.dumps({"slots": cases}), encoding="utf-8")
            report = make_plan(base, cases_root, output)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["diagnostic_checkpoints_added"], [10, 11, 12, 13, 14])
            self.assertEqual(result["slots"][10]["ptx"], "slot10.ptx")
            self.assertEqual(result["checkpoints"]["10"]["arena_offset"], 1010)
            self.assertEqual(result["checkpoints"]["10"]["tensor_type"], "fp16")
            self.assertEqual(result["checkpoints"]["10"]["reference"]["bytes"], 8)


if __name__ == "__main__":
    unittest.main()
