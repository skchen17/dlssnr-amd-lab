import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from scripts.make_full_graph_early_swin_checkpoint_plan import make_plan


class MakeFullGraphEarlySwinCheckpointPlanTests(unittest.TestCase):
    def test_uses_adjacent_consumer_state_from_the_same_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_root = root / "plan"
            plan_root.mkdir()
            (plan_root / "references").mkdir()
            slots = []
            windows = []
            blobs = {}
            specs = {3: ("e4m3", 8), 6: ("fp16", 12)}
            for slot in range(156):
                slots.append({
                    "slot": slot,
                    "ptx": f"slot{slot}.ptx",
                    "activation_param_views": [{"param_offset": 8, "arena_offset": 1000 + slot}],
                    "checkpoint": None,
                })
            for slot, (_, logical) in specs.items():
                data = bytes([slot + 1]) * (logical + 4)
                name = f"blobs/slot{slot + 1}_before.raw"
                blobs[name] = data
                windows.append({
                    "slot": slot + 1,
                    "param_offset": 0,
                    "resource_offset": 1000 + slot,
                    "capture_bytes": len(data),
                    "before_blob": name,
                    "before_sha256": hashlib.sha256(data).hexdigest().upper(),
                })
            archive = root / "capture.zip"
            with ZipFile(archive, "w") as zf:
                zf.writestr("full_graph_capture.json", json.dumps({"windows": windows}))
                for name, data in blobs.items():
                    zf.writestr(name, data)
            base = plan_root / "base.json"
            output = plan_root / "diagnostic.json"
            base.write_text(json.dumps({"slots": slots, "checkpoints": {}}), encoding="utf-8")
            report = make_plan(base, archive, output, specs)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["diagnostic_checkpoints_added"], [3, 6])
            self.assertEqual(report["checkpoint_capture_alignment"], "same_full_graph_archive")
            self.assertEqual(result["slots"][3]["ptx"], "slot3.ptx")
            self.assertEqual(result["checkpoints"]["3"]["tensor_type"], "e4m3")
            self.assertEqual(result["checkpoints"]["6"]["tensor_type"], "fp16")
            self.assertEqual(result["checkpoints"]["6"]["reference"]["bytes"], 12)
            self.assertEqual(
                (plan_root / "references" / "slot6_same_capture.raw").read_bytes(),
                bytes([7]) * 12,
            )


if __name__ == "__main__":
    unittest.main()
