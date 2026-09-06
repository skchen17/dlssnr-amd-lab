import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.make_full_graph_n0_scratch_checkpoint_plan import make_plan


class N0ScratchCheckpointPlanTests(unittest.TestCase):
    def test_replaces_slot1_checkpoint_with_settled_scratch(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "scripts.make_full_graph_n0_scratch_checkpoint_plan.SCRATCH_OFFSET", 4
        ), patch("scripts.make_full_graph_n0_scratch_checkpoint_plan.SCRATCH_BYTES", 4):
            root = Path(directory)
            plans = root / "plans"
            cases = root / "cases"
            plans.mkdir()
            (cases / "slot154").mkdir(parents=True)
            arena = b"xxxxDATAyyyy"
            digest = hashlib.sha256(arena).hexdigest().upper()
            (cases / "slot154" / "arena.raw").write_bytes(arena)
            (cases / "manifest.json").write_text(json.dumps({
                "source_archive": {"sha256": "ABC"},
                "slots": [{
                    "slot": 154,
                    "activation_arena": {
                        "path": "arena.raw", "bytes": len(arena), "sha256": digest,
                    },
                }],
            }), encoding="utf-8")
            base = plans / "base.json"
            base.write_text(json.dumps({
                "source_archive": {"sha256": "ABC"},
                "checkpoints": {"1": {}},
                "slots": [
                    {"slot": 1, "activation_param_views": [{"param_offset": 216, "arena_offset": 4}]},
                    {"slot": 154, "activation_param_views": [{"param_offset": 8, "arena_offset": 4}]},
                ],
            }), encoding="utf-8")
            output = plans / "output.json"
            result = make_plan(base, cases, output)
            plan = json.loads(output.read_text())
            self.assertEqual(result["sha256"], hashlib.sha256(b"DATA").hexdigest().upper())
            self.assertEqual(plan["checkpoints"]["1"]["logical_bytes"], 4)
            self.assertEqual(plan["variant"]["n0_scratch_checkpoint_slot"], 1)


if __name__ == "__main__":
    unittest.main()
