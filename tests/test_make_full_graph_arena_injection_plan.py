import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.make_full_graph_arena_injection_plan import make_plan


class FullArenaInjectionPlanTests(unittest.TestCase):
    def test_builds_same_capture_full_arena_injection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plans = root / "plans"
            cases = root / "cases"
            plans.mkdir()
            (cases / "slot154").mkdir(parents=True)
            arena = b"exact-state"
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
                "activation_arena_initial": {"bytes": len(arena)},
                "slots": [{"slot": 153}, {"slot": 154, "special": "post_linear_surface"}],
            }), encoding="utf-8")

            output = plans / "full-arena.json"
            result = make_plan(base, cases, output, 154, 153)
            plan = json.loads(output.read_text())
            injection = plan["diagnostic_injections"][0]
            self.assertEqual(injection["arena_offset"], 0)
            self.assertEqual(injection["bytes"], len(arena))
            self.assertEqual(injection["reference"]["sha256"], digest)
            self.assertEqual(result["counts_as_s7"], False)
            self.assertEqual(plan["variant"]["full_arena_injection"]["case_slot"], 154)

    def test_can_preserve_raw_special_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plans = root / "plans"
            cases = root / "cases"
            plans.mkdir()
            (cases / "slot154").mkdir(parents=True)
            arena = b"arena"
            digest = hashlib.sha256(arena).hexdigest().upper()
            (cases / "slot154" / "arena.raw").write_bytes(arena)
            (cases / "manifest.json").write_text(json.dumps({
                "source_archive": {"sha256": "ABC"},
                "slots": [{"slot": 154, "activation_arena": {
                    "path": "arena.raw", "bytes": len(arena), "sha256": digest,
                }}],
            }), encoding="utf-8")
            base = plans / "base.json"
            base.write_text(json.dumps({
                "source_archive": {"sha256": "ABC"},
                "activation_arena_initial": {"bytes": len(arena)},
                "slots": [{"slot": 153}, {"slot": 154, "special": "post_linear_surface"}],
            }), encoding="utf-8")

            output = plans / "preserve.json"
            make_plan(base, cases, output, 154, 153, True)
            plan = json.loads(output.read_text())
            self.assertEqual(plan["slots"][1]["special"], "normal")
            self.assertTrue(plan["variant"]["full_arena_injection"]["preserve_case_special_params"])


if __name__ == "__main__":
    unittest.main()
