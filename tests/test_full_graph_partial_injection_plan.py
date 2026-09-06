import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.make_full_graph_partial_injection_plan import make_plan
from scripts.run_full_graph_integrated import load_diagnostic_injections


class FullGraphPartialInjectionPlanTests(unittest.TestCase):
    def test_slices_reference_and_adjusts_arena_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = bytes(range(32))
            (root / "reference.raw").write_bytes(reference)
            base = root / "base.json"
            output = root / "partial.json"
            base.write_text(json.dumps({
                "checkpoints": {"3": {
                    "arena_offset": 1000,
                    "logical_bytes": 32,
                    "tensor_type": "e4m3",
                    "reference": {"path": "reference.raw"},
                }},
            }), encoding="utf-8")
            report = make_plan(base, output, 3, 8, 12)
            plan = json.loads(output.read_text(encoding="utf-8"))
            injection = plan["diagnostic_injections"][0]
            self.assertEqual(injection["arena_offset"], 1008)
            self.assertEqual(injection["reference_offset"], 8)
            self.assertEqual(report["sha256"], hashlib.sha256(reference[8:20]).hexdigest().upper())
            loaded = load_diagnostic_injections(plan, root)
            self.assertEqual(loaded[3][1], reference[8:20])
            self.assertFalse(plan["counts_as_s7"])


if __name__ == "__main__":
    unittest.main()
