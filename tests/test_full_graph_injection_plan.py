import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.make_full_graph_injection_plan import make_plan
from scripts.run_full_graph_integrated import load_diagnostic_injections


class FullGraphInjectionPlanTests(unittest.TestCase):
    def test_injection_is_explicitly_non_s7_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = b"RTX-state"
            (root / "state.raw").write_bytes(state)
            digest = hashlib.sha256(state).hexdigest().upper()
            base = root / "base.json"
            output = root / "injected.json"
            base.write_text(json.dumps({"checkpoints": {"9": {
                "arena_offset": 123, "logical_bytes": len(state), "tensor_type": "e4m3",
                "reference": {"path": "state.raw", "bytes": len(state), "sha256": digest},
            }}}), encoding="utf-8")
            report = make_plan(base, output, [9])
            plan = json.loads(output.read_text(encoding="utf-8"))
            loaded = load_diagnostic_injections(plan, root)
            self.assertFalse(report["counts_as_s7"])
            self.assertEqual(loaded[9][1], state)
            self.assertIn("never count as S7", plan["injection_warning"])


if __name__ == "__main__":
    unittest.main()
