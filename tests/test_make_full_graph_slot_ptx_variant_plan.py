import tempfile
import unittest
from pathlib import Path

from scripts.make_full_graph_slot_ptx_variant_plan import build


class FullGraphSlotPtxVariantPlanTest(unittest.TestCase):
    def test_replaces_only_explicit_slots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); ptx = root / "candidate.ptx"; ptx.write_text("ptx")
            plan = {"slots": [{"slot": i, "ptx": f"old{i}"} for i in range(156)]}
            output = build(plan, {6: ptx, 7: ptx}, "candidate")
            self.assertEqual([item["slot"] for item in output["variant"]["replacements"]], [6, 7])
            self.assertEqual(output["slots"][5]["ptx"], "old5")
            self.assertEqual(output["slots"][6]["ptx"], str(ptx.resolve()))
            self.assertEqual(plan["slots"][6]["ptx"], "old6")


if __name__ == "__main__":
    unittest.main()
