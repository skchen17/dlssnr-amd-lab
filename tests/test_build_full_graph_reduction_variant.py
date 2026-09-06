import tempfile
import unittest
from pathlib import Path

from scripts.build_full_graph_reduction_variant import FUNCTION, build


class FullGraphReductionVariantTests(unittest.TestCase):
    def test_changes_only_slots_three_and_four(self):
        with tempfile.TemporaryDirectory() as directory:
            ptx = Path(directory) / "candidate.ptx"; ptx.write_text("ptx")
            plan = {"slots": [
                {"slot": 2, "function": "other", "ptx": "old", "ptx_sha256": "old"},
                {"slot": 3, "function": FUNCTION, "ptx": "old", "ptx_sha256": "old"},
                {"slot": 4, "function": FUNCTION, "ptx": "old", "ptx_sha256": "old"},
                {"slot": 5, "function": "other", "ptx": "old", "ptx_sha256": "old"},
            ]}
            result, changed = build(plan, ptx)
            self.assertEqual(changed, [3, 4])
            self.assertEqual(plan["slots"][1]["ptx"], "old")
            self.assertEqual(result["slots"][1]["ptx"], str(ptx.resolve()))
            self.assertEqual(result["slots"][2]["ptx"], str(ptx.resolve()))
            self.assertEqual(result["slots"][0]["ptx"], "old")

    def test_records_explicit_single_target(self):
        with tempfile.TemporaryDirectory() as directory:
            ptx = Path(directory) / "candidate.ptx"; ptx.write_text("ptx")
            plan = {"slots": [
                {"slot": 3, "function": FUNCTION, "ptx": "old", "ptx_sha256": "old"},
                {"slot": 4, "function": FUNCTION, "ptx": "old", "ptx_sha256": "old"},
            ]}
            result, _ = build(plan, ptx, ["%r1855"], "slot34_single_reduction_search")
            self.assertEqual(result["variant"]["targets"], ["%r1855"])
            self.assertEqual(result["variant"]["kind"], "slot34_single_reduction_search")


if __name__ == "__main__":
    unittest.main()
