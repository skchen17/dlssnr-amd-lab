import unittest
from pathlib import Path


class RunAmdSwin2hScriptTest(unittest.TestCase):
    def test_captured_geometry_and_release_counts_are_encoded(self):
        text = (Path(__file__).parents[1] / "scripts" / "run_amd_swin2h_slots6_8.ps1").read_text()
        for fragment in ("grid=@(20,12,1); releases=240", "grid=@(21,13,1); releases=273",
                         "grid=@(21,12,1); releases=252", "--n1-block-y','2"):
            self.assertIn(fragment, text)


if __name__ == "__main__":
    unittest.main()
