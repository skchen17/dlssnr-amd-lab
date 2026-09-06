import unittest
from pathlib import Path


class Swin2hReferencePackageTest(unittest.TestCase):
    def test_runner_preserves_graph_geometry_and_slot9_abi(self):
        text = (Path(__file__).parents[1] / "scripts" / "run_nvidia_swin2h_slots6_9_reference.ps1").read_text()
        for fragment in (
            "grid=@(20,12,1);releases=240",
            "grid=@(21,13,1);releases=273",
            "grid=@(21,12,1);releases=252",
            "grid=@(20,13,1);releases=0",
            "--n1-block-y','2",
            "--n1-no-release",
            "--n1-extra-output",
        ):
            self.assertIn(fragment, text)


if __name__ == "__main__":
    unittest.main()
