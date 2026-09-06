import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.make_full_graph_decoder_checkpoint_plan import make_plan


class DecoderCheckpointPlanTests(unittest.TestCase):
    def test_adds_same_archive_settled_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_dir = root / "plans"
            cases = root / "cases"
            plan_dir.mkdir()
            (cases / "slot99").mkdir(parents=True)
            raw = b"abcd"
            (cases / "slot99" / "out.raw").write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest().upper()
            base = plan_dir / "base.json"
            base.write_text(json.dumps({
                "source_archive": {"sha256": "ABC"},
                "slots": [{
                    "slot": 99,
                    "activation_param_views": [{"param_offset": 16, "arena_offset": 64}],
                    "checkpoint": None,
                }],
                "checkpoints": {},
            }), encoding="utf-8")
            (cases / "manifest.json").write_text(json.dumps({
                "source_archive": {"sha256": "ABC"},
                "slots": [{
                    "slot": 99,
                    "outputs": [{
                        "param_offset": 16,
                        "arena_offset": 64,
                        "logical_bytes": 4,
                        "tensor_type": "e4m3",
                        "oracle_source": "slot100_input_before",
                        "asset": {"path": "out.raw", "bytes": 4, "sha256": digest},
                    }],
                }],
            }), encoding="utf-8")
            output = plan_dir / "output.json"
            result = make_plan(base, cases, output, 99, 99)
            written = json.loads(output.read_text())
            self.assertEqual(result["decoder_checkpoints_added"], [99])
            self.assertEqual(written["checkpoints"]["99"]["reference"]["sha256"], digest)
            self.assertEqual(written["slots"][0]["checkpoint"]["arena_offset"], 64)
            self.assertEqual(
                written["variant"]["checkpoint_capture_alignment"],
                "same_full_graph_archive",
            )


if __name__ == "__main__":
    unittest.main()
