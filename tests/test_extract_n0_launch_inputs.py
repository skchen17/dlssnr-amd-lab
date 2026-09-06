import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "extract_n0_launch_inputs.py"
SPEC = importlib.util.spec_from_file_location("extract_n0_launch", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class ExtractN0LaunchInputsTest(unittest.TestCase):
    def test_extract(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.jsonl"
            events = [
                {"ev": "nvapi_create_cu_function", "name": MODULE.ENTRY, "function": "0x1"},
                {"ev": "nvapi_launch_cu_kernel", "function": "0x1", "param_size": 264,
                 "param_hex": "00" * 264, "grid": [80, 48, 1], "block": [32, 1, 1]},
            ]
            trace.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
            raw, report = MODULE.extract(trace)
            self.assertEqual(len(raw), 264)
            self.assertEqual(report["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
