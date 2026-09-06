import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "extract_graph_slot_params.py"
SPEC = importlib.util.spec_from_file_location("extract_graph_slot_params", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class ExtractGraphSlotParamsTest(unittest.TestCase):
    def test_extracts_exact_parameter_bytes(self):
        fields = ["slot", "function_name", "module_dll_offset", "module_fnv1a64",
                  "grid_x", "grid_y", "grid_z", "block_x", "block_y", "block_z",
                  "dynamic_shared", "param_size", "parameters_stable_across_5_frames",
                  "frame1_param_hex"]
        row = dict.fromkeys(fields, "0")
        row.update({"slot": "2", "function_name": "kernel", "module_dll_offset": "0x10",
                    "module_fnv1a64": "ABC", "grid_x": "4", "grid_y": "3",
                    "grid_z": "1", "block_x": "32", "block_y": "1", "block_z": "1",
                    "param_size": "4", "parameters_stable_across_5_frames": "True",
                    "frame1_param_hex": "01020304"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(row)
            raw, report = MODULE.extract(path, 2, "kernel")
        self.assertEqual(raw, b"\x01\x02\x03\x04")
        self.assertEqual(report["grid"], [4, 3, 1])
        self.assertTrue(report["parameters_stable_across_5_frames"])


if __name__ == "__main__":
    unittest.main()
