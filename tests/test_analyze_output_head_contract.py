import importlib.util
import struct
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_output_head_contract.py"
SPEC = importlib.util.spec_from_file_location("output_head_contract", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class OutputHeadContractTest(unittest.TestCase):
    def test_decode_params(self):
        raw = bytearray(184)
        struct.pack_into("<2i", raw, 32, 384, 640)
        struct.pack_into("<2i", raw, 40, -4, -4)
        struct.pack_into("<f", raw, 48, 0.03125)
        struct.pack_into("<i", raw, 52, 1)
        struct.pack_into("<2f", raw, 72, 640.0, 360.0)
        struct.pack_into("<2f", raw, 80, 1.0 / 640.0, 1.0 / 360.0)
        struct.pack_into("<2i", raw, 172, 640, 360)
        decoded = MODULE.decode_params(bytes(raw))
        self.assertEqual(decoded["padded_feature_extent"], [384, 640])
        self.assertEqual(decoded["feature_origin"], [-4, -4])
        self.assertEqual(decoded["output_scale"], 0.03125)
        self.assertEqual(decoded["output_extent"], [640, 360])

    def test_decode_params_rejects_wrong_size(self):
        with self.assertRaisesRegex(ValueError, "184"):
            MODULE.decode_params(b"\0" * 183)

    def test_slot_and_tensor_lookup_require_unique_entry(self):
        self.assertEqual(MODULE.find_slot({"slots": [{"slot": 154}]}, 154)["slot"], 154)
        self.assertEqual(
            MODULE.find_tensor({"tensors": [{"name": "x"}]}, "x")["name"], "x"
        )
        with self.assertRaises(ValueError):
            MODULE.find_slot({"slots": []}, 154)
        with self.assertRaises(ValueError):
            MODULE.find_tensor({"tensors": [{"name": "x"}, {"name": "x"}]}, "x")


if __name__ == "__main__":
    unittest.main()
