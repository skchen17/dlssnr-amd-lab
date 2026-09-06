import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_output_head_qkv_contract.py"
SPEC = importlib.util.spec_from_file_location("output_head_qkv_contract", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class OutputHeadQkvContractTest(unittest.TestCase):
    def test_half_arithmetic_rounds_each_operation(self):
        one = MODULE.half_bits(1.0)
        half = MODULE.half_bits(0.5)
        self.assertEqual(MODULE.half_mul(one, half), half)
        self.assertEqual(MODULE.half_add(half, half), one)

    def test_balanced_reduction(self):
        values = [MODULE.half_bits(1.0)] * 32
        self.assertEqual(MODULE.half_value(MODULE.reduce_half(values, "balanced")), 32.0)

    def test_packed_k_source_channel(self):
        self.assertEqual(
            [MODULE.packed_k_source_channel(k) for k in range(16)],
            [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15],
        )

    def test_encode_e4m3(self):
        self.assertEqual(MODULE.encode_e4m3(MODULE.half_bits(1.0)), 0x38)
        self.assertEqual(MODULE.encode_e4m3(MODULE.half_bits(-1.0)), 0xB8)


if __name__ == "__main__":
    unittest.main()
