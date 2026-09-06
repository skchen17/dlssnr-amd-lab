import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_output_head_mma_dependencies.py"
SPEC = importlib.util.spec_from_file_location("output_head_mma_dependencies", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class OutputHeadMmaDependenciesTest(unittest.TestCase):
    def test_fragment_extractors_preserve_lane_major_order(self):
        trace = bytearray(2 * MODULE.LANES * MODULE.MMA_RECORD_BYTES)
        for mma in range(2):
            for lane in range(MODULE.LANES):
                base = (mma * MODULE.LANES + lane) * MODULE.MMA_RECORD_BYTES
                trace[base:base + MODULE.MMA_RECORD_BYTES] = bytes(
                    (mma * 41 + lane * 3 + byte) & 255
                    for byte in range(MODULE.MMA_RECORD_BYTES)
                )
        actual = MODULE._mma_fragment(bytes(trace), 1, 16, 8)
        expected = b"".join(
            trace[(MODULE.LANES + lane) * MODULE.MMA_RECORD_BYTES + 16:
                  (MODULE.LANES + lane) * MODULE.MMA_RECORD_BYTES + 24]
            for lane in range(MODULE.LANES)
        )
        self.assertEqual(actual, expected)

    def test_e4_fragment_groups_eight_operations_per_lane(self):
        trace = bytearray(16 * MODULE.LANES * MODULE.E4_RECORD_BYTES)
        for operation in range(16):
            for lane in range(MODULE.LANES):
                base = (operation * MODULE.LANES + lane) * MODULE.E4_RECORD_BYTES
                trace[base + 4:base + 6] = bytes((operation, lane))
        fragment = MODULE._e4_a_fragment(bytes(trace), 4)
        self.assertEqual(fragment[:16], b"".join(bytes((op, 0)) for op in range(4, 12)))
        self.assertEqual(fragment[16:32], b"".join(bytes((op, 1)) for op in range(4, 12)))

    def test_model_b_fragment(self):
        head = bytes(range(256)) * 4
        fragment = MODULE._model_b_fragment(head, 0, 1)
        self.assertEqual(fragment[:8], bytes(range(8, 16)))
        self.assertEqual(fragment[8:16], bytes(range(24, 32)))

    def test_half_mul_bits(self):
        one = 0x3C00
        half = 0x3800
        self.assertEqual(MODULE._half_mul_bits(one, half), half)


if __name__ == "__main__":
    unittest.main()
