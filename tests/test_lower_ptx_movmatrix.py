import importlib.util
import random
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "lower_ptx_movmatrix.py"
SPEC = importlib.util.spec_from_file_location("lower_ptx_movmatrix", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class LowerPtxMovmatrixTest(unittest.TestCase):
    def test_transpose_recovers_matrix_columns(self):
        words = []
        for lane in range(32):
            row, column_pair = lane >> 2, lane & 3
            low = row * 8 + column_pair * 2
            high = low + 1
            words.append(low | (high << 16))
        output = MODULE.transpose_words(words)
        for lane, packed in enumerate(output):
            column, row_pair = lane >> 2, lane & 3
            self.assertEqual(packed & 0xFFFF, (row_pair * 2) * 8 + column)
            self.assertEqual(packed >> 16, (row_pair * 2 + 1) * 8 + column)

    def test_randomized_reference_mapping(self):
        random.seed(7)
        words = [random.getrandbits(32) for _ in range(32)]
        self.assertEqual(len(MODULE.transpose_words(words)), 32)

    def test_rewrite(self):
        lowered, count = MODULE.lower(
            "movmatrix.sync.trans.aligned.m8n8.b16 %r2, %r1;"
        )
        self.assertEqual(count, 1)
        self.assertNotIn("movmatrix.sync", lowered)
        self.assertIn("shfl.sync.idx.b32", lowered)
        self.assertIn("or.b32 %r2", lowered)


if __name__ == "__main__":
    unittest.main()
