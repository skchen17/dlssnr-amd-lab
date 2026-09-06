import unittest

from scripts.process_n0_normal1_full_grid_traces_result import predicted_half, safe_name


class Normal1GridResultTests(unittest.TestCase):
    def test_reconstructs_rounded_product(self):
        sqrt_bits = 0x40000000  # 2.0
        sin_bits = 0x3F000000  # 0.5
        self.assertEqual(predicted_half(sqrt_bits, sin_bits), 0x3C00)

    def test_rejects_parent_zip_member(self):
        with self.assertRaises(ValueError):
            safe_name("../trace.raw")


if __name__ == "__main__":
    unittest.main()
