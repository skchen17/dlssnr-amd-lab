import struct
import unittest

from scripts.analyze_full_graph_integrated import compare_rgba16f_rgb


def rgba16f(pixels):
    return b"".join(struct.pack("<4e", *pixel) for pixel in pixels)


class FullGraphRgbaTests(unittest.TestCase):
    def test_constant_alpha_cannot_hide_wrong_rgb(self):
        reference = rgba16f([(0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0)])
        candidate = rgba16f([(0.5, 0.5, 0.5, 1.0), (0.5, 0.5, 0.5, 1.0)])
        report = compare_rgba16f_rgb(reference, candidate)
        self.assertEqual(report["normative_channels"], "RGB")
        self.assertEqual(report["alpha_exact_fraction"], 1.0)
        self.assertFalse(report["parity_gate"]["pass"])

    def test_rgb_match_passes_even_if_alpha_is_reported_separately(self):
        reference = rgba16f([(0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0)])
        candidate = rgba16f([(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 0.0)])
        report = compare_rgba16f_rgb(reference, candidate)
        self.assertTrue(report["parity_gate"]["pass"])
        self.assertEqual(report["alpha_exact_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
