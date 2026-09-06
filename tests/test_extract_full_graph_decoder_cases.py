import unittest

from scripts.extract_full_graph_decoder_cases import output_spec


class DecoderCaseSpecificationTests(unittest.TestCase):
    def test_stage_output_shapes(self):
        self.assertEqual(output_spec("dec_input"), (16, 122880, "e4m3"))
        self.assertEqual(output_spec("split_qkv"), (8, 122880, "e4m3"))
        self.assertEqual(output_spec("swin8_chained"), (8, 245760, "e4m3"))
        self.assertEqual(output_spec("swin1_outview"), (8, 1966080, "e4m3"))
        self.assertEqual(output_spec("post_block"), (16, 1843200, "rgba16f"))


if __name__ == "__main__":
    unittest.main()
