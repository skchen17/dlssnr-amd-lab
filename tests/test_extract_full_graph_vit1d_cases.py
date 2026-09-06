import unittest

from scripts.extract_full_graph_vit1d_cases import output_specs, tag_for


class Vit1dCaseSpecificationTests(unittest.TestCase):
    def test_all_nine_entries_have_output_specs(self):
        functions = [
            "cc_vit_1d_attention_chained_fp8",
            "cc_vit_1d_ffn_contract_chained_fp8",
            "cc_vit_1d_ffn_expand_chained_fp8",
            "cc_vit_1d_ffn_expand_publish_fp8",
            "cc_vit_1d_projection_chained_fp8",
            "cc_vit_1d_projection_wait_fp8",
            "cc_vit_1d_qkv_chained_fp8",
            "cc_vit_1d_repack_1d_to_2d_fp8",
            "cc_vit_1d_repack_2d_to_1d_fp8",
        ]
        for function in functions:
            tag = tag_for(function)
            specs = output_specs(57, tag)
            self.assertTrue(specs)
            self.assertTrue(all(size > 0 for _, size, _ in specs))

    def test_qkv_has_three_fp8_edges_and_one_fp16_accumulator(self):
        specs = output_specs(60, "qkv")
        self.assertEqual([kind for _, _, kind in specs],
                         ["e4m3", "e4m3", "e4m3", "fp16"])

    def test_ffn_expand_stores_fp8_not_fp16(self):
        for tag in ('ffn_expand', 'ffn_expand_publish'):
            self.assertEqual(output_specs(58, tag), [(16, 96*4096, 'e4m3')])


if __name__ == "__main__":
    unittest.main()
