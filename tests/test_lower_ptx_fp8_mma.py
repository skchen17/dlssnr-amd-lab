import importlib.util
import math
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "lower_ptx_fp8_mma.py"
SPEC = importlib.util.spec_from_file_location("lower_ptx_fp8_mma", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)

REFERENCE_SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_mma_fragments.py"
REFERENCE_SPEC = importlib.util.spec_from_file_location("mma_fragments", REFERENCE_SCRIPT)
REFERENCE = importlib.util.module_from_spec(REFERENCE_SPEC)
assert REFERENCE_SPEC.loader
REFERENCE_SPEC.loader.exec_module(REFERENCE)

ANALYZER_SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_slot3_mma_trace.py"
ANALYZER_SPEC = importlib.util.spec_from_file_location("slot3_mma_trace", ANALYZER_SCRIPT)
ANALYZER = importlib.util.module_from_spec(ANALYZER_SPEC)
assert ANALYZER_SPEC.loader
ANALYZER_SPEC.loader.exec_module(ANALYZER)


class LowerPtxFp8MmaTest(unittest.TestCase):
    def test_constructed_decode_all_codes(self):
        for code in range(256):
            expected = REFERENCE.e4m3(code)
            actual = MODULE.e4m3_constructed(code)
            if math.isnan(expected):
                self.assertTrue(math.isnan(actual))
            else:
                self.assertEqual(actual, expected)

    def test_fragment_sources_cover_requested_coordinates(self):
        for lane in range(32):
            for element in range(4):
                row, column = MODULE.cd_coord(lane, element)
                for k in range(32):
                    a_lane, a_element = MODULE.a_source(lane, element, k)
                    self.assertEqual(REFERENCE.a_coord(a_lane, a_element), (row, k))
                    b_lane, b_element = MODULE.b_source(lane, element, k)
                    self.assertEqual(REFERENCE.b_coord(b_lane, b_element), (k, column))

    def test_rewrite(self):
        statement = (
            "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
            "{%r1, %r2}, {%r3, %r4, %r5, %r6}, {%r7, %r8}, {%r9, %r10};"
        )
        lowered, count = MODULE.lower(statement)
        self.assertEqual(count, 1)
        self.assertNotIn("mma.sync", lowered)
        self.assertIn("fma.rn.f32", lowered)
        self.assertIn("mov.b32 %r1", lowered)

    def test_compact_rewrite_emits_single_helper_and_calls(self):
        statement = (
            "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
            "{%r1, %r2}, {%r3, %r4, %r5, %r6}, {%r7, %r8}, {%r9, %r10};"
        )
        source = ".version 8.7\n.target sm_80\n.address_size 64\n.visible .entry test() {\n" + statement * 2 + "\nret;\n}\n"
        lowered, count = MODULE.lower_compact(source)
        self.assertEqual(count, 2)
        self.assertNotIn("mma.sync", lowered)
        self.assertEqual(lowered.count(".func (.param"), 1)
        self.assertEqual(lowered.count("call.uni"), 2)
        self.assertLess(len(lowered), len(MODULE.replacement(MODULE.FP8_MMA.search(statement), 0)) * 2)

    def test_candidate_models_map_to_executable_options(self):
        names = [model["name"] for model in ANALYZER.candidate_models()]
        for name in names:
            if name == "exact_fsum":
                with self.assertRaisesRegex(ValueError, "CPU diagnostic"):
                    MODULE.candidate_model_options(name)
            else:
                options = MODULE.candidate_model_options(name)
                self.assertEqual(set(options), {
                    "accumulation_mode", "accumulation_order",
                    "accumulation_rounding_mode", "subnormal_mode", "product_mode",
                })

    def test_rz_product_and_pairwise_rewrites(self):
        statement = (
            "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
            "{%r1, %r2}, {%r3, %r4, %r5, %r6}, {%r7, %r8}, {%r9, %r10};"
        )
        rz, count = MODULE.lower(
            statement, accumulation_mode="mantissa11", accumulation_rounding_mode="rz"
        )
        self.assertEqual(count, 1)
        self.assertNotIn("round_lsb}, %__fp8mma_0_r_round_bits", rz)
        product, _ = MODULE.lower(statement, product_mode="f16")
        self.assertIn("cvt.rn.f16.f32 %__fp8mma_0_h_product_round", product)
        pairwise, _ = MODULE.lower(statement, accumulation_order="pairwise_c_first")
        self.assertIn("%__fp8mma_0_r_partial0_32", pairwise)
        self.assertIn("add.rn.f32 %__fp8mma_0_r_partial0_0", pairwise)


if __name__ == "__main__":
    unittest.main()
