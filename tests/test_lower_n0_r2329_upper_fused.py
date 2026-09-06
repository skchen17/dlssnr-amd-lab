import unittest
from pathlib import Path

from scripts.lower_n0_r2329_upper_fused import lower


class LowerN0R2329UpperFusedTest(unittest.TestCase):
    def test_inserts_one_scoped_fusion(self):
        source=Path("deliverables/n0_fp8_mma_trace_reference_20260901_023855/payload/n0_original.ptx").read_text(encoding="utf-8")
        output,count=lower(source)
        self.assertEqual(count,1)
        self.assertIn("max.f16x2 %r2329, %r2329, %__r2329_candidate;",output)

    def test_supports_fused_and_min_ablation_policies(self):
        source="{add.f16x2 %r2329,%r2276,%r2277;}"
        fused,_=lower(source,"fused"); minimum,_=lower(source,"min")
        self.assertIn("mov.b32 %r2329, %__r2329_candidate;",fused)
        self.assertIn("min.f16x2 %r2329, %r2329, %__r2329_candidate;",minimum)


if __name__ == "__main__":
    unittest.main()
