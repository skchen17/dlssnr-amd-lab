import json
import tempfile
import unittest
from pathlib import Path

from scripts.make_full_graph_n0_variant_plan import make_plan


class MakeFullGraphN0VariantPlanTest(unittest.TestCase):
    def test_replaces_only_slot_one(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/'base.json';ptx=root/'n0.ptx';out=root/'out.json';ptx.write_text('ptx',encoding='utf-8')
            slots=[{"slot":i,"function":"cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8" if i==1 else f"f{i}","ptx":"old","ptx_sha256":"OLD"} for i in range(156)]
            base.write_text(json.dumps({"slots":slots}),encoding='utf-8');make_plan(base,ptx,out,'min')
            plan=json.loads(out.read_text());self.assertEqual(plan['variant']['changed_slots'],[1]);self.assertEqual(plan['slots'][0]['ptx'],'old');self.assertEqual(plan['slots'][2]['ptx'],'old');self.assertEqual(plan['slots'][1]['ptx'],str(ptx.resolve()))


if __name__ == '__main__':
    unittest.main()
