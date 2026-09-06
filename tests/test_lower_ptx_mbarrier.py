import unittest

from scripts.lower_ptx_mbarrier import lower


class MbarrierLoweringTests(unittest.TestCase):
    def test_immediate_pair_is_lowered_to_stronger_cta_barrier(self):
        source = """mbarrier.init.shared.b64 [%r1], %r2;
mov.b32 %r9, -1;
elect.sync _|P_ELECT, %r9;
cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%r6], [%rd7], %r8, [%r1];
mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%r1], %r8;
mbarrier.arrive.shared::cta.b64 %rd3, [%r4], %r5;
{
.reg .pred P_OUT;
mbarrier.try_wait.shared::cta.b64 P_OUT, [%r4], %rd3;
}
"""
        result, counts = lower(source)
        self.assertEqual(counts["mbarrier_init_lowered"], 1)
        self.assertEqual(counts["full_warp_elect_lowered"], 1)
        self.assertEqual(counts["async_bulk_lowered"], 1)
        self.assertEqual(counts["mbarrier_expect_tx_lowered"], 1)
        self.assertEqual(counts["mbarrier_arrive_lowered"], 1)
        self.assertEqual(counts["mbarrier_try_wait_lowered"], 1)
        self.assertEqual(counts["remaining_mbarrier_instructions"], 0)
        self.assertIn("bar.sync 0;", result)
        self.assertIn("setp.eq.u32 P_ELECT, %laneid, 0;", result)
        self.assertIn("ld.global.v4.b32", result)
        self.assertIn("st.shared.v4.b32", result)
        self.assertIn("mov.b64 %rd3, 0;", result)
        self.assertIn("mov.pred P_OUT, 1;", result)


if __name__ == "__main__":
    unittest.main()
