import unittest
from scripts.lower_n0_lg2_ulp_bias import lower

class LowerN0Lg2UlpBiasTests(unittest.TestCase):
    def test_rewrites_both_box_muller_sites(self):
        source='lg2.approx.ftz.f32 %r218, %r181;\nlg2.approx.ftz.f32 %r222, %r205;'
        output,count=lower(source,2)
        self.assertEqual(count,2);self.assertIn('sub.u32 %r218, %r218, 2;',output);self.assertIn('sub.u32 %r222, %r222, 2;',output)

if __name__=='__main__':unittest.main()
