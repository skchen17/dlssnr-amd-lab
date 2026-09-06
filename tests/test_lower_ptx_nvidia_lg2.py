import unittest
from scripts.lower_ptx_nvidia_lg2 import lower

class LowerPtxNvidiaLg2Tests(unittest.TestCase):
    def test_inserts_table_and_rewrites_two_n0_sites(self):
        source='.version 8.7\n.visible .entry k(){\nlg2.approx.ftz.f32 %r218, %r181;\nlg2.approx.ftz.f32 %r222, %r205;\n}'
        output,count=lower(source,{'coefficients_u32':[[1,2,3]for _ in range(64)]})
        self.assertEqual(count,2);self.assertIn('.const .align 4 .u32 __n0_nvidia_lg2_coeffs[192]',output);self.assertIn('ld.const.b32 %__n0_lg2_0_c0b',output);self.assertNotIn('lg2.approx',output)

if __name__=='__main__':unittest.main()
