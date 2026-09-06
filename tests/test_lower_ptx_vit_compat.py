import unittest

from scripts.lower_ptx_vit_compat import lower


class VitCompatibilityLoweringTests(unittest.TestCase):
    def test_vector_reduction_and_release_fence_are_lowered(self):
        source = """red.global.v4.f16x2.add.noftz [%rd3], {%r1, %r2, %r3, %r4};
fence.release.gpu;
"""
        result, counts = lower(source)
        self.assertEqual(counts["vector_f16x2_reductions_lowered"], 1)
        self.assertEqual(counts["release_gpu_fences_lowered"], 1)
        self.assertIn("add.f16x2 %__dlssnr_red_0_3_new", result)
        self.assertIn("atom.global.cas.b32 %__dlssnr_red_0_3_old", result)
        self.assertIn("membar.gl;", result)
        self.assertNotIn("red.global.v4", result)
        self.assertNotIn("fence.release.gpu", result)


if __name__ == "__main__":
    unittest.main()
