import unittest
from pathlib import Path


class ProbeSurfaceTextureInitializationTests(unittest.TestCase):
    def test_zero_texture_upload_does_not_reuse_surface_initial_pixels(self):
        source = Path("tools/zluda_ptx_probe/zluda_ptx_probe.cpp").read_text(
            encoding="utf-8")
        create = source.index('"cuArrayCreate_n1_zero_texture"')
        reset = source.index("copy.srcHost = zero.data();", create)
        upload = source.index('"cuMemcpy2D_n1_zero_texture"', create)
        self.assertLess(create, reset)
        self.assertLess(reset, upload)


if __name__ == "__main__":
    unittest.main()
