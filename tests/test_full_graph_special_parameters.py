import struct
import unittest

from scripts.run_full_graph_integrated import patch_special_parameters


class FullGraphSpecialParameterTests(unittest.TestCase):
    def test_postblock_preserves_captured_offset56_value(self):
        params = bytearray(184)
        struct.pack_into("<Q", params, 56, 0x0000080200009801)
        patch_special_parameters(params, "post_linear_surface", 123, 456)
        self.assertEqual(struct.unpack_from("<Q", params, 56)[0], 0x0000080200009801)

    def test_copy_and_zero_texture_resources_are_patched(self):
        texture_params = bytearray(b"\xff" * 40)
        patch_special_parameters(texture_params, "zero_rgba16f_texture", 123, 456)
        self.assertEqual(struct.unpack_from("<Q", texture_params, 0)[0], 123)
        self.assertTrue(all(
            struct.unpack_from("<Q", texture_params, offset)[0] == 0
            for offset in (8, 16, 24, 32)
        ))
        copy_params = bytearray(16)
        patch_special_parameters(copy_params, "linear_final_copy", 123, 456)
        self.assertEqual(struct.unpack_from("<Q", copy_params, 8)[0], 456)

    def test_postblock_real_texture_replaces_captured_handle(self):
        params = bytearray(184)
        struct.pack_into("<Q", params, 56, 0x0000080200009801)
        patch_special_parameters(
            params, "post_linear_surface_texture", 123, 456, post_texture=789)
        self.assertEqual(struct.unpack_from("<Q", params, 56)[0], 789)


if __name__ == "__main__":
    unittest.main()
