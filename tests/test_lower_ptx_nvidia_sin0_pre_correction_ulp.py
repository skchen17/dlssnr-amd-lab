import unittest

from scripts.lower_ptx_nvidia_sin0_pre_correction_ulp import lower


class NvidiaSin0PreCorrectionUlpTests(unittest.TestCase):
    def test_shift_is_inserted_immediately_after_sine(self):
        model = {"segments": 2, "scale_u32": 0x3E22F983, "ulp_shifts_i32": [0, 1]}
        source = ".visible .entry x() {\nsin.approx.ftz.f32 %r228, %r226;\nadd.rn.f32 %r228, %r228, %r1;\n}\n"
        output, count = lower(source, model)
        self.assertEqual(count, 1)
        self.assertLess(output.index("add.s32 %__n0_sin_pre_ulp_0_bits"),
                        output.index("add.rn.f32 %r228, %r228, %r1"))


if __name__ == "__main__":
    unittest.main()
