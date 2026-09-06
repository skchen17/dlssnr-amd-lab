import unittest

from scripts.instrument_slot3_f16_path_trace import (
    ANCHOR,
    BYTES_PER_LANE,
    REGISTERS,
    instrument,
)


class InstrumentSlot3F16PathTraceTest(unittest.TestCase):
    def test_exports_selected_registers_for_selected_cta(self):
        source = f"""
.visible .entry kernel(.param .align 8 .b8 kernel_param_0[96]) {{
{ANCHOR}
ret;
}}
"""
        output, count = instrument(source, "kernel", 2, 3)
        self.assertEqual(count, 1)
        self.assertEqual(BYTES_PER_LANE, len(REGISTERS) * 4)
        self.assertIn("%__f16p_ctax, 2;", output)
        self.assertIn("%__f16p_ctay, 3;", output)
        self.assertIn("[%__f16p_address+0], %r1751", output)
        self.assertIn(
            f"[%__f16p_address+{(len(REGISTERS) - 1) * 4}], %r2110", output
        )


if __name__ == "__main__":
    unittest.main()
