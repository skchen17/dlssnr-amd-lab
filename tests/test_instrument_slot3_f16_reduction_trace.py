import unittest
from scripts.instrument_slot3_f16_reduction_trace import ANCHOR, BYTES_PER_LANE, REGISTERS, instrument


class InstrumentSlot3F16ReductionTraceTest(unittest.TestCase):
    def test_exports_reduction_registers(self):
        source = f".visible .entry kernel(.param .align 8 .b8 kernel_param_0[96]) {{\n{ANCHOR}\nret;\n}}"
        output, count = instrument(source, "kernel", 2, 3)
        self.assertEqual(count, 1); self.assertEqual(BYTES_PER_LANE, len(REGISTERS) * 4)
        self.assertIn("%__f16r_ctax, 2;", output)
        self.assertIn("[%__f16r_address+0], %r1751", output)
        self.assertIn(f"[%__f16r_address+{BYTES_PER_LANE - 4}], %r1924", output)


if __name__ == "__main__": unittest.main()
