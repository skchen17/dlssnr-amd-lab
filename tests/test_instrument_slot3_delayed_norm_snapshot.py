import unittest

from scripts.instrument_slot3_delayed_norm_snapshot import (
    ANCHOR,
    BYTES_PER_LANE,
    REGISTERS,
    instrument,
)


class DelayedNormSnapshotTest(unittest.TestCase):
    def test_places_all_stores_only_at_delayed_anchor(self):
        function = "kernel"
        text = (
            ".visible .entry kernel(\n.param .u64 kernel_param_0\n)\n{\n"
            "add.f16x2 %r1922, %r1875, %r1876;\n"
            + ANCHOR + "\n}\n"
        )
        output, count = instrument(text, function, 2, 0)
        self.assertEqual(count, 1)
        self.assertEqual(len(REGISTERS), 31)
        self.assertEqual(BYTES_PER_LANE, 124)
        self.assertEqual(output.count("st.global.b32"), len(REGISTERS))
        self.assertIn("[%__dn_address+120], %r2004", output)
        self.assertGreater(output.index("st.global.b32"), output.index(ANCHOR))
        self.assertNotIn("st.global.b32", output[:output.index(ANCHOR)])

    def test_rejects_missing_anchor(self):
        text = ".visible .entry kernel(.param .u64 kernel_param_0)\n{\nret;\n}\n"
        with self.assertRaisesRegex(ValueError, "anchor"):
            instrument(text, "kernel")


if __name__ == "__main__":
    unittest.main()
