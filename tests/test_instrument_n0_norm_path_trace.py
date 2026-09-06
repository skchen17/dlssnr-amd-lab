import unittest
from pathlib import Path

from scripts.instrument_n0_norm_path_trace import STAGES, TRACE_BYTES, instrument


class InstrumentN0NormPathTraceTest(unittest.TestCase):
    def test_instruments_all_unique_path_stages(self):
        source = Path(
            "deliverables/n0_fp8_mma_trace_reference_20260901_023855/payload/n0_original.ptx"
        ).read_text(encoding="utf-8")
        output, stages = instrument(source)
        self.assertEqual(len(stages), len(STAGES))
        self.assertEqual(TRACE_BYTES, 1280)
        for index, (name, _, register) in enumerate(STAGES):
            self.assertEqual(stages[index]["name"], name)
            self.assertIn(f"st.global.b32 [%__n0_norm_{index}_address], {register};", output)


if __name__ == "__main__":
    unittest.main()
