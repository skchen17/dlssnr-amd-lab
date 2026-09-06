import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_ptx_global_access import analyze


class PtxGlobalAccessAnalyzerTest(unittest.TestCase):
    def test_direct_parameter_symbol_is_a_pointer_origin(self):
        source = """
ld.param.b64 %rd1, [kernel_param_0+8];
cvta.to.global.u64 %rd2, %rd1;
ld.global.u32 %r1, [%rd2+12];
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.ptx"
            path.write_text(source, encoding="utf-8")
            result = analyze(path, {8})
        self.assertEqual(len(result["accesses"]), 1)
        self.assertEqual(result["accesses"][0]["param_offset"], 8)
        self.assertEqual(result["accesses"][0]["constant_offset"], 12)

    def test_pointer_origins_survive_dynamic_addressing(self):
        source = """
ld.param.b64 %rd1, [%rd9+216];
add.s64 %rd2, %rd1, %rd7;
add.s64 %rd3, %rd2, 64;
ld.global.v4.u32 {%r1,%r2,%r3,%r4}, [%rd3+16];
ld.param.b64 %rd4, [%rd9+248];
cvta.to.global.u64 %rd5, %rd4;
st.global.v2.b16 [%rd5+32], {%rs1,%rs2};
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.ptx"
            path.write_text(source, encoding="utf-8")
            result = analyze(path, {216, 248})
        accesses = result["accesses"]
        self.assertEqual(len(accesses), 2)
        self.assertEqual(accesses[0]["param_offset"], 216)
        self.assertEqual(accesses[0]["constant_offset"], 80)
        self.assertTrue(accesses[0]["has_dynamic_offset"])
        self.assertEqual(accesses[0]["width_bytes"], 16)
        self.assertEqual(accesses[1]["operation"], "write")
        self.assertEqual(accesses[1]["constant_offset"], 32)
        self.assertEqual(accesses[1]["width_bytes"], 4)


if __name__ == "__main__":
    unittest.main()
