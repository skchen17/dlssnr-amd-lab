import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_n0_lowering_delta.py"
SPEC = importlib.util.spec_from_file_location("n0_lowering_delta", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class N0LoweringDeltaTest(unittest.TestCase):
    @staticmethod
    def lines(statement, count):
        return "\n".join([f'Unrecognized statement "{statement}"'] * count)

    def test_expected_delta(self):
        numerical = "\n".join(
            [
                self.lines("cvt.rn.satfinite.e4m3x2.f16x2 x;", 424),
                self.lines("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 x;", 256),
                self.lines("movmatrix.sync.trans.aligned.m8n8.b16 x;", 32),
                self.lines("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 x;", 16),
            ]
        )
        mechanics = "\n".join(
            [
                self.lines("mov.b64 {%x, _}, %rd1;", 4),
                self.lines("mov.b128 {%x, %y}, %v;", 4),
                "\n".join(
                    ["Not yet implemented: st instruction with cache policy/eviction priority/cache hints"]
                    * 4
                ),
            ]
        )
        probe = {
            "device_name": "AMD Radeon RX 9070 XT [ZLUDA]",
            "module_loaded": True,
            "function_resolved": False,
            "kernel_launched": False,
        }
        report = MODULE.analyze(numerical + "\n" + mechanics, numerical, probe)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["removed_parser_diagnostics"], 12)
        self.assertEqual(report["lowered"]["parser_diagnostics"], 728)


if __name__ == "__main__":
    unittest.main()
