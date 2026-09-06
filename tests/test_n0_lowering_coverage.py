import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_n0_lowering_coverage.py"
SPEC = importlib.util.spec_from_file_location("n0_coverage", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class N0CoverageTest(unittest.TestCase):
    def test_classification(self):
        samples = {
            "cvt.rn.satfinite.e4m3x2.f16x2 %rs1, %r1;": "e4m3_conversion",
            "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 x;": "fp8_mma",
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 x;": "f16_mma",
            "movmatrix.sync.trans.aligned.m8n8.b16 x;": "movmatrix",
            "mov.b64 {%x, _}, %rd1;": "tuple_mov_b64",
            "mov.b128 {%x, %y}, %v;": "tuple_mov_b128",
        }
        for statement, expected in samples.items():
            self.assertEqual(MODULE.classify(statement), expected)

    def test_cache_hint_is_a_separate_parser_diagnostic(self):
        statement_lines = "\n".join(
            f'Unrecognized statement "{statement}"'
            for statement in (
                ["cvt.rn.satfinite.e4m3x2.f16x2 x;"] * 424
                + ["mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 x;"] * 256
                + ["movmatrix.sync.trans.aligned.m8n8.b16 x;"] * 32
                + ["mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 x;"] * 16
                + ["mov.b64 {%x, _}, %rd1;"] * 4
                + ["mov.b128 {%x, %y}, %v;"] * 4
            )
        )
        cache_lines = "\n".join(
            ["Not yet implemented: st instruction with cache policy/eviction priority/cache hints"] * 4
        )
        report = MODULE.analyze(statement_lines + "\n" + cache_lines)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["parser_diagnostics"], 740)
        self.assertEqual(report["semantic_oracle_covered_statements"], 728)
        self.assertEqual(report["mechanical_lowering_statements"], 12)
        self.assertEqual(report["unresolved_statements"], 12)


if __name__ == "__main__":
    unittest.main()
