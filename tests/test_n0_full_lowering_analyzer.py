import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_n0_full_lowering.py"
SPEC = importlib.util.spec_from_file_location("n0_full_lowering", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class N0FullLoweringAnalyzerTest(unittest.TestCase):
    def test_diagnostic_counter_includes_both_classes(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parser.log"
            path.write_text(
                'Unrecognized statement "a;"\nNot yet implemented: sample\n',
                encoding="utf-8",
            )
            self.assertEqual(MODULE.diagnostics(path), 2)


if __name__ == "__main__":
    unittest.main()
