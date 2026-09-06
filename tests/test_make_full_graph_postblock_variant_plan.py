import json
import tempfile
import unittest
from pathlib import Path

from scripts.make_full_graph_postblock_variant_plan import main


class FullGraphPostblockVariantPlanTests(unittest.TestCase):
    def test_module_imports(self):
        self.assertTrue(callable(main))


if __name__ == "__main__":
    unittest.main()
