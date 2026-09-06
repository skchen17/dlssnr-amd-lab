import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "lower_ptx_zluda_compat.py"
SPEC = importlib.util.spec_from_file_location("ptx_lower", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class LowerPtxCompatTest(unittest.TestCase):
    def test_discard_and_wide_store(self):
        source = """
mov.b64 {%r1, _}, %rd1;
mov.b64 {_, %r2}, %rd2;
{ .reg .b128 v; mov.b128 v, {%r3,%r4,%r5,%r6}; st.global.L1::no_allocate.b128 [%rd3], v; }
"""
        lowered, report = MODULE.lower(source)
        self.assertEqual(report["discard_mov_b64_lowered"], 2)
        self.assertEqual(report["wide_mov_store_b128_lowered"], 1)
        self.assertNotIn("mov.b128", lowered)
        self.assertNotIn("no_allocate", lowered)
        self.assertIn("st.global.b32 [%rd3+12], %r6", lowered)

    def test_entry_specific_counts_can_be_zero(self):
        lowered, report = MODULE.lower("mov.u32 %r1, %r2;")
        self.assertEqual(lowered, "mov.u32 %r1, %r2;")
        self.assertEqual(report["discard_mov_b64_lowered"], 0)
        self.assertEqual(report["wide_mov_store_b128_lowered"], 0)

    def test_release_store_keeps_ordering_and_drops_only_cache_hint(self):
        source = "st.release.gpu.global.L1::no_allocate.s32 [%rd1], %r2;"
        lowered, report = MODULE.lower(source)
        self.assertEqual(lowered, "st.release.gpu.global.s32 [%rd1], %r2;")
        self.assertEqual(report["release_cache_hint_stores_lowered"], 1)

    def test_relaxed_load_keeps_ordering_and_drops_only_cache_hint(self):
        source = "ld.relaxed.gpu.global.L1::no_allocate.s32 %r1, [%rd2];"
        lowered, report = MODULE.lower(source)
        self.assertEqual(lowered, "ld.relaxed.gpu.global.s32 %r1, [%rd2];")
        self.assertEqual(report["relaxed_cache_hint_loads_lowered"], 1)

    def test_shared_cta_scope_is_redundant(self):
        source = ("ld.shared::cta.v4.u32 {%r1,%r2,%r3,%r4}, [%r5];\n"
                  "st.shared::cta.v4.u32 [%r5], {%r1,%r2,%r3,%r4};")
        lowered, report = MODULE.lower(source)
        self.assertNotIn("shared::cta", lowered)
        self.assertIn("ld.shared.v4.u32", lowered)
        self.assertIn("st.shared.v4.u32", lowered)
        self.assertEqual(report["shared_cta_scopes_lowered"], 2)


if __name__ == "__main__":
    unittest.main()
