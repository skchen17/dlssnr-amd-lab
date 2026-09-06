import unittest

from scripts.lower_ptx_f16x2_local_reduction import TARGETS, lower


class Slot3LocalReductionLoweringTests(unittest.TestCase):
    def test_resolves_four_inputs_and_replaces_only_targets(self):
        lines = []
        expected_sources = {}
        for index, target in enumerate(TARGETS):
            left, right = f"%p{index}a", f"%p{index}b"
            sources = tuple(f"%s{index}_{item}" for item in range(4))
            expected_sources[target] = list(sources)
            lines.extend((
                f"add.f16x2 {left}, {sources[0]}, {sources[1]};",
                f"add.f16x2 {right}, {sources[2]}, {sources[3]};",
                f"add.f16x2 {target}, {left}, {right};",
            ))
        lines.append("add.f16x2 %untouched, %x, %y;")

        output, reports = lower("\n".join(lines))

        self.assertEqual(len(reports), 16)
        self.assertEqual([item["destination"] for item in reports], list(TARGETS))
        self.assertEqual(
            [item["half_inputs"] for item in reports],
            [expected_sources[target] for target in TARGETS],
        )
        self.assertIn("add.f16x2 %untouched, %x, %y;", output)
        self.assertNotIn("add.f16x2 %r1835", output)
        self.assertIn("add.rn.f32 %__slot3_red_0_pair0_lo", output)
        self.assertIn("cvt.rn.f16.f32 %__slot3_red_15_out_hi", output)

    def test_rejects_missing_pair_definition(self):
        with self.assertRaisesRegex(ValueError, "expected unique pair definitions"):
            lower("add.f16x2 %r1835, %missing0, %missing1;")

    def test_can_select_one_target(self):
        lines = []
        for index, target in enumerate(TARGETS):
            lines.extend((
                f"add.f16x2 %p{index}a, %s{index}_0, %s{index}_1;",
                f"add.f16x2 %p{index}b, %s{index}_2, %s{index}_3;",
                f"add.f16x2 {target}, %p{index}a, %p{index}b;",
            ))
        output, reports = lower("\n".join(lines), ("%r1871",))
        self.assertEqual([item["destination"] for item in reports], ["%r1871"])
        self.assertNotIn("add.f16x2 %r1871", output)
        self.assertIn("add.f16x2 %r1835", output)


if __name__ == "__main__":
    unittest.main()
