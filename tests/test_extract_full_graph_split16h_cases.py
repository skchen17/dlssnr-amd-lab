import unittest

from scripts.extract_full_graph_split16h_cases import abi_for


class Split16hAbiTests(unittest.TestCase):
    def test_all_slots_have_nonoverlapping_required_pointer_fields(self):
        for slot in range(24, 57):
            abi = abi_for(slot)
            offsets = [abi[name] for name in ("input", "output", "weights")]
            offsets += [abi[name] for name in ("input2", "wait", "release", "extra")
                        if name in abi and abi[name] is not None]
            self.assertEqual(len(offsets), len(set(offsets)), slot)
            self.assertTrue(all(offset % 8 == 0 for offset in offsets), slot)

    def test_special_and_repeating_abis(self):
        self.assertEqual(abi_for(24)["ptx_tag"], "ffwd_inpview")
        self.assertEqual(abi_for(25)["input2"], 8)
        self.assertEqual(abi_for(26)["wait"], 40)
        self.assertEqual(abi_for(27)["output"], 16)
        self.assertEqual(abi_for(28)["wait"], 32)
        self.assertEqual(abi_for(29)["ptx_tag"], "ffwd_proj")
        self.assertEqual(abi_for(55)["extra"], 24)
        self.assertIsNone(abi_for(56)["release"])


if __name__ == "__main__":
    unittest.main()
