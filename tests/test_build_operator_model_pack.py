import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_operator_model_pack import STAGES, build_pack, classify


class OperatorModelPackTests(unittest.TestCase):
    def test_operator_classification(self):
        self.assertEqual("qkv", classify("cc_vit_1d_qkv_chained_fp8"))
        self.assertEqual("attention", classify("cc_vit_1d_attention_chained_fp8"))
        self.assertEqual("post_block", classify("cc_tinlayout_fused_post_block_swin_1h_32_fp8"))
        self.assertEqual("final_copy", classify("cg2r_copy_kernel"))
        self.assertEqual("swin_output", classify("cc_tinlayout_fused_swin_8h_256_8_outview_wait_fp8"))

    def test_builds_private_semantic_pack(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = bytes(range(64))
            (root / "model.raw").write_bytes(weights)
            slots = []
            for slot in range(156):
                function = "cc_vit_1d_qkv_chained_fp8" if slot not in (0, 154, 155) else {
                    0: "cc_cb_clear",
                    154: "cc_tinlayout_fused_post_block_swin_1h_32_fp8",
                    155: "cg2r_copy_kernel",
                }[slot]
                slots.append(
                    {
                        "slot": slot,
                        "function": function,
                        "activation_param_views": [],
                        "weight_param_views": ([{"param_offset": 0, "weight_offset": 0}]
                                               if slot == 1 else []),
                        "special": "normal",
                    }
                )
            plan = {
                "activation_arena_bytes": 4096,
                "model_arena": {
                    "path": "model.raw",
                    "bytes": len(weights),
                    "sha256": hashlib.sha256(weights).hexdigest().upper(),
                },
                "slots": slots,
            }
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            output = root / "pack"
            manifest = build_pack(plan_path, output, copy_weights=True)
            self.assertEqual("SEMANTIC_GRAPH_READY", manifest["status"])
            self.assertEqual(len(STAGES), len(manifest["stages"]))
            self.assertEqual(156, manifest["source"]["captured_slot_count"])
            self.assertEqual(weights, (output / "weights.raw").read_bytes())

            override = root / "decoded.raw"
            override.write_bytes(weights)
            override_output = root / "override-pack"
            override_manifest = build_pack(
                plan_path, override_output, copy_weights=False, model_arena_override=override
            )
            self.assertEqual("decoded.raw", override_manifest["model"]["source_filename"])

            decoded_dir = root / "decoded"
            decoded_dir.mkdir()
            decoded_arena = decoded_dir / "model_arena.raw"
            decoded_arena.write_bytes(weights)
            digest = hashlib.sha256(weights).hexdigest().upper()
            (decoded_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "experiment": "local_dlssnr_weight_resource_decode",
                        "model_arena": {"bytes": len(weights), "sha256": digest},
                        "tensors": [
                            {
                                "name": "block0.layer0.layer",
                                "arena_offset": 0,
                                "data_bytes": len(weights),
                                "element_count": len(weights) // 2,
                                "dtype": "fp16",
                                "sha256": digest,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            decoded_output = root / "decoded-pack"
            decoded_manifest = build_pack(
                plan_path, decoded_output, copy_weights=False, model_arena_override=decoded_arena
            )
            self.assertEqual(1, decoded_manifest["model"]["tensor_count"])
            self.assertEqual("block0.layer0.layer", decoded_manifest["weight_spans"][0]["name"])
            self.assertEqual("resource_record_exact",
                             decoded_manifest["weight_spans"][0]["boundary_confidence"])


if __name__ == "__main__":
    unittest.main()
