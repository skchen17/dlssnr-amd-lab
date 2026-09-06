import hashlib
import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts.process_postblock_native_stage_sweep_result import (
    EXPECTED_STAGES,
    process,
)


class PostblockNativeStageSweepReceiverTests(unittest.TestCase):
    def make_archive(self, root: Path, reference: bytes, *, bad_hash=False) -> Path:
        changed = bytearray(reference)
        changed[0:2] = struct.pack("<e", 0.5)
        stages = []
        files = {}
        for order, name in enumerate(EXPECTED_STAGES):
            data = reference if order < 2 else bytes(changed)
            runs = []
            for run_index in (1, 2):
                filename = f"{name}_run{run_index}.raw"
                files[filename] = data
                digest = hashlib.sha256(data).hexdigest()
                if bad_hash and name == "e4m3" and run_index == 1:
                    digest = "0" * 64
                runs.append({
                    "stage": name,
                    "run": run_index,
                    "probe_exit": 0,
                    "probe_pass": True,
                    "execution_verified": True,
                    "device_name": "NVIDIA GeForce RTX 5070",
                    "output_file": filename,
                    "output_bytes": len(data),
                    "output_sha256": digest,
                    "reference_exact": order < 2,
                })
            stages.append({
                "name": name,
                "order": order,
                "valid": True,
                "repeat_bitwise_exact": True,
                "reference_exact": order < 2,
                "runs": runs,
            })
        manifest = {
            "experiment": "rtx_postblock_native_resource_lowering_stage_sweep",
            "status": "PASS",
            "classification": "RTX_POSTBLOCK_FIRST_BREAKING_LOWERING_STAGE",
            "counts_as_s7": False,
            "payload_integrity": True,
            "grid": [81, 49, 1],
            "block": [32, 1, 1],
            "output_reference_sha256": hashlib.sha256(reference).hexdigest(),
            "baseline_reference_exact": True,
            "stages": stages,
        }
        archive = root / "return.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("result/manifest.json", json.dumps(manifest))
            for name, data in files.items():
                zf.writestr(f"result/{name}", data)
        return archive

    def test_accepts_and_finds_first_breaking_stage(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "scripts.process_postblock_native_stage_sweep_result.EXPECTED_OUTPUT_BYTES", 8
        ):
            root = Path(directory)
            reference = struct.pack("<4e", 0.0, 0.0, 0.0, 1.0)
            ref_path = root / "reference.raw"
            ref_path.write_bytes(reference)
            receipt = process(
                self.make_archive(root, reference), root / "output", ref_path)
            self.assertEqual(receipt["first_breaking_stage"], "e4m3")
            comparison = json.loads((root / "output/comparison.json").read_text())
            self.assertTrue(comparison["stages"][0]["reference_exact"])

    def test_rejects_run_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "scripts.process_postblock_native_stage_sweep_result.EXPECTED_OUTPUT_BYTES", 8
        ):
            root = Path(directory)
            reference = struct.pack("<4e", 0.0, 0.0, 0.0, 1.0)
            ref_path = root / "reference.raw"
            ref_path.write_bytes(reference)
            with self.assertRaisesRegex(ValueError, "stage_e4m3"):
                process(self.make_archive(root, reference, bad_hash=True),
                        root / "output", ref_path)


if __name__ == "__main__":
    unittest.main()
