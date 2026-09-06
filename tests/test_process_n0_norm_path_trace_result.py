import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.analyze_n0_norm_path_trace import sha256
from scripts.instrument_n0_norm_path_trace import STAGES, TRACE_BYTES
from scripts.process_n0_norm_path_trace_result import process_archive


class ProcessN0NormPathTraceResultTest(unittest.TestCase):
    def test_accepts_valid_controlled_result(self):
        trace = bytes([1]) * TRACE_BYTES
        manifest = {
            "schema": 1, "experiment": "rtx_n0_normalization_path_trace", "status": "PASS",
            "payload_integrity": True, "probe_exit": 0, "probe_pass": True,
            "baseline_probe_exit": 0, "baseline_probe_pass": True,
            "device_name": "NVIDIA GeForce RTX 5070", "grid": [1, 1, 1], "block": [32, 1, 1],
            "kernel_launched": True, "stage_count": len(STAGES),
            "stages": [{"index": i, "name": name, "register": reg} for i, (name, _, reg) in enumerate(STAGES)],
            "checkpoint_bytes": TRACE_BYTES, "checkpoint_nonzero_bytes": TRACE_BYTES,
            "checkpoint_sha256": sha256(trace), "reference_output_preserved": True,
            "instrumentation_perturbed": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); archive = root / "result.zip"; amd = root / "amd.raw"
            amd.write_bytes(trace)
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("result/manifest.json", json.dumps(manifest))
                zf.writestr("result/norm_path_trace.raw", trace)
            receipt = process_archive(archive, amd, root / "out")
        self.assertEqual(receipt["status"], "PASS")
        self.assertIsNone(receipt["first_divergent_stage"])


if __name__ == "__main__":
    unittest.main()
