import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.analyze_n0_mma178_b_path_trace import analyze, sha256
from scripts.instrument_n0_mma178_b_path_trace import LANES, STAGES, TRACE_BYTES, instrument
from scripts.process_n0_mma178_b_path_trace_result import process_archive


class N0Mma178BPathTraceTest(unittest.TestCase):
    def test_instruments_all_unique_stages(self):
        source = Path(
            "deliverables/n0_fp8_mma_trace_reference_20260901_023855/payload/n0_original.ptx"
        ).read_text(encoding="utf-8")
        output, stages = instrument(source)
        self.assertEqual(len(stages), 9)
        self.assertEqual(TRACE_BYTES, 1152)
        for index, (name, _, register) in enumerate(STAGES):
            self.assertEqual(stages[index]["name"], name)
            self.assertIn(f"st.global.b32 [%__n0_mma178_b_{index}_address], {register};", output)

    def test_analyzer_localizes_scale_result(self):
        rtx = bytearray(TRACE_BYTES)
        amd = bytearray(rtx)
        amd[(6 * LANES + 27) * 4 + 2] = 1
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rtx.raw").write_bytes(rtx)
            (root / "amd.raw").write_bytes(amd)
            report = analyze(root / "rtx.raw", root / "amd.raw")
        self.assertEqual(report["first_divergent_stage"], 6)
        self.assertEqual(report["first_divergent_stage_name"], "scale_mma146_r2842")
        self.assertEqual(report["stages"][6]["mismatches"][0]["lane"], 27)

    def test_receiver_accepts_valid_zero_input_result(self):
        trace = bytes([1]) * TRACE_BYTES
        manifest = {
            "schema": 1, "experiment": "rtx_n0_mma178_b_path_trace", "status": "PASS",
            "payload_integrity": True, "probe_exit": 0, "probe_pass": True,
            "baseline_probe_exit": 0, "baseline_probe_pass": True,
            "device_name": "NVIDIA GeForce RTX 5070", "grid": [1, 1, 1], "block": [32, 1, 1],
            "kernel_launched": True, "stage_count": len(STAGES),
            "stages": [{"index": i, "name": name, "register": reg} for i, (name, _, reg) in enumerate(STAGES)],
            "checkpoint_bytes": TRACE_BYTES, "checkpoint_nonzero_bytes": TRACE_BYTES,
            "checkpoint_sha256": sha256(trace), "reference_output_preserved": True,
            "instrumentation_perturbed": False, "input_variant": "zero_rgba16f_full_graph",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); archive = root / "result.zip"; amd = root / "amd.raw"
            amd.write_bytes(trace)
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("result/manifest.json", json.dumps(manifest))
                zf.writestr("result/mma178_b_path_trace.raw", trace)
            receipt = process_archive(archive, amd, root / "out")
        self.assertEqual(receipt["status"], "PASS")
        self.assertIsNone(receipt["first_divergent_stage"])


if __name__ == "__main__":
    unittest.main()
