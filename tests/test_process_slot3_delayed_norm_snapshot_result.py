import hashlib
import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.instrument_slot3_delayed_norm_snapshot import BYTES_PER_LANE, LANES, REGISTERS
from scripts.process_slot3_delayed_norm_snapshot_result import EXPECTED_OUTPUT, process


class ProcessDelayedNormSnapshotTest(unittest.TestCase):
    def test_accepts_strict_repeat_and_compares(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace = bytearray(LANES * BYTES_PER_LANE)
            for lane in range(LANES):
                for index in range(len(REGISTERS)):
                    struct.pack_into("<I", trace, lane * BYTES_PER_LANE + index * 4,
                                     lane * 100 + index + 1)
            trace = bytes(trace); digest = hashlib.sha256(trace).hexdigest().upper()
            runs = [{"probe_exit": 0, "probe_pass": True, "execution_verified": True,
                     "device_name": "NVIDIA GeForce RTX test", "trace_bytes": len(trace),
                     "trace_nonzero_bytes": 1, "trace_sha256": digest,
                     "output_sha256": EXPECTED_OUTPUT, "output_reference_exact": True}
                    for _ in range(2)]
            manifest = {
                "package_revision": "v1_slot3_same_input_delayed_norm_snapshot",
                "experiment": "rtx5070_slot3_delayed_norm_snapshot", "status": "PASS",
                "classification": "RTX_SLOT3_DELAYED_NORMALIZATION_SNAPSHOT_ORACLE",
                "payload_integrity": True, "target_cta": [2, 0, 0],
                "registers": REGISTERS, "lanes": LANES,
                "bytes_per_lane": BYTES_PER_LANE, "trace_bytes": len(trace),
                "output_reference_sha256": EXPECTED_OUTPUT,
                "trace_repeat_bitwise_exact": True, "runs": runs,
            }
            archive = root / "result.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("result/manifest.json", json.dumps(manifest))
                zf.writestr("result/run1.trace.raw", trace)
                zf.writestr("result/run2.trace.raw", trace)
            amd = root / "amd.raw"; amd.write_bytes(trace)
            receipt = process(archive, root / "accepted", amd)
            self.assertEqual(receipt["status"], "PASS")
            self.assertEqual(receipt["word_mismatches"], 0)


if __name__ == "__main__":
    unittest.main()
