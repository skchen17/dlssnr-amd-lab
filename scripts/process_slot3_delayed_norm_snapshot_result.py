#!/usr/bin/env python3
"""Strictly ingest and compare the RTX slot-3 delayed normalization snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zipfile
from pathlib import Path, PurePosixPath

try:
    from scripts.instrument_slot3_delayed_norm_snapshot import BYTES_PER_LANE, LANES, REGISTERS
except ModuleNotFoundError:
    from instrument_slot3_delayed_norm_snapshot import BYTES_PER_LANE, LANES, REGISTERS


EXPECTED_REVISION = "v1_slot3_same_input_delayed_norm_snapshot"
EXPECTED_OUTPUT = "33FE600487C7CF89F8D8F238999D8E0C4602A865E33802CAB09C5D1DC50BD4F9"
TRACE_BYTES = LANES * BYTES_PER_LANE


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def compare(rtx: bytes, amd: bytes) -> dict:
    if len(rtx) != TRACE_BYTES or len(amd) != TRACE_BYTES:
        raise ValueError(f"both traces must be {TRACE_BYTES} bytes")
    per_register = []
    first = None
    total = 0
    for register_index, register in enumerate(REGISTERS):
        examples = []
        mismatches = 0
        for lane in range(LANES):
            offset = lane * BYTES_PER_LANE + register_index * 4
            rtx_word = struct.unpack_from("<I", rtx, offset)[0]
            amd_word = struct.unpack_from("<I", amd, offset)[0]
            if rtx_word != amd_word:
                mismatches += 1; total += 1
                item = {"lane": lane, "rtx_u32": f"0x{rtx_word:08X}",
                        "amd_u32": f"0x{amd_word:08X}"}
                if first is None:
                    first = {"register": register, "register_index": register_index, **item}
                if len(examples) < 8:
                    examples.append(item)
        per_register.append({"register": register, "mismatch_lanes": mismatches,
                             "examples": examples})
    return {
        "schema": 1,
        "experiment": "slot3_delayed_norm_snapshot_rtx5070_vs_rx9070xt",
        "status": "PASS",
        "classification": "OUTPUT_PRESERVING_NORMALIZATION_DIVERGENCE_LOCALIZATION",
        "rtx_sha256": sha256(rtx), "amd_sha256": sha256(amd),
        "bitwise_equal": total == 0, "word_mismatches": total,
        "first_mismatch": first, "per_register": per_register,
    }


def process(archive: Path, output: Path, amd_trace: Path) -> dict:
    archive_bytes = archive.read_bytes()
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        if any(PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
               for name in names):
            raise ValueError("unsafe archive member")

        def one(leaf: str) -> str:
            matches = [name for name in names if PurePosixPath(name).name == leaf]
            if len(matches) != 1:
                raise ValueError(f"expected one {leaf}, found {len(matches)}")
            return matches[0]

        manifest_bytes = zf.read(one("manifest.json"))
        manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
        trace1 = zf.read(one("run1.trace.raw")); trace2 = zf.read(one("run2.trace.raw"))
    runs = manifest.get("runs", [])
    checks = {
        "revision": manifest.get("package_revision") == EXPECTED_REVISION,
        "experiment": manifest.get("experiment") == "rtx5070_slot3_delayed_norm_snapshot",
        "status": manifest.get("status") == "PASS",
        "classification": manifest.get("classification") ==
            "RTX_SLOT3_DELAYED_NORMALIZATION_SNAPSHOT_ORACLE",
        "integrity": manifest.get("payload_integrity") is True,
        "target": manifest.get("target_cta") == [2, 0, 0],
        "layout": manifest.get("registers") == REGISTERS and
            manifest.get("lanes") == LANES and manifest.get("bytes_per_lane") == BYTES_PER_LANE,
        "shape": manifest.get("trace_bytes") == TRACE_BYTES and
            len(trace1) == TRACE_BYTES and len(trace2) == TRACE_BYTES,
        "repeat": manifest.get("trace_repeat_bitwise_exact") is True and trace1 == trace2,
        "reference": str(manifest.get("output_reference_sha256", "")).upper() == EXPECTED_OUTPUT,
        "runs": len(runs) == 2 and all(
            run.get("probe_exit") == 0 and run.get("probe_pass") is True and
            run.get("execution_verified") is True and
            "NVIDIA" in str(run.get("device_name", "")).upper() and
            run.get("trace_bytes") == TRACE_BYTES and run.get("trace_nonzero_bytes", 0) > 0 and
            str(run.get("trace_sha256", "")).upper() == sha256(trace) and
            str(run.get("output_sha256", "")).upper() == EXPECTED_OUTPUT and
            run.get("output_reference_exact") is True
            for run, trace in zip(runs, (trace1, trace2))
        ),
    }
    if not all(checks.values()):
        raise ValueError("result validation failed: " +
                         ", ".join(key for key, value in checks.items() if not value))
    amd = amd_trace.read_bytes()
    comparison = compare(trace1, amd)
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_bytes(manifest_bytes)
    (output / "rtx_trace.raw").write_bytes(trace1)
    (output / "amd_trace.raw").write_bytes(amd)
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "schema": 1, "experiment": "slot3_delayed_norm_snapshot_result_ingestion",
        "status": "PASS", "classification": "SLOT3_DELAYED_NORM_SNAPSHOT_ACCEPTED",
        "counts_as_s7": False, "source_archive": str(archive.resolve()),
        "source_archive_sha256": sha256(archive_bytes), "checks": checks,
        "rtx_trace_sha256": sha256(trace1), "amd_trace_sha256": sha256(amd),
        "word_mismatches": comparison["word_mismatches"],
        "first_mismatch": comparison["first_mismatch"],
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path); parser.add_argument("output", type=Path)
    parser.add_argument("--amd-trace", type=Path,
                        default=Path("results/20260904_900000_slot3_delayed_norm_snapshot/amd/delayed_norm_snapshot.raw"))
    args = parser.parse_args()
    print(json.dumps(process(args.archive, args.output, args.amd_trace), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
