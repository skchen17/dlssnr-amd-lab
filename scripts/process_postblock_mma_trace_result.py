#!/usr/bin/env python3
"""Strictly ingest an RTX post-block MMA trace result and compare with RX."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

try:
    from scripts.analyze_postblock_mma_trace import analyze_bytes
    from scripts.instrument_postblock_mma_trace import TRACE_BYTES
except ModuleNotFoundError:
    from analyze_postblock_mma_trace import analyze_bytes
    from instrument_postblock_mma_trace import TRACE_BYTES


EXPECTED_REVISION = "v1_postblock_exact_frame1_selected_cta_mma_trace"
EXPECTED_OUTPUT = "8ADB4DE9E238DDA7A1155BD817E22236963A067C52C58A182AF50A09BAE6E8C6"
EXPECTED_TARGET = [70, 26, 0]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


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
        trace1 = zf.read(one("run1.trace.raw"))
        trace2 = zf.read(one("run2.trace.raw"))

    runs = manifest.get("runs", [])
    checks = {
        "package_revision": manifest.get("package_revision") == EXPECTED_REVISION,
        "experiment": manifest.get("experiment") ==
            "rtx_postblock_exact_frame1_selected_cta_mma_trace",
        "status": manifest.get("status") == "PASS",
        "classification": manifest.get("classification") ==
            "RTX_POSTBLOCK_FP8_F16_MMA_ORACLE",
        "payload_integrity": manifest.get("payload_integrity") is True,
        "target_cta": manifest.get("target_cta") == EXPECTED_TARGET,
        "trace_shape": len(trace1) == TRACE_BYTES and len(trace2) == TRACE_BYTES and
            manifest.get("trace_bytes") == TRACE_BYTES,
        "trace_repeat": trace1 == trace2 and
            manifest.get("trace_repeat_bitwise_exact") is True,
        "run_count": len(runs) == 2,
        "runs_valid": len(runs) == 2 and all(
            run.get("probe_exit") == 0 and run.get("probe_pass") is True and
            run.get("execution_verified") is True and
            "NVIDIA" in str(run.get("device_name", "")).upper() and
            run.get("surface_initial_loaded") is True and
            run.get("trace_bytes") == TRACE_BYTES and
            run.get("trace_nonzero_bytes", 0) > 0 and
            run.get("trace_sha256", "").upper() == sha256(trace) and
            run.get("output_sha256", "").upper() == EXPECTED_OUTPUT and
            run.get("output_reference_exact") is True
            for run, trace in zip(runs, (trace1, trace2))),
        "reference_hash": manifest.get("output_reference_sha256", "").upper() ==
            EXPECTED_OUTPUT,
    }
    if not all(checks.values()):
        raise ValueError("result validation failed: " +
                         ", ".join(key for key, value in checks.items() if not value))
    amd = amd_trace.read_bytes()
    comparison = analyze_bytes(trace1, amd)
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_bytes(manifest_bytes)
    (output / "rtx_trace.raw").write_bytes(trace1)
    (output / "amd_trace.raw").write_bytes(amd)
    (output / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "schema": 1,
        "experiment": "postblock_exact_mma_trace_result_ingestion",
        "status": "PASS",
        "classification": "POSTBLOCK_FP8_F16_MMA_ORACLE_ACCEPTED",
        "counts_as_s7": False,
        "source_archive": str(archive.resolve()),
        "source_archive_sha256": sha256(archive_bytes),
        "checks": checks,
        "rtx_trace_sha256": sha256(trace1),
        "amd_trace_sha256": sha256(amd),
        "comparison_bitwise_exact": comparison["bitwise_exact"],
        "first_mismatch": comparison["first_mismatch"],
        "first_output_mismatch_with_local_record_inputs_exact":
            comparison["first_output_mismatch_with_local_record_inputs_exact"],
        "first_output_mismatch_with_warp_inputs_exact":
            comparison["first_output_mismatch_with_warp_inputs_exact"],
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--amd-trace", type=Path,
        default=Path("results/20260904_750000_postblock_exact_mma_trace/amd_trace.raw"))
    args = parser.parse_args()
    receipt = process(args.archive, args.output, args.amd_trace)
    print(json.dumps(receipt, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
