#!/usr/bin/env python3
"""Ingest an RTX selected-CTA f16 MMA trace and compare it with RX 9070 XT."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

try:
    from scripts.analyze_n0_f16_mma_cta_trace import TRACE_BYTES, analyze_bytes, sha256
    from scripts.process_slot3_mma_trace_result import archive_sha256, inspect_archive
except ModuleNotFoundError:
    from analyze_n0_f16_mma_cta_trace import TRACE_BYTES, analyze_bytes, sha256
    from process_slot3_mma_trace_result import archive_sha256, inspect_archive


def process_archive(archive: Path, amd_trace: Path, output: Path,
                    target_cta: list[int]) -> dict:
    archive, amd_trace = archive.resolve(strict=True), amd_trace.resolve(strict=True)
    amd = amd_trace.read_bytes()
    if len(amd) != TRACE_BYTES:
        raise ValueError(f"AMD trace must be {TRACE_BYTES} bytes")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    with zipfile.ZipFile(archive) as zf:
        manifest_info, trace_info = inspect_archive(zf, TRACE_BYTES)
        manifest = json.loads(zf.read(manifest_info).decode("utf-8-sig"))
        rtx = zf.read(trace_info)
    checks = {
        "experiment": manifest.get("experiment") == "rtx_n0_selected_cta_f16_mma_trace",
        "status": manifest.get("status") == "PASS",
        "integrity": manifest.get("payload_integrity") is True,
        "device": isinstance(manifest.get("device_name"), str) and "NVIDIA" in manifest["device_name"].upper(),
        "launch": manifest.get("grid") == [80, 48, 1] and manifest.get("block") == [32, 1, 1]
        and manifest.get("target_cta") == target_cta and manifest.get("kernel_launched") is True,
        "mma_count": manifest.get("mma_count") == 16,
        "trace": manifest.get("checkpoint_bytes") == TRACE_BYTES and len(rtx) == TRACE_BYTES
        and str(manifest.get("checkpoint_sha256", "")).upper() == sha256(rtx),
        "scratch": manifest.get("scratch_extra_bytes") == TRACE_BYTES,
        "input": manifest.get("input_variant") == "zero_rgba16f_full_graph",
        "preserved": manifest.get("reference_output_preserved") is True
        and manifest.get("instrumentation_perturbed") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("result manifest validation failed: " + ", ".join(failed))
    output.mkdir(parents=True)
    (output / "rtx_mma_trace.raw").write_bytes(rtx)
    shutil.copyfile(amd_trace, output / "amd_mma_trace.raw")
    (output / "rtx_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    comparison = analyze_bytes(rtx, amd)
    comparison["target_cta"] = target_cta
    comparison["reference_output_preserved"] = True
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "schema": 1, "experiment": "n0_f16_mma_selected_cta_trace_result_ingestion", "status": "PASS",
        "source_archive": str(archive), "source_archive_sha256": archive_sha256(archive),
        "rtx_device_name": manifest["device_name"], "target_cta": target_cta,
        "rtx_trace_sha256": sha256(rtx), "amd_trace_sha256": sha256(amd),
        "reference_output_preserved": True,
        "first_pre_fragment_mismatch_mma": comparison["first_pre_fragment_mismatch_mma"],
        "first_d_mismatch_mma": comparison["first_d_mismatch_mma"],
        "d_half_mismatches": comparison["d_half_mismatches"],
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--amd-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target-x", type=int, default=8)
    parser.add_argument("--target-y", type=int, default=0)
    args = parser.parse_args()
    output = args.output or repo / "results" / f"{datetime.now():%Y%m%d_%H%M%S}_n0_f16_mma_cta_cross_vendor"
    try:
        receipt = process_archive(args.archive, args.amd_trace, output, [args.target_x, args.target_y, 0])
    except (FileNotFoundError, FileExistsError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2)); print(f"Result: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
