#!/usr/bin/env python3
"""Ingest the RTX full-grid N0 selected-CTA all-MMA trace."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

try:
    from scripts.analyze_slot3_mma_trace import analyze
    from scripts.process_slot3_mma_trace_result import archive_sha256, inspect_archive, sha256
except ModuleNotFoundError:
    from analyze_slot3_mma_trace import analyze
    from process_slot3_mma_trace_result import archive_sha256, inspect_archive, sha256


TRACE_BYTES = 256 * 32 * 40
EXPECTED_EXPERIMENT = "rtx_n0_selected_cta_fp8_mma_trace"
DEFAULT_TARGET_CTA = [1, 0, 0]


def require_manifest(manifest: dict, trace: bytes, target_cta: list[int]) -> None:
    checks = {
        "schema": manifest.get("schema") == 1,
        "experiment": manifest.get("experiment") == EXPECTED_EXPERIMENT,
        "status": manifest.get("status") == "PASS",
        "payload_integrity": manifest.get("payload_integrity") is True,
        "probe": manifest.get("probe_exit") == 0 and manifest.get("probe_pass") is True,
        "baseline": manifest.get("baseline_probe_exit") == 0 and manifest.get("baseline_probe_pass") is True,
        "device": isinstance(manifest.get("device_name"), str) and "NVIDIA" in manifest["device_name"].upper(),
        "launch": manifest.get("grid") == [80, 48, 1] and manifest.get("block") == [32, 1, 1]
        and manifest.get("target_cta") == target_cta and manifest.get("kernel_launched") is True,
        "mma_count": manifest.get("mma_count") == 256,
        "trace_bytes": manifest.get("checkpoint_bytes") == TRACE_BYTES,
        "trace_nonzero": manifest.get("checkpoint_nonzero_bytes") == sum(x != 0 for x in trace),
        "trace_hash": str(manifest.get("checkpoint_sha256", "")).upper() == sha256(trace),
        "extra_scratch": manifest.get("scratch_extra_bytes") == TRACE_BYTES,
        "input_variant": manifest.get("input_variant") == "zero_rgba16f_full_graph",
        "perturbation": isinstance(manifest.get("reference_output_preserved"), bool)
        and isinstance(manifest.get("instrumentation_perturbed"), bool)
        and manifest["reference_output_preserved"] != manifest["instrumentation_perturbed"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("result manifest validation failed: " + ", ".join(failed))


def process_archive(archive: Path, amd_trace: Path, output: Path,
                    target_cta: list[int] = DEFAULT_TARGET_CTA) -> dict:
    archive, amd_trace = archive.resolve(strict=True), amd_trace.resolve(strict=True)
    if amd_trace.stat().st_size != TRACE_BYTES:
        raise ValueError(f"AMD trace must be {TRACE_BYTES} bytes")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    with zipfile.ZipFile(archive) as zf:
        manifest_info, trace_info = inspect_archive(zf)
        manifest = json.loads(zf.read(manifest_info).decode("utf-8-sig")); trace = zf.read(trace_info)
    require_manifest(manifest, trace, target_cta)
    output.mkdir(parents=True)
    rtx_path, amd_path = output / "rtx_mma_trace.raw", output / "amd_mma_trace.raw"
    rtx_path.write_bytes(trace); shutil.copyfile(amd_trace, amd_path)
    (output / "rtx_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    comparison = analyze(rtx_path, amd_path, include_models=True)
    comparison["experiment"] = "n0_cta1_0_internal_fp8_mma_rtx_vs_rx9070xt"
    comparison["target_cta"] = target_cta
    comparison["reference_output_preserved"] = manifest["reference_output_preserved"]
    comparison["instrumentation_perturbed"] = manifest["instrumentation_perturbed"]
    comparison["admissible_as_uninstrumented_oracle"] = manifest["reference_output_preserved"]
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "schema": 1, "experiment": "n0_fp8_mma_selected_cta_trace_result_ingestion", "status": "PASS",
        "source_archive": str(archive), "source_archive_sha256": archive_sha256(archive),
        "rtx_device_name": manifest["device_name"], "target_cta": target_cta,
        "rtx_trace_sha256": sha256(trace), "amd_trace_sha256": comparison["amd_sha256"],
        "reference_output_preserved": manifest["reference_output_preserved"],
        "instrumentation_perturbed": manifest["instrumentation_perturbed"],
        "first_pre_fragment_mismatch_mma": comparison["first_pre_fragment_mismatch_mma"],
        "first_d_mismatch_mma": comparison["first_d_mismatch_mma"],
        "d_half_mismatches": comparison["d_half_mismatches"],
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(); parser.add_argument("archive", type=Path)
    parser.add_argument("--amd-trace", type=Path, default=repo / "results/20260901_190000_n0_cta1_0_fp8_mma_trace/amd/mma_trace.raw")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target-x", type=int, default=1)
    parser.add_argument("--target-y", type=int, default=0)
    args = parser.parse_args()
    output = args.output or repo / "results" / f"{datetime.now():%Y%m%d_%H%M%S}_n0_cta1_0_fp8_mma_cross_vendor"
    try:
        receipt = process_archive(args.archive, args.amd_trace, output, [args.target_x, args.target_y, 0])
    except (FileNotFoundError, FileExistsError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr); return 2
    print(json.dumps(receipt, indent=2)); print(f"Result: {output.resolve()}"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
