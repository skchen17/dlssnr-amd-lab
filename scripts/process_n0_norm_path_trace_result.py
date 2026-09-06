#!/usr/bin/env python3
"""Safely ingest an RTX N0 normalization-path trace and compare it with RX."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath

try:
    from scripts.analyze_n0_norm_path_trace import analyze, sha256
    from scripts.instrument_n0_norm_path_trace import STAGES, TRACE_BYTES
    from scripts.process_slot3_mma_trace_result import archive_sha256, normalized_member
except ModuleNotFoundError:
    from analyze_n0_norm_path_trace import analyze, sha256
    from instrument_n0_norm_path_trace import STAGES, TRACE_BYTES
    from process_slot3_mma_trace_result import archive_sha256, normalized_member


EXPECTED_EXPERIMENT = "rtx_n0_normalization_path_trace"
MAX_MEMBERS = 32
MAX_TOTAL_UNCOMPRESSED = 256 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024


def inspect_archive(zf: zipfile.ZipFile) -> tuple[zipfile.ZipInfo, zipfile.ZipInfo]:
    infos = zf.infolist()
    if not infos or len(infos) > MAX_MEMBERS:
        raise ValueError(f"unexpected ZIP member count: {len(infos)}")
    if sum(info.file_size for info in infos) > MAX_TOTAL_UNCOMPRESSED:
        raise ValueError("ZIP uncompressed payload exceeds safety limit")
    seen: set[str] = set()
    files: list[tuple[PurePosixPath, zipfile.ZipInfo]] = []
    for info in infos:
        path = normalized_member(info.filename)
        key = path.as_posix().casefold()
        if key in seen:
            raise ValueError(f"duplicate ZIP member rejected: {info.filename!r}")
        seen.add(key)
        if not info.is_dir():
            files.append((path, info))
    manifests = [(path, info) for path, info in files if path.name == "manifest.json"]
    traces = [(path, info) for path, info in files if path.name == "norm_path_trace.raw"]
    if len(manifests) != 1 or len(traces) != 1:
        raise ValueError("archive must contain exactly one manifest.json and norm_path_trace.raw")
    manifest_path, manifest_info = manifests[0]
    trace_path, trace_info = traces[0]
    if manifest_path.parent != trace_path.parent or len(manifest_path.parts) != 2:
        raise ValueError("manifest and trace must share one top-level result directory")
    if manifest_info.file_size > MAX_MANIFEST_BYTES:
        raise ValueError("manifest exceeds safety limit")
    if trace_info.file_size != TRACE_BYTES:
        raise ValueError(f"norm_path_trace.raw must be {TRACE_BYTES} bytes")
    return manifest_info, trace_info


def require_manifest(manifest: dict, trace: bytes) -> None:
    expected_stages = [
        {"index": index, "name": name, "register": register}
        for index, (name, _, register) in enumerate(STAGES)
    ]
    checks = {
        "schema": manifest.get("schema") == 1,
        "experiment": manifest.get("experiment") == EXPECTED_EXPERIMENT,
        "status": manifest.get("status") == "PASS",
        "payload_integrity": manifest.get("payload_integrity") is True,
        "probe": manifest.get("probe_exit") == 0 and manifest.get("probe_pass") is True,
        "baseline_probe": manifest.get("baseline_probe_exit") == 0
        and manifest.get("baseline_probe_pass") is True,
        "device": isinstance(manifest.get("device_name"), str)
        and "NVIDIA" in manifest["device_name"].upper(),
        "launch": manifest.get("grid") == [1, 1, 1]
        and manifest.get("block") == [32, 1, 1]
        and manifest.get("kernel_launched") is True,
        "stages": manifest.get("stage_count") == len(STAGES)
        and manifest.get("stages") == expected_stages,
        "trace_bytes": manifest.get("checkpoint_bytes") == TRACE_BYTES,
        "trace_nonzero": manifest.get("checkpoint_nonzero_bytes")
        == sum(value != 0 for value in trace),
        "trace_hash": str(manifest.get("checkpoint_sha256", "")).upper() == sha256(trace),
        "perturbation_fields": isinstance(manifest.get("reference_output_preserved"), bool)
        and isinstance(manifest.get("instrumentation_perturbed"), bool)
        and manifest["reference_output_preserved"] != manifest["instrumentation_perturbed"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("result manifest validation failed: " + ", ".join(failed))


def process_archive(archive: Path, amd_trace: Path, output: Path) -> dict:
    archive = archive.resolve(strict=True)
    amd_trace = amd_trace.resolve(strict=True)
    if amd_trace.stat().st_size != TRACE_BYTES:
        raise ValueError(f"AMD trace must be {TRACE_BYTES} bytes")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    with zipfile.ZipFile(archive) as zf:
        manifest_info, trace_info = inspect_archive(zf)
        manifest = json.loads(zf.read(manifest_info).decode("utf-8-sig"))
        trace = zf.read(trace_info)
    require_manifest(manifest, trace)
    output.mkdir(parents=True)
    rtx_path = output / "rtx_norm_path_trace.raw"
    amd_path = output / "amd_norm_path_trace.raw"
    rtx_path.write_bytes(trace)
    shutil.copyfile(amd_trace, amd_path)
    (output / "rtx_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    comparison = analyze(rtx_path, amd_path)
    comparison["reference_output_preserved"] = manifest["reference_output_preserved"]
    comparison["instrumentation_perturbed"] = manifest["instrumentation_perturbed"]
    comparison["admissible_as_uninstrumented_oracle"] = manifest["reference_output_preserved"]
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "schema": 1,
        "experiment": "n0_normalization_path_trace_result_ingestion",
        "status": "PASS",
        "source_archive": str(archive),
        "source_archive_sha256": archive_sha256(archive),
        "rtx_device_name": manifest["device_name"],
        "rtx_trace_sha256": sha256(trace),
        "amd_trace_sha256": comparison["amd_sha256"],
        "reference_output_preserved": manifest["reference_output_preserved"],
        "instrumentation_perturbed": manifest["instrumentation_perturbed"],
        "first_divergent_stage": comparison["first_divergent_stage"],
        "first_divergent_stage_name": comparison["first_divergent_stage_name"],
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--amd-trace", type=Path, default=repo / "results/20260901_123000_n0_norm_path_trace/amd/norm_path_trace.raw")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or repo / "results" / f"{datetime.now():%Y%m%d_%H%M%S}_n0_norm_path_cross_vendor"
    try:
        receipt = process_archive(args.archive, args.amd_trace, output)
    except (FileNotFoundError, FileExistsError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2))
    print(f"Result: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
