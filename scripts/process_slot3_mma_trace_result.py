#!/usr/bin/env python3
"""Validate and analyze a returned RTX slot-3 MMA trace archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath

try:
    from scripts.analyze_slot3_mma_trace import analyze
except ModuleNotFoundError:
    from analyze_slot3_mma_trace import analyze


TRACE_BYTES = 256 * 32 * 40
EXPECTED_EXPERIMENT = "rtx5070_slot3_all_fp8_mma_register_trace"
MAX_MEMBERS = 32
MAX_TOTAL_UNCOMPRESSED = 256 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def archive_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def normalized_member(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        raise ValueError(f"absolute ZIP member path rejected: {name!r}")
    path = PurePosixPath(normalized)
    if not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"unsafe ZIP member path rejected: {name!r}")
    return path


def inspect_archive(zf: zipfile.ZipFile,
                    expected_trace_bytes: int = TRACE_BYTES) -> tuple[zipfile.ZipInfo, zipfile.ZipInfo]:
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
    traces = [(path, info) for path, info in files if path.name == "mma_trace.raw"]
    if len(manifests) != 1 or len(traces) != 1:
        raise ValueError("archive must contain exactly one manifest.json and mma_trace.raw")
    manifest_path, manifest_info = manifests[0]
    trace_path, trace_info = traces[0]
    if manifest_path.parent != trace_path.parent or len(manifest_path.parts) != 2:
        raise ValueError("manifest and trace must share one top-level result directory")
    if manifest_info.file_size > MAX_MANIFEST_BYTES:
        raise ValueError("manifest exceeds safety limit")
    if trace_info.file_size != expected_trace_bytes:
        raise ValueError(f"mma_trace.raw must be {expected_trace_bytes} bytes")
    return manifest_info, trace_info


def require_manifest(manifest: dict, trace: bytes) -> None:
    checks = {
        "schema": manifest.get("schema") == 1,
        "experiment": manifest.get("experiment") == EXPECTED_EXPERIMENT,
        "status": manifest.get("status") == "PASS",
        "payload_integrity": manifest.get("payload_integrity") is True,
        "probe_exit": manifest.get("probe_exit") == 0,
        "probe_pass": manifest.get("probe_pass") is True,
        "device_name": isinstance(manifest.get("device_name"), str)
        and "NVIDIA" in manifest["device_name"].upper(),
        "kernel_launched": manifest.get("kernel_launched") is True,
        "checkpoint_bytes": manifest.get("checkpoint_bytes") == TRACE_BYTES,
        "checkpoint_nonzero_bytes": manifest.get("checkpoint_nonzero_bytes")
        == sum(value != 0 for value in trace),
        "checkpoint_sha256": str(manifest.get("checkpoint_sha256", "")).upper()
        == sha256(trace),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("result manifest validation failed: " + ", ".join(failed))


def process_archive(archive: Path, amd_trace: Path, output: Path,
                    *, include_models: bool = True) -> dict:
    archive = archive.resolve(strict=True)
    amd_trace = amd_trace.resolve(strict=True)
    if amd_trace.stat().st_size != TRACE_BYTES:
        raise ValueError(f"AMD trace must be {TRACE_BYTES} bytes")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    with zipfile.ZipFile(archive, "r") as zf:
        manifest_info, trace_info = inspect_archive(zf)
        manifest = json.loads(zf.read(manifest_info).decode("utf-8-sig"))
        trace = zf.read(trace_info)
    require_manifest(manifest, trace)

    output.mkdir(parents=True)
    rtx_trace = output / "rtx_mma_trace.raw"
    rtx_trace.write_bytes(trace)
    (output / "rtx_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copyfile(amd_trace, output / "amd_mma_trace.raw")
    comparison = analyze(rtx_trace, output / "amd_mma_trace.raw",
                         include_models=include_models)
    (output / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    receipt = {
        "schema": 1,
        "experiment": "slot3_mma_trace_result_ingestion",
        "status": "PASS" if comparison["input_fragment_gate"] else "FAIL",
        "classification": "VALIDATED_RTX_RX_INTERNAL_MMA_COMPARISON",
        "source_archive": str(archive),
        "source_archive_sha256": archive_sha256(archive),
        "rtx_device_name": manifest["device_name"],
        "cta": manifest.get("cta"),
        "rtx_trace_sha256": sha256(trace),
        "amd_trace_sha256": comparison["amd_sha256"],
        "input_fragment_gate": comparison["input_fragment_gate"],
        "mma_result_bitwise_gate": comparison["mma_result_bitwise_gate"],
        "first_pre_fragment_mismatch_mma": comparison["first_pre_fragment_mismatch_mma"],
        "first_d_mismatch_mma": comparison["first_d_mismatch_mma"],
        "d_half_mismatches": comparison["d_half_mismatches"],
        "exact_candidate_models": comparison["exact_candidate_models"],
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Safely ingest and compare an RTX slot-3 MMA trace result ZIP."
    )
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--amd-trace", type=Path,
        default=repo / "results/20260901_003000_slot3_mma_trace/amd/mma_trace.raw",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or (
        repo / "results" / f"{datetime.now():%Y%m%d_%H%M%S}_slot3_mma_trace_cross_vendor"
    )
    try:
        receipt = process_archive(args.archive, args.amd_trace, output)
    except (FileNotFoundError, FileExistsError, ValueError, zipfile.BadZipFile,
            json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2))
    print(f"Result: {output.resolve()}")
    return 0 if receipt["input_fragment_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
