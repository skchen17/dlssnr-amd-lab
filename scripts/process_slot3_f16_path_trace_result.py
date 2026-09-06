#!/usr/bin/env python3
"""Safely validate and compare a returned RTX slot-3 packed-FP16 trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
import zipfile
from datetime import datetime
from pathlib import Path

try:
    from scripts.instrument_slot3_f16_path_trace import LANES, REGISTERS
    from scripts.process_slot3_mma_trace_result import (
        archive_sha256,
        normalized_member,
    )
except ModuleNotFoundError:
    from instrument_slot3_f16_path_trace import LANES, REGISTERS
    from process_slot3_mma_trace_result import archive_sha256, normalized_member


TRACE_BYTES = LANES * len(REGISTERS) * 4
EXPECTED_EXPERIMENT = "rtx5070_slot3_f16_path_register_trace"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def inspect_archive(zf: zipfile.ZipFile) -> tuple[zipfile.ZipInfo, zipfile.ZipInfo]:
    infos = zf.infolist()
    if not infos or len(infos) > 32:
        raise ValueError(f"unexpected ZIP member count: {len(infos)}")
    if sum(info.file_size for info in infos) > 256 * 1024 * 1024:
        raise ValueError("ZIP uncompressed payload exceeds safety limit")
    seen: set[str] = set()
    files = []
    for info in infos:
        path = normalized_member(info.filename)
        key = path.as_posix().casefold()
        if key in seen:
            raise ValueError(f"duplicate ZIP member rejected: {info.filename!r}")
        seen.add(key)
        if not info.is_dir():
            files.append((path, info))
    manifests = [(path, info) for path, info in files if path.name == "manifest.json"]
    traces = [(path, info) for path, info in files if path.name == "f16_path_trace.raw"]
    if len(manifests) != 1 or len(traces) != 1:
        raise ValueError("archive must contain exactly one manifest and F16 trace")
    manifest_path, manifest_info = manifests[0]
    trace_path, trace_info = traces[0]
    if manifest_path.parent != trace_path.parent or len(manifest_path.parts) != 2:
        raise ValueError("manifest and trace must share one top-level result directory")
    if trace_info.file_size != TRACE_BYTES:
        raise ValueError(f"F16 trace must be {TRACE_BYTES} bytes")
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
        "cta": manifest.get("cta") == [2, 0, 0],
        "registers": manifest.get("registers") == REGISTERS,
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


def compare(rtx: bytes, amd: bytes) -> dict:
    if len(rtx) != TRACE_BYTES or len(amd) != TRACE_BYTES:
        raise ValueError(f"trace size must be {TRACE_BYTES} bytes")
    rtx_words = struct.unpack(f"<{TRACE_BYTES // 4}I", rtx)
    amd_words = struct.unpack(f"<{TRACE_BYTES // 4}I", amd)
    per_register = []
    first = None
    for index, register in enumerate(REGISTERS):
        mismatches = []
        for lane in range(LANES):
            offset = lane * len(REGISTERS) + index
            if rtx_words[offset] != amd_words[offset]:
                mismatches.append({
                    "lane": lane,
                    "rtx_u32": f"0x{rtx_words[offset]:08X}",
                    "amd_u32": f"0x{amd_words[offset]:08X}",
                })
                if first is None:
                    first = {"register": register, "register_index": index,
                             **mismatches[-1]}
        per_register.append({
            "register": register,
            "mismatch_lanes": len(mismatches),
            "examples": mismatches[:8],
        })
    total = sum(item["mismatch_lanes"] for item in per_register)
    return {
        "schema": 1,
        "experiment": "slot3_f16_path_rtx5070_vs_rx9070xt",
        "status": "PASS",
        "classification": "FIRST_PACKED_F16_PATH_DIVERGENCE_LOCALIZATION",
        "rtx_sha256": sha256(rtx),
        "amd_sha256": sha256(amd),
        "bitwise_equal": total == 0,
        "word_mismatches": total,
        "first_mismatch": first,
        "per_register": per_register,
    }


def process_archive(archive: Path, amd_trace: Path, output: Path) -> dict:
    archive = archive.resolve(strict=True)
    amd_trace = amd_trace.resolve(strict=True)
    amd = amd_trace.read_bytes()
    if len(amd) != TRACE_BYTES:
        raise ValueError(f"AMD trace must be {TRACE_BYTES} bytes")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    with zipfile.ZipFile(archive, "r") as zf:
        manifest_info, trace_info = inspect_archive(zf)
        manifest = json.loads(zf.read(manifest_info).decode("utf-8-sig"))
        rtx = zf.read(trace_info)
    require_manifest(manifest, rtx)
    output.mkdir(parents=True)
    (output / "rtx_f16_path_trace.raw").write_bytes(rtx)
    shutil.copyfile(amd_trace, output / "amd_f16_path_trace.raw")
    (output / "rtx_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    comparison = compare(rtx, amd)
    (output / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    receipt = {
        "schema": 1,
        "experiment": "slot3_f16_path_trace_result_ingestion",
        "status": "PASS",
        "source_archive": str(archive),
        "source_archive_sha256": archive_sha256(archive),
        "rtx_device_name": manifest["device_name"],
        "cta": manifest["cta"],
        "rtx_trace_sha256": sha256(rtx),
        "amd_trace_sha256": sha256(amd),
        "bitwise_equal": comparison["bitwise_equal"],
        "word_mismatches": comparison["word_mismatches"],
        "first_mismatch": comparison["first_mismatch"],
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--amd-trace", type=Path, default=
                        repo / "results/20260901_040000_slot3_cta2_f16_path_trace/amd/f16_path_trace.raw")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or repo / "results" / f"{datetime.now():%Y%m%d_%H%M%S}_slot3_f16_path_cross_vendor"
    try:
        receipt = process_archive(args.archive, args.amd_trace, output)
    except (FileNotFoundError, FileExistsError, ValueError, zipfile.BadZipFile,
            json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2))
    print(f"Result: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
