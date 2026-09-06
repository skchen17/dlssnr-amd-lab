#!/usr/bin/env python3
"""Validate and compare a returned RTX slot-3 approximate-rsqrt trace."""

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
    from scripts.instrument_slot3_rsqrt_trace import (
        EXPECTED_BLOCKS, LANES, WORDS_PER_BLOCK,
    )
    from scripts.process_slot3_mma_trace_result import archive_sha256, normalized_member
except ModuleNotFoundError:
    from instrument_slot3_rsqrt_trace import EXPECTED_BLOCKS, LANES, WORDS_PER_BLOCK
    from process_slot3_mma_trace_result import archive_sha256, normalized_member


TRACE_BYTES = LANES * EXPECTED_BLOCKS * WORDS_PER_BLOCK * 4
EXPECTED_EXPERIMENT = "rtx5070_slot3_rsqrt_register_trace"
FIELDS = ["input_f16x2", "result_low_f32", "result_high_f32", "output_f16x2"]


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
    traces = [(path, info) for path, info in files if path.name == "rsqrt_trace.raw"]
    if len(manifests) != 1 or len(traces) != 1:
        raise ValueError("archive must contain exactly one manifest and rsqrt trace")
    mp, mi = manifests[0]
    tp, ti = traces[0]
    if mp.parent != tp.parent or len(mp.parts) != 2:
        raise ValueError("manifest and trace must share one top-level result directory")
    if ti.file_size != TRACE_BYTES:
        raise ValueError(f"rsqrt trace must be {TRACE_BYTES} bytes")
    return mi, ti


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
        "blocks": isinstance(manifest.get("blocks"), list)
        and len(manifest["blocks"]) == EXPECTED_BLOCKS,
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


def compare(rtx: bytes, amd: bytes, blocks: list[dict]) -> dict:
    if len(rtx) != TRACE_BYTES or len(amd) != TRACE_BYTES:
        raise ValueError(f"trace size must be {TRACE_BYTES} bytes")
    rw = struct.unpack(f"<{TRACE_BYTES // 4}I", rtx)
    aw = struct.unpack(f"<{TRACE_BYTES // 4}I", amd)
    reports = []
    first = None
    total = 0
    for block in range(EXPECTED_BLOCKS):
        counts = {field: 0 for field in FIELDS}
        examples = []
        for lane in range(LANES):
            base = (lane * EXPECTED_BLOCKS + block) * WORDS_PER_BLOCK
            for field_index, field in enumerate(FIELDS):
                offset = base + field_index
                if rw[offset] != aw[offset]:
                    counts[field] += 1
                    total += 1
                    item = {
                        "lane": lane, "field": field,
                        "rtx_u32": f"0x{rw[offset]:08X}",
                        "amd_u32": f"0x{aw[offset]:08X}",
                    }
                    if first is None:
                        first = {"block": block, **item}
                    if len(examples) < 8:
                        examples.append(item)
        reports.append({
            "block": block,
            "source": blocks[block].get("source"),
            "destination": blocks[block].get("destination"),
            "mismatch_lanes_by_field": counts,
            "examples": examples,
        })
    return {
        "schema": 1,
        "experiment": "slot3_rsqrt_rtx5070_vs_rx9070xt",
        "status": "PASS",
        "classification": "APPROX_RSQRT_SEMANTIC_LOCALIZATION",
        "rtx_sha256": sha256(rtx), "amd_sha256": sha256(amd),
        "bitwise_equal": total == 0, "word_mismatches": total,
        "first_mismatch": first, "per_block": reports,
    }


def process_archive(archive: Path, amd_trace: Path, output: Path) -> dict:
    archive = archive.resolve(strict=True)
    amd_trace = amd_trace.resolve(strict=True)
    amd = amd_trace.read_bytes()
    if len(amd) != TRACE_BYTES:
        raise ValueError(f"AMD trace must be {TRACE_BYTES} bytes")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    with zipfile.ZipFile(archive) as zf:
        mi, ti = inspect_archive(zf)
        manifest = json.loads(zf.read(mi).decode("utf-8-sig"))
        rtx = zf.read(ti)
    require_manifest(manifest, rtx)
    output.mkdir(parents=True)
    (output / "rtx_rsqrt_trace.raw").write_bytes(rtx)
    shutil.copyfile(amd_trace, output / "amd_rsqrt_trace.raw")
    (output / "rtx_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    comparison = compare(rtx, amd, manifest["blocks"])
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "schema": 1, "experiment": "slot3_rsqrt_trace_result_ingestion",
        "status": "PASS", "source_archive": str(archive),
        "source_archive_sha256": archive_sha256(archive),
        "rtx_device_name": manifest["device_name"], "cta": manifest["cta"],
        "rtx_trace_sha256": sha256(rtx), "amd_trace_sha256": sha256(amd),
        "bitwise_equal": comparison["bitwise_equal"],
        "word_mismatches": comparison["word_mismatches"],
        "first_mismatch": comparison["first_mismatch"],
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--amd-trace", type=Path, default=repo / "results/20260901_060000_slot3_cta2_rsqrt_trace/amd/rsqrt_trace.raw")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or repo / "results" / f"{datetime.now():%Y%m%d_%H%M%S}_slot3_rsqrt_cross_vendor"
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
