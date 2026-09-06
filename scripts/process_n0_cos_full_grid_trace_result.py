#!/usr/bin/env python3
"""Validate and ingest the RTX/RX full-grid N0 cosine approximation dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
import zipfile
from collections import Counter
from pathlib import Path


SAMPLES = 80 * 48 * 64
BYTES_PER_SAMPLE = 8
TRACE_BYTES = SAMPLES * BYTES_PER_SAMPLE


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def ordered(bits: int) -> int:
    return 0x80000000 - bits if bits & 0x80000000 else 0x80000000 + bits


def f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def process(archive: Path, amd_path: Path, output: Path) -> dict:
    archive = archive.resolve(strict=True)
    amd_path = amd_path.resolve(strict=True)
    amd = amd_path.read_bytes()
    if len(amd) != TRACE_BYTES:
        raise ValueError(f"AMD trace must be {TRACE_BYTES} bytes")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    with zipfile.ZipFile(archive) as zf:
        manifests = [i for i in zf.infolist() if Path(i.filename).name == "manifest.json"]
        traces = [i for i in zf.infolist() if Path(i.filename).name == "cos_trace.raw"]
        if len(manifests) != 1 or len(traces) != 1 or traces[0].file_size != TRACE_BYTES:
            raise ValueError("archive layout or trace size is invalid")
        manifest = json.loads(zf.read(manifests[0]).decode("utf-8-sig"))
        rtx = zf.read(traces[0])
    checks = {
        "experiment": manifest.get("experiment") == "rtx_n0_cos_full_grid_trace",
        "status": manifest.get("status") == "PASS",
        "integrity": manifest.get("payload_integrity") is True,
        "device": isinstance(manifest.get("device_name"), str)
        and "NVIDIA" in manifest["device_name"].upper(),
        "launch": manifest.get("grid") == [80, 48, 1]
        and manifest.get("block") == [32, 1, 1]
        and manifest.get("kernel_launched") is True,
        "shape": manifest.get("sample_count") == SAMPLES
        and manifest.get("bytes_per_sample") == BYTES_PER_SAMPLE
        and manifest.get("checkpoint_bytes") == TRACE_BYTES,
        "hash": str(manifest.get("checkpoint_sha256", "")).upper() == sha256(rtx),
        "preserved": manifest.get("reference_output_preserved") is True
        and manifest.get("instrumentation_perturbed") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("manifest validation failed: " + ", ".join(failed))

    input_mismatches = 0
    output_mismatches = 0
    first_output = None
    deltas: Counter[int] = Counter()
    max_abs_error = 0.0
    sum_abs_error = 0.0
    unique_inputs = set()
    for index in range(SAMPLES):
        ri, ro = struct.unpack_from("<II", rtx, index * BYTES_PER_SAMPLE)
        ai, ao = struct.unpack_from("<II", amd, index * BYTES_PER_SAMPLE)
        unique_inputs.add(ri)
        if ri != ai:
            input_mismatches += 1
        if ro != ao:
            output_mismatches += 1
            if first_output is None:
                first_output = index
            deltas[ordered(ro) - ordered(ao)] += 1
            error = abs(f32(ro) - f32(ao))
            max_abs_error = max(max_abs_error, error)
            sum_abs_error += error
    comparison = {
        "schema": 1,
        "experiment": "n0_cos_full_grid_rtx5070_vs_rx9070xt",
        "classification": "BITWISE_IDENTICAL_INPUTS_ARCHITECTURE_SPECIFIC_COS_APPROXIMATION",
        "sample_count": SAMPLES,
        "unique_input_count": len(unique_inputs),
        "input_mismatch_samples": input_mismatches,
        "output_mismatch_samples": output_mismatches,
        "output_exact_fraction": 1.0 - output_mismatches / SAMPLES,
        "first_output_mismatch_sample": first_output,
        "mean_abs_error_over_mismatches": sum_abs_error / output_mismatches,
        "max_abs_error": max_abs_error,
        "ordered_ulp_delta_min": min(deltas),
        "ordered_ulp_delta_max": max(deltas),
        "most_common_ordered_ulp_deltas": [
            {"delta": delta, "samples": count} for delta, count in deltas.most_common(16)
        ],
        "rtx_sha256": sha256(rtx),
        "amd_sha256": sha256(amd),
    }
    output.mkdir(parents=True)
    (output / "rtx_cos_trace.raw").write_bytes(rtx)
    shutil.copyfile(amd_path, output / "amd_cos_trace.raw")
    (output / "rtx_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "schema": 1,
        "experiment": "n0_cos_full_grid_trace_result_ingestion",
        "status": "PASS",
        "source_archive": str(archive),
        "source_archive_sha256": sha256(archive.read_bytes()),
        "rtx_device_name": manifest["device_name"],
        "reference_output_preserved": True,
        **comparison,
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--amd-trace",
        type=Path,
        default=repo / "results" / "20260904_010000_n0_cos_full_grid_trace" / "amd_cos_trace.raw",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = process(args.archive, args.amd_trace, args.output)
    except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    print(f"Result: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
