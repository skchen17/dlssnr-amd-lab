#!/usr/bin/env python3
"""Validate and join the RTX/RX N0 normal0 full-grid boundary traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath


SAMPLES = 80 * 48 * 64
BYTES_PER_SAMPLE = 8
TRACE_BYTES = SAMPLES * BYTES_PER_SAMPLE


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def safe_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe ZIP member: {name}")
    return path


def f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def rounded_f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def half_bits(value: float) -> int:
    return struct.unpack("<H", struct.pack("<e", value))[0]


def predicted_half(sqrt_bits: int, cos_bits: int) -> int:
    return half_bits(rounded_f32(f32(sqrt_bits) * f32(cos_bits)))


def process(archive: Path, amd_trace: Path, cos_root: Path, output: Path) -> dict:
    archive = archive.resolve(strict=True)
    amd_trace = amd_trace.resolve(strict=True)
    cos_root = cos_root.resolve(strict=True)
    amd = amd_trace.read_bytes()
    if len(amd) != TRACE_BYTES:
        raise ValueError(f"AMD trace must be {TRACE_BYTES} bytes")
    if output.exists():
        raise FileExistsError(output)

    with zipfile.ZipFile(archive) as zf:
        files = [(safe_name(item.filename), item) for item in zf.infolist() if not item.is_dir()]
        if not files or len(files) > 16 or sum(item.file_size for _, item in files) > 4 * 1024 * 1024:
            raise ValueError("invalid ZIP size/count")

        def one(name: str) -> zipfile.ZipInfo:
            matches = [item for path, item in files if path.name == name]
            if len(matches) != 1:
                raise ValueError(f"expected one {name}")
            return matches[0]

        manifest = json.loads(zf.read(one("manifest.json")).decode("utf-8-sig"))
        rtx = zf.read(one("normal0_boundary_trace.raw"))

    checks = {
        "experiment": manifest.get("experiment") == "rtx_n0_normal0_boundary_full_grid_trace",
        "status": manifest.get("status") == "PASS",
        "integrity": manifest.get("payload_integrity") is True,
        "device": isinstance(manifest.get("device_name"), str)
        and "NVIDIA" in manifest["device_name"].upper(),
        "launch": manifest.get("grid") == [80, 48, 1]
        and manifest.get("block") == [32, 1, 1],
        "shape": manifest.get("sample_count") == SAMPLES
        and manifest.get("bytes_per_sample") == BYTES_PER_SAMPLE
        and manifest.get("checkpoint_bytes") == TRACE_BYTES
        and len(rtx) == TRACE_BYTES,
        "fields": manifest.get("fields") == ["sqrt0_f32", "normal0_f16_as_u32"],
        "hash": str(manifest.get("checkpoint_sha256", "")).upper() == sha256(rtx),
        "preserved": manifest.get("reference_output_preserved") is True
        and manifest.get("instrumentation_perturbed") is False
        and manifest.get("output_sha256") == manifest.get("baseline_output_sha256"),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("manifest validation failed: " + ", ".join(failed))

    rtx_cos = (cos_root / "rtx_cos_trace.raw").read_bytes()
    amd_cos = (cos_root / "amd_cos_trace.raw").read_bytes()
    if len(rtx_cos) != TRACE_BYTES or len(amd_cos) != TRACE_BYTES:
        raise ValueError("cosine trace size invalid")

    sqrt_mismatches = normal_mismatches = 0
    rtx_prediction_mismatches = amd_prediction_mismatches = 0
    cosine_input_mismatches = 0
    categories: Counter[str] = Counter()
    first = []
    for sample in range(SAMPLES):
        rs, rn_word = struct.unpack_from("<II", rtx, sample * BYTES_PER_SAMPLE)
        amd_sqrt, an_word = struct.unpack_from("<II", amd, sample * BYTES_PER_SAMPLE)
        ri, rc = struct.unpack_from("<II", rtx_cos, sample * BYTES_PER_SAMPLE)
        ai, ac = struct.unpack_from("<II", amd_cos, sample * BYTES_PER_SAMPLE)
        rn, an = rn_word & 0xFFFF, an_word & 0xFFFF
        if rn_word >> 16 or an_word >> 16:
            raise ValueError(f"nonzero upper half at sample {sample}")
        sqrt_equal = rs == amd_sqrt
        cos_input_equal = ri == ai
        cos_equal = rc == ac
        if not sqrt_equal:
            sqrt_mismatches += 1
        if not cos_input_equal:
            cosine_input_mismatches += 1
        if predicted_half(rs, rc) != rn:
            rtx_prediction_mismatches += 1
        if predicted_half(amd_sqrt, ac) != an:
            amd_prediction_mismatches += 1
        if rn != an:
            normal_mismatches += 1
            key = f"sqrt_{'equal' if sqrt_equal else 'different'}__cos_{'equal' if cos_equal else 'different'}"
            categories[key] += 1
            if len(first) < 64:
                linear_cta, local = divmod(sample, 64)
                cta_y, cta_x = divmod(linear_cta, 80)
                first.append({
                    "sample": sample,
                    "cta": [cta_x, cta_y, 0],
                    "local_sample": local,
                    "rtx_sqrt_u32": f"0x{rs:08X}",
                    "amd_sqrt_u32": f"0x{amd_sqrt:08X}",
                    "rtx_cos_u32": f"0x{rc:08X}",
                    "amd_cos_u32": f"0x{ac:08X}",
                    "rtx_normal_f16": f"0x{rn:04X}",
                    "amd_normal_f16": f"0x{an:04X}",
                })

    comparison = {
        "schema": 1,
        "experiment": "n0_normal0_boundary_full_grid_rtx5070_vs_rx9070xt",
        "status": "PASS",
        "classification": "FULL_GRID_CONSUMED_FP16_APPROXIMATION_BOUNDARY",
        "sample_count": SAMPLES,
        "sqrt0_mismatch_samples": sqrt_mismatches,
        "normal0_f16_mismatch_samples": normal_mismatches,
        "normal0_f16_exact_fraction": 1.0 - normal_mismatches / SAMPLES,
        "cosine_input_mismatch_samples": cosine_input_mismatches,
        "normal_mismatch_categories": dict(sorted(categories.items())),
        "rtx_rounded_stage_reconstruction_mismatches": rtx_prediction_mismatches,
        "amd_rounded_stage_reconstruction_mismatches": amd_prediction_mismatches,
        "first_normal_mismatches": first,
        "rtx_trace_sha256": sha256(rtx),
        "amd_trace_sha256": sha256(amd),
    }
    output.mkdir(parents=True)
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    (output / "rtx_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "rtx_normal0_boundary_trace.raw").write_bytes(rtx)
    shutil.copyfile(amd_trace, output / "amd_normal0_boundary_trace.raw")
    receipt = {
        "schema": 1,
        "experiment": "n0_normal0_boundary_full_grid_result_ingestion",
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
        default=repo / "results/20260904_100000_n0_normal0_boundary_full_grid_trace/amd_boundary_trace.raw",
    )
    parser.add_argument(
        "--cos-root",
        type=Path,
        default=repo / "results/20260904_020000_n0_cos_full_grid_cross_vendor",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = process(args.archive, args.amd_trace, args.cos_root, args.output)
    except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError, struct.error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    print(f"Result: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
