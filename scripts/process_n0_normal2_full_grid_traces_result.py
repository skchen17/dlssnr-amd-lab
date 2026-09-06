#!/usr/bin/env python3
"""Validate and compare RTX/RX full-grid second Box-Muller traces."""

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


def process(archive: Path, amd_root: Path, output: Path) -> dict:
    archive = archive.resolve(strict=True)
    amd_root = amd_root.resolve(strict=True)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    amd_cos_path = amd_root / "amd_cos1_trace.raw"
    amd_boundary_path = amd_root / "amd_normal2_boundary_trace.raw"
    amd_cos = amd_cos_path.read_bytes()
    amd_boundary = amd_boundary_path.read_bytes()
    if len(amd_cos) != TRACE_BYTES or len(amd_boundary) != TRACE_BYTES:
        raise ValueError("AMD trace size invalid")

    with zipfile.ZipFile(archive) as zf:
        files = [(safe_name(item.filename), item) for item in zf.infolist() if not item.is_dir()]
        if not files or len(files) > 16 or sum(item.file_size for _, item in files) > 5 * 1024 * 1024:
            raise ValueError("invalid ZIP size/count")

        def one(name: str) -> zipfile.ZipInfo:
            matches = [item for path, item in files if path.name == name]
            if len(matches) != 1:
                raise ValueError(f"expected one {name}")
            return matches[0]

        manifest = json.loads(zf.read(one("manifest.json")).decode("utf-8-sig"))
        rtx_cos = zf.read(one("cos1_trace.raw"))
        rtx_boundary = zf.read(one("normal2_boundary_trace.raw"))

    variants = manifest.get("variants")
    if not isinstance(variants, list):
        raise ValueError("manifest variants missing")
    by_name = {row.get("name"): row for row in variants if isinstance(row, dict)}
    cos_row = by_name.get("cos1", {})
    boundary_row = by_name.get("normal2_boundary", {})
    expected_fields = {
        "cos1": ["phase1_f32", "cos1_f32"],
        "normal2_boundary": ["sqrt1_f32", "normal2_f16_as_u32"],
    }
    variant_checks = []
    for name, row, data, filename in (
        ("cos1", cos_row, rtx_cos, "cos1_trace.raw"),
        ("normal2_boundary", boundary_row, rtx_boundary, "normal2_boundary_trace.raw"),
    ):
        variant_checks.append(
            row.get("fields") == expected_fields[name]
            and row.get("probe_exit") == 0
            and row.get("probe_pass") is True
            and row.get("execution_verified") is True
            and row.get("reference_output_preserved") is True
            and row.get("output_sha256") == manifest.get("baseline_output_sha256")
            and row.get("trace_filename") == filename
            and str(row.get("trace_sha256", "")).upper() == sha256(data)
        )
    checks = {
        "experiment": manifest.get("experiment") == "rtx_n0_normal2_full_grid_traces",
        "status": manifest.get("status") == "PASS",
        "integrity": manifest.get("payload_integrity") is True,
        "device": isinstance(manifest.get("device_name"), str)
        and "NVIDIA" in manifest["device_name"].upper(),
        "launch": manifest.get("grid") == [80, 48, 1]
        and manifest.get("block") == [32, 1, 1],
        "shape": manifest.get("sample_count") == SAMPLES
        and manifest.get("bytes_per_sample") == BYTES_PER_SAMPLE
        and manifest.get("trace_bytes_per_variant") == TRACE_BYTES
        and len(rtx_cos) == TRACE_BYTES
        and len(rtx_boundary) == TRACE_BYTES,
        "variants": len(variants) == 2 and all(variant_checks),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("manifest validation failed: " + ", ".join(failed))

    phase_mismatches = cos_mismatches = sqrt_mismatches = normal_mismatches = 0
    rtx_prediction_mismatches = amd_prediction_mismatches = 0
    rtx_sqrt_amd_cos_mismatches = amd_sqrt_rtx_cos_mismatches = 0
    categories: Counter[str] = Counter()
    first = []
    for sample in range(SAMPLES):
        rphase, rcos = struct.unpack_from("<II", rtx_cos, sample * BYTES_PER_SAMPLE)
        aphase, acos = struct.unpack_from("<II", amd_cos, sample * BYTES_PER_SAMPLE)
        rsqrt, rnormal_word = struct.unpack_from("<II", rtx_boundary, sample * BYTES_PER_SAMPLE)
        asqrt, anormal_word = struct.unpack_from("<II", amd_boundary, sample * BYTES_PER_SAMPLE)
        if rnormal_word >> 16 or anormal_word >> 16:
            raise ValueError(f"nonzero upper half at sample {sample}")
        rnormal, anormal = rnormal_word & 0xFFFF, anormal_word & 0xFFFF
        phase_equal = rphase == aphase
        cos_equal = rcos == acos
        sqrt_equal = rsqrt == asqrt
        if not phase_equal:
            phase_mismatches += 1
        if not cos_equal:
            cos_mismatches += 1
        if not sqrt_equal:
            sqrt_mismatches += 1
        if predicted_half(rsqrt, rcos) != rnormal:
            rtx_prediction_mismatches += 1
        if predicted_half(asqrt, acos) != anormal:
            amd_prediction_mismatches += 1
        if predicted_half(rsqrt, acos) != rnormal:
            rtx_sqrt_amd_cos_mismatches += 1
        if predicted_half(asqrt, rcos) != rnormal:
            amd_sqrt_rtx_cos_mismatches += 1
        if rnormal != anormal:
            normal_mismatches += 1
            categories[f"sqrt_{'equal' if sqrt_equal else 'different'}__cos_{'equal' if cos_equal else 'different'}"] += 1
            if len(first) < 64:
                linear_cta, local = divmod(sample, 64)
                cta_y, cta_x = divmod(linear_cta, 80)
                first.append({
                    "sample": sample,
                    "cta": [cta_x, cta_y, 0],
                    "local_sample": local,
                    "rtx_phase_u32": f"0x{rphase:08X}",
                    "amd_phase_u32": f"0x{aphase:08X}",
                    "rtx_cos_u32": f"0x{rcos:08X}",
                    "amd_cos_u32": f"0x{acos:08X}",
                    "rtx_sqrt_u32": f"0x{rsqrt:08X}",
                    "amd_sqrt_u32": f"0x{asqrt:08X}",
                    "rtx_normal_f16": f"0x{rnormal:04X}",
                    "amd_normal_f16": f"0x{anormal:04X}",
                    "rtx_sqrt_amd_cos_f16": f"0x{predicted_half(rsqrt, acos):04X}",
                    "amd_sqrt_rtx_cos_f16": f"0x{predicted_half(asqrt, rcos):04X}",
                })

    comparison = {
        "schema": 1,
        "experiment": "n0_normal2_full_grid_rtx5070_vs_rx9070xt",
        "status": "PASS",
        "classification": "SECOND_BOX_MULLER_CONSUMED_FP16_BOUNDARY",
        "sample_count": SAMPLES,
        "phase1_mismatch_samples": phase_mismatches,
        "cos1_mismatch_samples": cos_mismatches,
        "sqrt1_mismatch_samples": sqrt_mismatches,
        "normal2_f16_mismatch_samples": normal_mismatches,
        "normal2_f16_exact_fraction": 1.0 - normal_mismatches / SAMPLES,
        "normal_mismatch_categories": dict(sorted(categories.items())),
        "rtx_rounded_stage_reconstruction_mismatches": rtx_prediction_mismatches,
        "amd_rounded_stage_reconstruction_mismatches": amd_prediction_mismatches,
        "ideal_rtx_sqrt_with_amd_cos_mismatch_samples": rtx_sqrt_amd_cos_mismatches,
        "ideal_amd_sqrt_with_rtx_cos_mismatch_samples": amd_sqrt_rtx_cos_mismatches,
        "first_normal_mismatches": first,
        "rtx_cos1_trace_sha256": sha256(rtx_cos),
        "amd_cos1_trace_sha256": sha256(amd_cos),
        "rtx_boundary_trace_sha256": sha256(rtx_boundary),
        "amd_boundary_trace_sha256": sha256(amd_boundary),
    }
    output.mkdir(parents=True)
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    (output / "rtx_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "rtx_cos1_trace.raw").write_bytes(rtx_cos)
    (output / "rtx_normal2_boundary_trace.raw").write_bytes(rtx_boundary)
    shutil.copyfile(amd_cos_path, output / "amd_cos1_trace.raw")
    shutil.copyfile(amd_boundary_path, output / "amd_normal2_boundary_trace.raw")
    receipt = {
        "schema": 1,
        "experiment": "n0_normal2_full_grid_traces_result_ingestion",
        "status": "PASS",
        "source_archive": str(archive),
        "source_archive_sha256": sha256(archive.read_bytes()),
        "rtx_device_name": manifest["device_name"],
        "reference_outputs_preserved": True,
        **comparison,
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--amd-root",
        type=Path,
        default=repo / "results/20260904_160000_n0_normal2_full_grid_traces",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = process(args.archive, args.amd_root, args.output)
    except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError, struct.error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    print(f"Result: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
